# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.nn.functional import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .voting import (
    argmax_xyr,
    argmax_xy,
    conv2d_fft_batchwise,
    expectation_xyr,
    expectation_xy_radar,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    nll_loss_xy,
    TemplateSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall


class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": True,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)
        ppm = conf.pixel_per_meter
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        # if conf.bev_net is None:
        #     self.feature_projection = torch.nn.Linear(
        #         conf.latent_dim, conf.matching_dim
        #     )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    def exhaustive_voting_radar(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)#[1,8,80,224]
            f_map = normalize(f_map, dim=1)#[1,8,256,256]

        # Build the templates and exhaustively match against the map.#这里不需要旋转
        # mask的部分后期再加
        # if confidence_bev is not None:
        #     f_bev = f_bev * confidence_bev.unsqueeze(1)
        # f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        # f_bev = f_bev.masked_fill(torch.logical_not(valid_bev.unsqueeze(1)), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        #模版和地图进行卷积（匹配）得到相似性得分
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels
        # in each template. Axis-aligned rotation have the maximum number of valid pixels.
        # valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        # num_valid = valid_templates.float().sum((-3, -2, -1))
        # scores = scores / num_valid[..., None, None]
        return scores


    def _forward(self, data):
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]#[1,8,256,256]

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level] #[1,128,80,224]
        f_bev = f_image
        #f_bev [1,8,80,224]
        #f_map [1,8,256,256]
        #scores [1,256,256,64]
        #scores [B,H,W,1]代表分数
        valid_bev=  torch.ones_like(f_bev)  # 一个虚拟的掩码，因为BEV被跳过了
        pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
        f_bev = pred_bev["output"]#[1,8,80,224]
        scores = self.exhaustive_voting_radar(
            f_bev, f_map,valid_bev
        )
        # scores = scores.squeeze(-1)
        scores = scores.moveaxis(1, -1)  # B,H,W,N
        # if "log_prior" in pred_map and self.conf.apply_map_prior:
        #     scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())
        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):
        xy_gt = data["uv"]
        yaw_gt = data["roll_pitch_yaw"][..., -1]
        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt,yaw_gt)
        loss = {"total": nll, "nll": nll}
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            # "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            # "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            # "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }