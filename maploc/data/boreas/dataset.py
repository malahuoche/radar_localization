# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import collections.abc
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from ... import logger, DATASETS_PATH
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn
from .utils import parse_split_file, parse_gps_file,parse_combined_file, get_camera_calibration
import cv2
class BoreasDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "boreas",
        # paths and fetch
        "data_dir": "/home/classlab2/16T/datasets/boreas.",
        "tiles_filename": "boreastiles.pkl",
        "splits": {
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        "max_init_error_rotation": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "boreas."

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)#"/home/classlab2/radiate"
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        if not (self.root.exists() and (self.root / ".downloaded").exists()):
            # raise FileNotFoundError(
            #     "Cannot find the KITTI dataset, run maploc.data.kitti.prepare"
            # )
            print("1")
            print("1")

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            names, shifts = parse_split_file(self.root / split_arg)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _, _ in ns}
        # print(dates)
        for d in dates:
            #print(d)
            self.calibrations[d] = get_camera_calibration(
                self.root/ d /"calib")#radiate不需要相机index
            #print(self.calibrations[d])
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            # print(stage)
            names = []
            data = {}
            for i, (date, drive, mask_cart, index) in enumerate(self.splits[stage]):
                # print(index)
                d = self.get_frame_data(date, drive, index)#data的数据t_c2w、roll_pitch_yaw,index
                # print(d)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/{mask_cart}/{index}"#要修改路径
                # print(path)
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))#要修改路径
                # print((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            shifts = self.shifts[stage]
            if shifts is not None:
                data["shifts"] = torch.from_numpy(shifts.astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

   
    def get_frame_data(self, date, drive, index):


        _, R_cam_gps, t_cam_gps = self.calibrations[date]
        # Transform the GPS pose to the camera pose
        # 根据radar图像的index找到对应的gps数据
        
        gps_path = (
            self.root / date / "gps" / Path(index).with_suffix(".txt")
        )
        with open(gps_path, 'r') as file:
        # 读取文件的一行
            line = file.readline()
            # 提取一行的第三列的数字
            yaw_gps = float(line.split()[2])

        # gps_list = parse_combined_file(self.root, date, index)
        # #提取utm坐标系下的radar点
        # resolution = 0.2384  # m/pixel
        # # 使用提供的函数提取亮点的本地坐标
        # cartesian_image_path = self.root / date / "radar/cart_orignal" / Path(index).with_suffix(".txt")
        # cartesian_image = cv2.imread(cartesian_image_path)
        # point_cloud_local = extract_local_coordinates(cartesian_image, resolution)
        # points_local = point_cloud_local.T
        # gps_utm = [utm.from_latlon(lat, lon) for lat, lon in gps_list]
        # print("utm",gps_utm)
        # yaw = yaw_gps
        # tInd = 1
        # points_utm = points_local_to_utm(gps_list, tInd, points_local)
        # utm_x = points_utm[0, :]
        # utm_y = points_utm[1, :]
        # gps_utm_array = np.array(gps_utm)

        # # Calculate coordinate difference, assuming the rotation center is gps_utm
        # coord_diff = np.array(points_utm) - gps_utm_array.reshape(-1, 1)
        # rotated_coord_diff = rotate_points(coord_diff, yaw)
        # # 将旋转后的坐标差异加回到原始雷达点云的UTM坐标上
        # points_utm_rotated = gps_utm_array.reshape(-1, 1) + rotated_coord_diff


        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw_gps], np.float32)  # for some reason
        # print("yawyaw:",yaw)
        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
            # "radar_point":points_utm,
            # "radar_rotated": points_utm_rotated,
            "index": int(index.split(".")[0]),
        } 

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.image_paths[stage]
        # group images by sequence (date/drive)
        seq2indices = defaultdict(list)
        for index, (_, date_drive, _) in enumerate(keys):
            seq2indices[date_drive].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(
                self.data[stage], indices, names=self.image_paths[stage], **kwargs
            )
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, seq_keys, chunk2idx
