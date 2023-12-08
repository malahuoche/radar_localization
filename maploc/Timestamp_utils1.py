import os
import re
directory_path = "/home/classlab2/16T/datasets/boreas./boreas-2021-01-15-12-17/radar/cart"

# 获取目录中所有文件的文件名
file_names = [filename for filename in os.listdir(directory_path)
              if os.path.isfile(os.path.join(directory_path, filename))]

# 提取文件名中的时间戳部分
files_with_timestamp = [(filename, re.search(r'\d+', filename).group())
                        for filename in file_names if re.search(r'\d+', filename)]

# 按照时间戳排序文件
files_with_timestamp.sort(key=lambda x: x[1])
base_directory = "/home/classlab2/16T/datasets/boreas."
output_file_path = os.path.join(base_directory,'boreas-2021-01-15-12-17', 'combined.txt')
# # 打印按照时间戳排序后的文件名
# for filename, timestamp in files_with_timestamp:
#     name, extension = os.path.splitext(filename)
#     txt_path = os.path.join(base_directory,'boreas-2021-01-15-12-17', 'gps', f'{timestamp}.txt')
#     if os.path.isfile(txt_path):
#         with open(txt_path, 'r') as txt_file:
#             # 读取txt文件的内容
#             gps_info = txt_file.read()
#             # 创建新的txt文件并写入时间戳和gps信息
#             with open(output_file_path, 'a') as output_file:
#                 output_file.write(f'{timestamp} {gps_info}\n')
#             print(f'Combined file for {timestamp} created at {output_file_path}')
#     else:
#         print(f'Txt file not found for timestamp: {timestamp}')
# 打印按照时间戳排序后的文件名
for i in range(len(files_with_timestamp) - 1):
    current_filename, current_timestamp = files_with_timestamp[i]
    next_filename, next_timestamp = files_with_timestamp[i + 1]

    current_txt_path = os.path.join(base_directory, 'boreas-2021-01-15-12-17', 'gps', f'{current_timestamp}.txt')
    next_txt_path = os.path.join(base_directory, 'boreas-2021-01-15-12-17', 'gps', f'{next_timestamp}.txt')

    if os.path.isfile(current_txt_path) and os.path.isfile(next_txt_path):
        with open(current_txt_path, 'r') as current_txt_file, open(next_txt_path, 'r') as next_txt_file:
            # 读取当前和下一个txt文件的内容
            current_gps_info = current_txt_file.read()
            next_gps_info = next_txt_file.read()

            # 创建新的txt文件并写入当前和下一个时间戳以及gps信息
            with open(output_file_path, 'a') as output_file:
                output_file.write(f'{current_timestamp} {current_gps_info} {next_gps_info}\n')

            print(f'Combined files for {current_timestamp} and {next_timestamp} created at {output_file_path}')
    else:
        print(f'Txt files not found for timestamps: {current_timestamp}, {next_timestamp}')

