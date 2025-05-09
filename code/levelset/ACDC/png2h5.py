import h5py
from PIL import Image
import numpy as np
import os

# 定义输入和输出文件夹路径
input_folder = '/home/prejudice/datasets/ACDC_levelset/png2bin/binarize'
output_folder = '/home/prejudice/datasets/ACDC_levelset/png2h5'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 .png 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)

        # 去掉前缀并更改扩展名
        output_filename = filename.replace('binarized_', '').replace('.png', '.h5')
        output_path = os.path.join(output_folder, output_filename)

        # 打开 .png 文件并转换为 NumPy 数组
        with Image.open(input_path) as img:
            data = np.array(img)

        # 创建 256x256 的图像，并填充为黑色
        padded_data = np.zeros((256, 256), dtype=np.uint8)

        # 将 150x150 的图像放置在 256x256 的中央
        start_x = (256 - 150) // 2
        start_y = (256 - 150) // 2
        padded_data[start_y:start_y + 150, start_x:start_x + 150] = data

        # 确保数据在 [0, 255] 范围内并转换为浮点型
        padded_data = padded_data.astype(np.float32) / 1.0

        # 创建 HDF5 文件并保存数据集
        with h5py.File(output_path, 'w') as h5file:
            h5file.create_dataset('mask_levelset', data=padded_data)

        print(f"Saved {output_path}")
