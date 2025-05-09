import h5py
from PIL import Image
import numpy as np
import os

# 定义输入和输出文件夹路径
input_folder = '/home/prejudice/datasets/ACDC/ACDC/data/slices'
output_folder = '/home/prejudice/datasets/ACDC_levelset/h52png'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 .h5 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.h5'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.h5', '.png'))

        # 打开 .h5 文件
        with h5py.File(input_path, 'r') as h5file:
            # 读取数据集
            data = h5file['image'][:]

        # 确保数据在 [0, 1] 范围内
        data = np.clip(data, 0, 1)

        # 转换为 8 位无符号整数
        uint8_data = (data * 255).astype(np.uint8)

        # 保存为 PNG 图像
        image = Image.fromarray(uint8_data)
        image.save(output_path)

        print(f"Saved {output_path}")
