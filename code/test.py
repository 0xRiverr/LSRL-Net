# import os
# import h5py
#
# def get_relative_h5_files(directory, case_start, case_end):
#     """
#     获取从 case_start 到 case_end 范围内的所有 h5 文件的相对路径。
#
#     Args:
#         directory (str): h5 文件所在的目录。
#         case_start (int): 起始 Case 编号。
#         case_end (int): 结束 Case 编号（包括）。
#
#     Returns:
#         list: 包含所有符合条件的 h5 文件相对路径。
#     """
#     h5_files = []
#     case_range = range(case_start, case_end + 1)
#
#     for case_number in case_range:
#         case_prefix = f"Case{case_number:02d}_"
#         case_files = [f for f in os.listdir(directory) if f.startswith(case_prefix) and f.endswith('.h5')]
#         h5_files.extend([os.path.relpath(os.path.join(directory, f), directory) for f in case_files])
#
#     return h5_files
#
#
# # 获取 Case00 到 Case34 的所有 h5 文件相对路径
# directory = '/home/prejudice/datasets/promise12_levelset/png2h5'
# case_start = 0
# case_end = 34
#
# h5_files = get_relative_h5_files(directory, case_start, case_end)
# case_file = h5_files[1 % len(h5_files)]
# file_path = os.path.join('/home/prejudice/datasets/promise12_levelset/png2h5', case_file)
# with h5py.File(file_path, "r") as h5f_levelset:
#     mask_levelset = h5f_levelset["mask_levelset"][:]
# print((mask_levelset==1).sum())
#
# print(10 % 1012)
# # print(relative_h5_file_paths[0])
# # # # 打印所有相对路径
# # for path in relative_h5_file_paths:
# #     print(path)
#
#
import random

import  numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=1, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)
        img_h, img_w = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)
            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

    return img, mask
np_data_path = '/home/prejudice/datasets/promise12_levelset/npy_image'
X_train = np.load(os.path.join(np_data_path, 'X_train.npy'))
y_train = np.load(os.path.join(np_data_path, 'y_train.npy'))
# print(len(X_train))

img= X_train[10]  # [224,224] [224,224]
mask = y_train[50]
img, mask = cutout_gray(img, mask)
# print(img)
#
plt.imshow(img, cmap='gray')
plt.show()
# img = Image.fromarray(img.astype('uint8'))
#
# img.save(('/home/prejudice/datasets/promise12_levelset/test.png'))

# for i in range(0, 47):
#     img = X_train[i]
#     plt.imshow(img)
#     plt.show()