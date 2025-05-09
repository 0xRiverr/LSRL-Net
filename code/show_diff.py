import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# 读取 .nii.gz 文件
def load_nifti(file_path):
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    return data

# /home/nobodyjiang/datasets/show_diff/LSRL
# 加载图像数据
img = load_nifti("/home/nobodyjiang/datasets/show_diff/LSRL/patient022_frame02_img.nii.gz")   # 背景图像
gt = load_nifti("/home/nobodyjiang/datasets/show_diff/LSRL/patient022_frame02_gt.nii.gz")     # Ground Truth
pred = load_nifti("/home/nobodyjiang/datasets/show_diff/LSRL/patient022_frame02_pred.nii.gz") # 预测结果

# 检查维度是否一致
assert img.shape == gt.shape == pred.shape, "图像维度必须一致！"

# 选择需要绘制的切片 (例如中间切片)
slice_idx = img.shape[2] // 2  # 假设选择 z 轴中间切片
img_slice = img[:, :, slice_idx]
gt_slice = gt[:, :, slice_idx]
pred_slice = pred[:, :, slice_idx]

# 计算不一致区域
difference = (gt_slice != pred_slice).astype(np.uint8)
num_inconsistent_pixels = np.sum(difference)  # 不一致像素点的个数

# 创建标注图像
colored_overlay = np.zeros((*img_slice.shape, 3), dtype=np.uint8)
colored_overlay[:, :, 0] = difference * 255  # 红色通道标记不一致区域

# 将背景图像标准化为 0-255 并转换为 RGB 格式
img_norm = ((img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice)) * 255).astype(np.uint8)
img_rgb = np.stack([img_norm] * 3, axis=-1)

# 叠加不一致区域
alpha = 0.5  # 透明度
overlay_result = (img_rgb * (1 - alpha) + colored_overlay * alpha).astype(np.uint8)

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(overlay_result)
plt.title(f"Inconsistent Pixels: {num_inconsistent_pixels}")
plt.axis("off")
plt.show()
