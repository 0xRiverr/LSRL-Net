import nibabel as nib
import numpy as np

# 加载分割预测结果和标签图像
prediction_nii = nib.load('/mnt/d/works/pycharmWorks/LabelPropagation/model/ACDC_wsl/Lab_Propagation_norelearning_3/unet_predictions/patient001_frame01_pred.nii.gz')
ground_truth_nii = nib.load('/mnt/d/works/pycharmWorks/LabelPropagation/model/ACDC_wsl/Lab_Propagation_norelearning_3/unet_predictions/patient001_frame01_gt.nii.gz')


# 提取图像数据（3D数组）
prediction = prediction_nii.get_fdata()
ground_truth = ground_truth_nii.get_fdata()

# 确保两个图像的形状一致
assert prediction.shape == ground_truth.shape, "预测图像和标签图像大小不一致"
# 找到不一致的区域
inconsistency_map = np.where(prediction != ground_truth, 1, 0)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 选择一个感兴趣的切片进行可视化，例如第50个切片
slice_idx = 5

# 显示预测、标签和不一致区域
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(prediction[:, :, slice_idx], cmap='gray')
plt.title('Prediction')

plt.subplot(1, 3, 2)
plt.imshow(ground_truth[:, :, slice_idx], cmap='gray')
plt.title('Ground Truth')

plt.subplot(1, 3, 3)
plt.imshow(inconsistency_map[:, :, slice_idx], cmap='hot')
plt.title('Inconsistencies')

plt.savefig('/mnt/d/works/pycharmWorks/LabelPropagation/inconsistent_areas1.png')
