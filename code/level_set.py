import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter


# 读取H5文件中的医学图像
def read_h5_image(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        image = np.array(f[dataset_name])
    return image


# 初始化水平集函数的方法
def initialize_level_set(image, method='random_points', num_points=50, noise_scale=0.1, sigma=3):
    if method == 'random_points':
        # 随机生成点
        points_x = np.random.randint(0, image.shape[1], num_points)
        points_y = np.random.randint(0, image.shape[0], num_points)

        # 创建二值图像，随机点为1，其余为0
        binary_image = np.zeros_like(image)
        binary_image[points_y, points_x] = 1

        # 计算距离变换
        phi = distance_transform_edt(1 - binary_image) - distance_transform_edt(binary_image)

    elif method == 'noise':
        # 添加噪声
        noise = np.random.normal(scale=noise_scale, size=image.shape)
        phi = image + noise

        # 使用阈值生成初始曲面
        phi = np.sign(phi - np.mean(phi))

    elif method == 'random_field':
        # 生成随机场
        random_field = np.random.normal(size=image.shape)

        # 平滑随机场
        phi = gaussian_filter(random_field, sigma=sigma)

        # 阈值处理生成初始曲面
        phi = np.sign(phi)

    else:
        raise ValueError("Invalid initialization method")

    return phi


# 主程序
def main():
    file_path = '/home/prejudice/datasets/ACDC/ACDC/data/slices/patient001_frame01_slice_1.h5'  # 修改为你的H5文件路径
    dataset_name = 'image'  # 修改为你的数据集名称
    image = read_h5_image(file_path, dataset_name)

    # 选择初始化方法：'random_points', 'noise', 'random_field'
    method = 'random_field'
    phi = initialize_level_set(image, method=method)

    # 可视化图像和水平集函数
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Medical Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.contour(phi, levels=[0], colors='r')
    plt.title('Initialized Level Set Function')
    plt.show()


if __name__ == "__main__":
    main()
