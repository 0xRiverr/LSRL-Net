import os
import SimpleITK as sitk
import h5py
import matplotlib.pyplot as plt
import numpy as np

path = "/home/prejudice/datasets/promise12/Prostate"
with open(path + '/all.list', 'r') as f1:
    train_list = f1.readlines()
train_list = [item.replace('\n', '') for item in train_list]

for image_name in train_list:
    image = sitk.ReadImage(path + '/training_data/' + image_name + '.mhd')
    label = sitk.ReadImage(path + '/training_data/' + image_name + '_segmentation.mhd')

    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label)

    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    # image = image.astype(np.float32)
    print(image_array.shape)
    selected_h5 = h5py.File(path+"/data/"+image_name+".h5", "w")
    selected_h5.create_dataset("image", data=image_array)
    selected_h5.create_dataset("label", data=label_array)
    selected_h5.close()

    for i in range(image_array.shape[0]):
        selected_image_slice = image_array[i]
        selected_label_slice = label_array[i]

        selected_slice_h5 = h5py.File(path+"/data/slices/" + image_name + "_slice_" + str(i) + ".h5", "w")
        selected_slice_h5.create_dataset("image", data=selected_image_slice)
        selected_slice_h5.create_dataset("label", data=selected_label_slice)

        selected_slice_h5.close()
