import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage import io
from os import listdir
from os.path import isfile, join

padding = 2 # pixels all around
original_dataset_dir = "original_dataset"
train_output_dataset_dir = "mass_%dx_padding_dataset_train" % padding
val_output_dataset_dir = "mass_%dx_padding_dataset_val" % padding
test_output_dataset_dir = "mass_%dx_padding_dataset_test" % padding
train_fraction = 0.7
val_fraction = 0.1
test_fraction = 1.0 - train_fraction - val_fraction

def get_nonmask_filenames():
    filenames = []
    for f in listdir(original_dataset_dir):
        if isfile(join(original_dataset_dir, f)):
            filename_without_extension = os.path.splitext(f)[0]
            if not filename_without_extension.endswith("mask"):
                filenames.append((f, filename_without_extension))
    return filenames

def get_cropped_mass(filename, filename_without_extension):
    img_full_path = join(original_dataset_dir, filename)
    original_image = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
    mask_full_path = join(original_dataset_dir, filename_without_extension + "_mask.tif")
    mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = np.where(mask == 255)
    startx, starty = np.min(rows), np.min(cols)
    endx, endy = np.max(rows), np.max(cols)
    xpadding = (endx - startx) / 2
    ypadding = (endy - starty) / 2
    startx = max(0, startx-xpadding)
    endx = min(endx+1+xpadding, original_image.shape[0])
    starty = max(0, starty-ypadding)
    endy = min(endy+1+ypadding, original_image.shape[1])
    if starty == 0:
        y_indices = np.where(original_image[startx:startx+1, 0:ypadding] == 65535)[1]
        if y_indices.shape[0] != 0:
            starty = np.max(y_indices) + 1
    if endy == original_image.shape[1]:
        y_indices = np.where(original_image[startx:startx+1, endy-ypadding:endy] == 65535)[1]
        if y_indices.shape[0] != 0:
            endy = np.min(y_indices) + endy - ypadding
    cropped_image = original_image[startx:endx, starty:endy]
    return cropped_image

def create_dir_if_not_exists(output_dataset_dir):
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

def write_mass_dataset(nonmask_filenames, output_dataset_dir):
    for filename, filename_without_extension in nonmask_filenames:
        print filename
        cropped_mass = get_cropped_mass(filename, filename_without_extension)
        output_full_path = join(output_dataset_dir, filename)
        io.imsave(output_full_path, cropped_mass, plugin='tifffile')
        break

def write_mass_datasets(nonmask_filenames):
    create_dir_if_not_exists(train_output_dataset_dir)
    create_dir_if_not_exists(val_output_dataset_dir)
    create_dir_if_not_exists(test_output_dataset_dir)
    num_files = len(nonmask_filenames)
    train_split_idx = int(train_fraction * num_files)
    val_split_idx = int(val_fraction * num_files) + train_split_idx
    train_filenames = nonmask_filenames[:train_split_idx]
    val_filenames = nonmask_filenames[train_split_idx:val_split_idx]
    test_filenames = nonmask_filenames[val_split_idx:]
    write_mass_dataset(train_filenames, train_output_dataset_dir)
    write_mass_dataset(val_filenames, val_output_dataset_dir)
    write_mass_dataset(test_filenames, test_output_dataset_dir)

def main():
    nonmask_filenames = get_nonmask_filenames()
    write_mass_datasets(nonmask_filenames)
  
# filename = 'P_00001_LEFT_CC_1.tif'
# test_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
# mask = cv2.imread('P_00001_LEFT_CC_1_mask.tif', cv2.IMREAD_GRAYSCALE)
# rows, cols = np.where(mask == 255)
# startx, starty = np.min(rows), np.min(cols)
# endx, endy = np.max(rows), np.max(cols)
# print mask.shape
# print test_image.shape
# print startx, starty
# print endx, endy
# cropped_image = test_image[startx:endx+1, starty: endy+1]
# print cropped_image

# plt.imshow(cropped_image, cmap='gray')
# plt.show()

# io.imsave('a.tif', cropped_image, plugin='tifffile')

if __name__ == "__main__":
    main()