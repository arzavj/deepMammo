import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage import io
from os import listdir
from os.path import isfile, join

padding = 50 # pixels all around
original_dataset_dir = "original_dataset"
output_dataset_dir = "mass_%d_padding_dataset" % padding


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
    startx = max(0, startx-padding)
    endx = min(endx+1+padding, original_image.shape[0])
    starty = max(0, starty-padding)
    endy = min(endy+1+padding, original_image.shape[1])
    if starty == 0:
        y_indices = np.where(original_image[startx:startx+1, 0:padding] == 65535)[1]
        if y_indices.shape[0] != 0:
            starty = np.max(y_indices) + 1
    if endy == original_image.shape[1]:
        y_indices = np.where(original_image[startx:startx+1, endy-padding:endy] == 65535)[1]
        if y_indices.shape[0] != 0:
            endy = np.min(y_indices) + endy - padding
    cropped_image = original_image[startx:endx, starty:endy]
    return cropped_image

def write_mass_dataset(nonmask_filenames):
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)
    for filename, filename_without_extension in nonmask_filenames:
        print filename
        cropped_mass = get_cropped_mass(filename, filename_without_extension)
        output_full_path = join(output_dataset_dir, filename)
        io.imsave(output_full_path, cropped_mass, plugin='tifffile')

def main():
    nonmask_filenames = get_nonmask_filenames()
    write_mass_dataset(nonmask_filenames)
  
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