import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import skimage
from skimage import io

filename = 'P_00001_LEFT_CC_1.tif'
test_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
mask = cv2.imread('P_00001_LEFT_CC_1_mask.tif', cv2.IMREAD_GRAYSCALE)
rows, cols = np.where(mask == 255)
startx, starty = np.min(rows), np.min(cols)
endx, endy = np.max(rows), np.max(cols)
print mask.shape
print test_image.shape
print startx, starty
print endx, endy
cropped_image = test_image[startx:endx+1, starty: endy+1]
print cropped_image

plt.imshow(cropped_image, cmap='gray')
plt.show()

io.imsave('a.tif', cropped_image, plugin='tifffile')