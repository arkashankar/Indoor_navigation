# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 05:27:44 2020

@author: Mypc
"""

from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

image = plt.imread('new_img/frameL01P03.jpg180.jpg')
image.shape
plt.imshow(image)

gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')

gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')