# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:35:53 2020

@author: Mypc
"""
import os
import cv2
import numpy as np
from skimage import measure
file_found=[]
count=0    
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB):
	m = mse(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB, multichannel=True)
	return m,s

for _, dirnames, filenames in os.walk(r'D:\Indoor_loc\edged_video_frame'):
    count = count+1
for _,dirnames1,filenames1 in os.walk(r'D:/Indoor_loc/edged_data'):
    count = count+1   
    

image = cv2.imread('edged_video_frame/frame1914.jpg.jpg') 

# Find Canny edges 
    #edged = cv2.Canny(gray, 30, 200)
mse_list = []
ssim_list = []
for j in filenames1:
        img = cv2.imread('D:/Indoor_loc/edged_data/'+j)
        #pic = img_edge(img) 
        #gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      
     
        #edged1 = cv2.Canny(gray1, 50, 200) 
        m,n = compare_images(image,img)
        mse_list.append(m)
        ssim_list.append(n)
        
maximus = np.max(ssim_list)
key = ssim_list.index(maximus)
file_found.append(filenames1[key])