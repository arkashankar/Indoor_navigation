# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:45:10 2020

@author: Mypc
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from skimage.measure import structural_similarity as ssim
from skimage import measure
from PIL import Image
img = plt.imread('new_img/frameL01P01.jpg-60.jpg')
def img_edge(img):
    #define the vertical filter
    vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
    
    #define the horizontal filter
    horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]
    
    #read in the pinwheel image
    
    
    #get the dimensions of the image
    n,m,d = img.shape
    
    #initialize the edges image
    edges_img = img.copy()
    
    #loop over all pixels in the image
    for row in range(3, n-2):
        for col in range(3, m-2):
            
            #create little local 3x3 box
            local_pixels = img[row-1:row+2, col-1:col+2, 0]
            
            #apply the vertical filter
            vertical_transformed_pixels = vertical_filter*local_pixels
            #remap the vertical score
            vertical_score = vertical_transformed_pixels.sum()/4
            
            #apply the horizontal filter
            horizontal_transformed_pixels = horizontal_filter*local_pixels
            #remap the horizontal scoreplt.imshow(gray, cmap='gray')
            horizontal_score = horizontal_transformed_pixels.sum()/4
            
            #combine the horizontal and vertical scores into a total edge score
            edge_score = (vertical_score**2 + horizontal_score**2)**.5
            
            #insert this edge score into the edges image
            edges_img[row, col] = [edge_score]*3
    
    #remap the values in the 0-1 range in case they went out of boundsQ
    edges_img = edges_img/edges_img.max()
    #median_edges = np.median(edges_img, axis=0)
   
    return edges_img
pic = img_edge(img)
img2 = plt.imread('data/frame0.jpg')
pic2 = img_edge(img2)
plt.imshow(img_edge(img))
plt.imshow(pic2)
median_edges = np.median(pic)
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
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB, multichannel=True)
	return m,s
count=0
file_found = []
for _, dirnames, filenames in os.walk(r'D:\Indoor_loc\new_img'):
    count = count+1
for _,dirnames1,filenames1 in os.walk(r'D:/Indoor_loc/data'):
    count = count+1  
for j in filenames1:
    image = cv2.imread('data/'+j) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# Find Canny edges 
    #edged = cv2.Canny(gray, 30, 200)
    mse_list = []
    ssim_list = []
    for i in filenames:    
        img = cv2.imread('D:/Indoor_loc/new_img/'+i)
        #pic = img_edge(img) 
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      
     
        #edged1 = cv2.Canny(gray1, 50, 200) 
        m,n = compare_images(gray,gray1)
        mse_list.append(m)
        ssim_list.append(n)
        
    maximus = np.max(ssim_list)
    key = ssim_list.index(maximus)
    file_found.append(filenames[key])

    

