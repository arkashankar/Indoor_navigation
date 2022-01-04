import cv2 
import numpy as np 
from skimage import measure
import os
# Let's load a simple image with 3 black squares 
image = cv2.imread('data/frame0.jpg') 
cv2.waitKey(0) 
  
# Grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# Find Canny edges 
edged = cv2.Canny(gray, 30, 200) 
cv2.waitKey(0) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
  
cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 
  
print("Number of Contours found = " + str(len(contours))) 





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


file_found=[]
count=0    
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
