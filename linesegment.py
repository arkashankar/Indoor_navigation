import cv2
from pylsd.lsd import lsd
import numpy as np
from matplotlib import pyplot as plt 
img = cv2.imread('data/frame1740.jpg')
#Read gray image


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lines = lsd(gray)
for i in range(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    cv2.line(img, pt1, pt2, (0, 0, 255), int(np.ceil(width / 4)))
cv2.imshow('lines',img)
cv2.waitKey(0)
count = 0
angles = []
for i in range (lines.shape[0]):
    if(int(lines[i,0]-lines[i,2])!=0):
        angles.append(int(lines[i,1]-lines[i,3])/int(lines[i,0]-lines[i,2]))
    else:
        angles.append(99999999)

corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10) 
corners = np.int0(corners) 
  
# we iterate through each corner,  
# making a circle at each point that we think is a corner. 
for i in corners: 
    x, y = i.ravel() 
    cv2.circle(img, (x, y), 3, 255, -1) 
  
plt.imshow(img), plt.show() 