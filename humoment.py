import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
print (sys.argv)
#Step 1 - Bring in an image and turn it grayscale
img = cv2.imread('2018-05-17-3of4-GramNegative.png')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray, (7,7), 0)

#Check to see that you have the right image
cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)
cv2.imshow('Image', imgblur)
cv2.waitKey(0)

#Step 2 - Set the threshold
#threshValue = float(input("Enter a value from 0 to 255 with 127 recommended: "))
#### This section could be a good place to use mixture model method or Otsu Segmentation
#ret, thresh = cv2.threshold(imggray, 25, 255,0)
ret3, otsuThresh = cv2.threshold(imgblur,127, 255,cv2.THRESH_BINARY)
invOtsuThresh = cv2.bitwise_not(otsuThresh)
			
cv2.imshow('Image', invOtsuThresh)
cv2.waitKey(0)		

#Step 3 - Contour detection and display
im2, contours, hierarchy = cv2.findContours(invOtsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Image', img)		
cv2.waitKey(0)

# Find the corners
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)

# Petri Dish pixel to mm conversion
petriDishSize = 100
width, height = img.shape[:2]
petriDishConversion = petriDishSize / height

#Step 4 - loop over the contours, finding the moments
for c in contours:

	# compute the center of the contour
	M = cv2.HuMoments(c)
	
	# compute the area of each contour
	Area = cv2.contourArea(c)
	Areamm = int(Area * petriDishConversion * petriDishConversion) 
	diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)
	print(M)
	# if the moment not equal 0 and the area is large enough, draw the shape and label
	if(M["m00"] != 0) and (20 > diametermm > 5):
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		# draw the contour and center of the shape on the image
		cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
		cv2.circle(img, (cX, cY), 7, (0, 0, 0), -1)
		
		# label the circle with area and diameter in mm
		text = "diameter " + str(diametermm) + " mm "
		cv2.putText(img, text, (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	 
		# show the image
		cv2.imshow("Image", img)
		cv2.waitKey(0)
