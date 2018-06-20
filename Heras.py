#Import Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from franges import drange, frange


#Take an input image and do pre-processing
img = cv2.imread('Simple.JPG')
imgblur = cv2.GaussianBlur(img, (11,11), 0)
imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
imghsv = cv2.cvtColor(imgblur, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(imghsv)
#turn it to B/W and blur it to eliminate noise

#width = imgblur.shape[0]
#height = imgblur.shape[1]
#val = []
#for x in range(0, width):
#	for y in range(0, height):
#		R = imgblur[x,y,2]
#		G = imgblur[x,y,1]
#		B = imgblur[x,y,0]
#		cmax = max(R,G,B)

		#if(D == 0):
			#H = 0
		#elif(cmax == R):
			#H = 60*(((G-B)/D)%6)
		#elif(cmax == G):
			#H = 60*(((B-R)/D)+2)
		#else:
			#H = 60*(((R-G)/D)+4)

		#if(cmax == 0):
			#S = 0
		#else:
			#S = D/cmax

#		V = cmax

#		imgblur[x,y,0] = V
#		imgblur[x,y,1] = V
#		imgblur[x,y,2] = V
#		val.append(V)

print(np.mean(val))
#Test Statement
retval, otsuThresh = cv2.threshold(imggray,np.mean(val)+math.sqrt(np.std(val)),255,cv2.THRESH_BINARY)
inv = cv2.bitwise_not(otsuThresh)
#Threshold with mean of value+variance because standard deviation always made the threshold too high

cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)
cv2.imshow('Image', inv)
cv2.waitKey(0)
#Test case

#Step 3 - Contour detection and display
im2, contours, hierarchy = cv2.findContours(inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Find the contours


# Draw the corners
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)

#Commented out for now - this is to split apart the shapes
#corners = cv2.cornerHarris(img,2,3,0.04)

# Petri Dish pixel to mm conversion
petriDishSize = 100
width, height = img.shape[:2]
petriDishConversion = petriDishSize / height

#Step 4 - loop over the contours, finding the moments
for c in contours:

	# compute the center of the contour
	M = cv2.moments(c)
	
	# compute the area of each contour
	Area = cv2.contourArea(c)
	Areamm = int(Area * petriDishConversion * petriDishConversion) 
	diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)
	
	# Checks the area, makes sure it's "disk sized", then compares it's circularity, by making sure it has even variance
	if(M["m00"] != 0) and (5 > diametermm > 3) and 1.2 > M["mu20"]/M["mu02"] > 0.8:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
 
		# draw the contour and center of the shape on the image
		cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
		cv2.circle(img, (cX, cY), 7, (0, 0, 0), -1)
		
		# label the circle with area and diameter in mm
		text = "diameter " + str(diametermm) + " mm "
		cv2.putText(img, text, (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 5)
	 
		# show the image
		cv2.imshow("Image", img)
		cv2.waitKey(0)