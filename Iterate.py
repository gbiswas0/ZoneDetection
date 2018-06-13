import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from franges import drange, frange

#This program finds the best threshold for adaptive thresholds

##Step 1 - Bring in an image and turn it grayscale
img = cv2.imread('2018-05-17-1of4-GramNegative.png')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray, (15,15), 0) 

#Check to see that you have the right image
cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)
cv2.imshow('Image', imgblur)
cv2.waitKey(0)

#points defines the number of coordinates found by the contour. The array defines how many were found per threshold
points = []
#Shapes defines the shapes found by each iteration of the thresholding
shapes = []
#v is used to cound the shapes that pass the sanity check that are passed into the "shapes" array.
v=0


#check threshold value with a loop
for x in frange(1, 7, 0.1):
	v=0

	#Thresholds using the value given by the loop
	simple = cv2.adaptiveThreshold(imgblur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,x)
	#Inverts image
	inv = cv2.bitwise_not(simple)
			
	#Detects contours using thresholded image
	drawn, contours, hierarchy = cv2.findContours(inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	points.append(len(contours))

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
	
	# if the moment not equal 0 and the area is large enough, draw the shape and label
		if(M["m00"] != 0) and (13 > diametermm > 2):
			#test loop and add 1 to v to count the number of shapes given this threshold
			v=v+1
	shapes.append(v)
	print(v)
	print(x)

print(shapes)
print(points)
w = 0.0

for y in range(0,len(shapes)):

	if(9 >= shapes[y] >= 5):
		w = (y*0.1) + 1
print(w)

if(w == 0.0):
	w = 7

normal = cv2.adaptiveThreshold(imgblur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
invnorm = cv2.bitwise_not(normal)
			
cv2.imshow('Image', invnorm)
cv2.waitKey(0)

cret3,otsuThresh = cv2.threshold(invnorm,127,255,cv2.THRESH_OTSU)
invOtsuThresh = cv2.bitwise_not(otsuThresh)
			
#Step 3 - Contour detection and display
im2, contours, hierarchy = cv2.findContours(invOtsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Image', img)
cv2.waitKey(0)

# Find the corners
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
	
	# if the moment not equal 0 and the area is large enough, draw the shape and label
	if(M["m00"] != 0) and (13 > diametermm > 2):
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		print(diametermm)
 
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
