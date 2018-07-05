import cv2
import numpy as np
from matplotlib import pyplot as plt

#This program goes through each threshold value and finds the best one to use on the image.


##Step 1 - Bring in an image and turn it grayscale
img = cv2.imread('image2018-06-20_074607.jpg')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray, (7,7), 0)

#Check to see that you have the right image
cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)
cv2.imshow('Image', imgblur)
cv2.waitKey(0)

v = 0
# keeps track of the number of contours
index = []
#indexes the number of contours for each threshold value

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (700,700)
fontScale              = 10
fontColor              = (255,255,255)
lineType               = 3

for x in range(0,255):

	ret3,otsuThresh = cv2.threshold(imgblur,x,255,cv2.THRESH_BINARY)
	# going through each threshold value
	invOtsuThresh = cv2.bitwise_not(otsuThresh)
	print(otsuThresh)
	im2, contours, hierarchy = cv2.findContours(invOtsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	#fileindex = str(x)
	#filename = "zz"+fileindex
	cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Image', 600,600)
	#cv2.putText(otsuThresh,fileindex, 
    #(700,1000), 
    #font, 
    #fontScale,
    #(0,0,0),
    #lineType)
	#cv2.putText(otsuThresh,fileindex, 
    #bottomLeftCornerOfText, 
    #font, 
    #fontScale,
    #fontColor,
    #lineType)
	cv2.imshow('Image', otsuThresh)
	#cv2.imwrite(filename+".png", otsuThresh)
	cv2.waitKey(0)
	print(x)

	#testing

	petriDishSize = 100
	width, height = img.shape[:2]
	petriDishConversion = petriDishSize / height


	for c in contours:

		M = cv2.moments(c)
	
		Area = cv2.contourArea(c)
		Areamm = int(Area * petriDishConversion * petriDishConversion) 
		diametermm = int((((3.14 * Area)/4) ** 0.5) *  petriDishConversion)

		if(M["m00"] != 0) and (13 > diametermm > 3):
			v = v+1
			
			#Records every time a contour was made given the current threshold
	print(v)
	#index.append(v)
	#Records the number of contours per threshold.
	v = 0

bestthresh = []
print(index)
for y in range (0, len(index)):
	if(index[y] == 4):
		bestthresh.append(y)
		#Looks at what threshold has the most contours
print(bestthresh)

if(len(bestthresh) == 0):
	bestthresh.append(127)

#Uses the threshold found
ret3,otsuThresh = cv2.threshold(imgblur,bestthresh[0],255,cv2.THRESH_BINARY)
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
	if(M["m00"] != 0) and (13 > diametermm > 3):
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		print(M)
		print(M["m11"])
		print(cX)
		print(cY)
 
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
