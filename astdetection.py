import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
print (sys.argv)
##Step 1 - Bring in an image and turn it grayscale
userimage = input("Enter filepath ")
print(userimage)
img = cv2.imread(userimage)
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray, (5,5), 0)

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

#Commented out for now - this is to split apart the shapes
#corners = cv2.cornerHarris(img,2,3,0.04)

# Petri Dish pixel to mm conversion
petriDishSize = 100
width, height = img.shape[:2]
petriDishConversion = petriDishSize / height

#circles = cv2.HoughCircles(imgblur,cv2.HOUGH_GRADIENT,1,20,
                           # param1=50,param2=30,minRadius=0,maxRadius=0)

#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
    # draw the outer circle
    #cv2.circle(imgblur,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    #cv2.circle(imgblur,(i[0],i[1]),2,(0,0,255),3)
#cv2.imshow('detected circles',imgblur)

#Step 4 - loop over the contours, finding the moments
for c in contours:

	# compute the center of the contour
	M = cv2.moments(c)
	
	# compute the area of each contour
	Area = cv2.contourArea(c)
	Areamm = int(Area * petriDishConversion * petriDishConversion) 
	diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)
	
	# if the moment not equal 0 and the area is large enough, draw the shape and label
	if(M["m00"] != 0) and (20 > diametermm > 5):
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		

		if( (int(cv2.arcLength(c, True) / 6.24) - math.sqrt(int(Area / 3.14))) > 30 ):
			print(int(cv2.arcLength(c, True) / 6.24))
			print(math.sqrt(int(Area / 3.14)))
			leftmost = tuple(c[c[:,:,0].argmin()][0])
			rightmost = tuple(c[c[:,:,0].argmax()][0])
			topmost = tuple(c[c[:,:,1].argmin()][0])
			bottommost = tuple(c[c[:,:,1].argmax()][0])
			#cv2.rectangle(img,(leftmost[0],topmost[1]),(rightmost[0],bottommost[1]),(0,255,0),3)
			cv2.rectangle(img,(178,171),(331,366),(0,255,0),3)
			print(leftmost[0])
			print(topmost[1])
			print(rightmost[0])
			print(bottommost[1])
			wideness = int(rightmost[0])-int(leftmost[0])
			tallness = int(bottommost[1])-int(topmost[1])
			print(wideness)
			print(tallness)
			cropped = img[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
			#cropped = imgblur[leftmost[0]:topmost[1], rightmost[0]:bottommost[1]]
			#cropped = cv2.imread(userimage)
			#cv2.namedWindow('edit' ,cv2.WINDOW_NORMAL)
			#cv2.resizeWindow('edit', 600,600)
			cv2.imshow('cropped', cropped)
			cv2.waitKey(0)





			#contours,hierarchy = cv2.findContours(imgblur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			#cnt = contours[c]
			#hull = cv2.convexHull(cnt,returnPoints = False)
			#defects = cv2.convexityDefects(cnt,hull)

			#for i in range(defects.shape[0]):
   			#	s,e,f,d = defects[i,0]
   			#	start = tuple(cnt[s][0])
   			#	end = tuple(cnt[e][0])
   			#	far = tuple(cnt[f][0])
   			#	cv2.line(img,start,end,[0,255,0],2)
   			#	cv2.circle(img,far,5,[0,0,255],-1)

			#cv2.imshow('Image',img)
			#cv2.waitKey(0)
		# draw the contour and center of the shape on the image
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
























