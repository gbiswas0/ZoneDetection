import cv2
import numpy as np

img = cv2.imread('New1.jpg')
img = cv2.GaussianBlur(img, (7,7), 0)
cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret3,normal = cv2.threshold(cimg,150,255,cv2.THRESH_BINARY)
invnorm = cv2.bitwise_not(normal)

			
cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)
cv2.imshow('Image', normal)
cv2.waitKey(0)

circles = cv2.HoughCircles(normal,cv2.HOUGH_GRADIENT,2,500,
                            param1=400,param2=70,minRadius=100,maxRadius=500)

print(circles)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)
cv2.imshow('Image', img)
cv2.waitKey(0)

