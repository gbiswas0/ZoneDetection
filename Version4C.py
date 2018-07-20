#Import Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from franges import drange, frange
import copy
import matplotlib.pyplot as plt
from scipy import stats
#FYI ZOE(s) = Zone of Inhibition(s)

#Take an input image and do pre-processing
wordup = 'image2018-06-20_074300.jpg'
# No current way to set multiple images to each other this string is used multiple points in the code for the program to see what image to read
cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)

img = cv2.imread(wordup)
imgblur = cv2.GaussianBlur(img, (15,15), 0)
imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgbgr = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
#the2nd = cv2.cvtColor(imghsv, cv2.COLOR_HSV2GRAY)
#turn it to B/W and blur it to eliminate noise
hue, sat, val = cv2.split(imghsv)
#The image has to be split into hsv channels to get the value of all the pixels
width, height = img.shape[:2]
dishblank = np.zeros((width,height, 3), np.uint8)
#val = val - 50
#newhsv = cv2.merge((hue, sat, val))
#img.convertTo(darkimg, -1, 1, -50)
#value = -10
#vvalue = imghsv[..., 2]
#imghsv = np.where((255-vvalue)<value,255, vvalue+value)
#imggray[...,2] = imggray[...,2]*0.1
#blankval = np.array(np.zeros((255,), dtype = int))
#valhist = list(blankval)
#imggray = imggray *.6
#for x in range(0, width):
#	for y in range(0, height):
#		imggray[x][y] = int(imggray[x][y] * .5)
#
#print(imggray)
#print(val.flatten())
#print(imggray, imggray.shape)
#for i in val:
#	for h in i:
#		valhist[h] = valhist[h] + 1
#print(val.flatten())
#newval = val.flatten()
#plt.hist(newval, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#lt.show()

discnum = 6
retval, otsuThresh = cv2.threshold(imggray,np.mean(val)+(np.std(val)),255,cv2.THRESH_BINARY)
#The mean value of all the pixels + the standard deviation is used to determine the usual threshold optimal for finding discs on a dish
retrival, dishimg = cv2.threshold(imggray,np.mean(val)-(np.std(val)),255,cv2.THRESH_BINARY)
#Mean value minus standard deviation is ideal for finding the plate, the edges of the plate must be found
#, as they often interfere with the zone of inhibition; more elaboration later on
inv = cv2.bitwise_not(otsuThresh)
# Invert the image

cv2.imshow('Image', imgblur)
cv2.waitKey(0)

cv2.imshow('Image', otsuThresh)
cv2.waitKey(0)
#Make a window and test that the program has the correct image

im, contours, hierarchy = cv2.findContours(inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Finds the contours of the discs 
im2, dishape, hierarch = cv2.findContours(dishimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Finds the contour of the petri dish

petriDishSize = 150
width, height = img.shape[:2]
print('widthheight', width, height)
petriDishConversion = petriDishSize / height
#Records the converson rate from px to inches/mm

pxtomm = 0.0307692

discs = {}
dindex = []
# Arrays made to store information about the discs, as they are the central feature used to find ZOEs
maxarea = dishape[0]
M = cv2.moments(dishape[0])
if(M["m00"] == 0):
	M["m00"] = 1
cdX = int(M["m10"] / M["m00"])
cdY = int(M["m01"] / M["m00"])
dishrad = math.sqrt(cv2.contourArea(dishape[0])/math.pi)

#The array to be used to store the contour that represents the petri dish
for d in dishape:

	M = cv2.moments(d)

	if cv2.contourArea(maxarea) < cv2.contourArea(d):
		maxarea = d
		cdX = int(M["m10"] / M["m00"])
		cdY = int(M["m01"] / M["m00"])
		dishrad = math.sqrt(cv2.contourArea(d)/math.pi)
		print('whyy', cdX, cdY, dishrad)

cv2.circle(dishblank, (int(cdX), int(cdY)), int(dishrad), (255, 255, 255), 1)
maxarea = maxarea*0.1
cv2.imshow('Image' , dishblank)
print(cdX,cdY, dishrad)
print(int(cdX*0.1), int(cdY*0.1))
cv2.waitKey(0)
#Occasionally there are spurious contours found in the thresholded image primarily used to find the dish. 
#This makes sure "maxarea" represents the petri dish



# line means the code is repeated, the character chosen is arbitrary, but same character almost the same code	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
v = 0
# V represents the number of discs

#This double checks that the program has found the discs In the future the program should be able to make sure without inputting discnum
#Change later

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
discs={}
# Because the threshold was wrong, create a new discs dicionary
imgclean = cv2.imread(wordup)
#Get a new clean photo to work on
threshphoto={}
#is a dictionary of the 5d arrays that store each image
edgephoto={}
#is a dictionary of each of the contour arrays that correspond to a certain threshold
index = []
#Index that stores the number of discs found per threshold

for x in range(0, 256):

	retval3,threshphoto[x] = cv2.threshold(imggray,x,255,cv2.THRESH_BINARY)
#Creates a different image for each threshold possible and stores it at threshphoto[x] where x is the threshold of given iamge
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


for x in range(0, 256):
	edgephoto[x], contours, hierarchy = cv2.findContours(threshphoto[x],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		#Finds contours and stores them in edgephoto[x] where x is the given threshold, just like threshphoto
	v = 0
		# v serves the same purpose here, to cound the discs found, but this time it's for every threshold, It must be reset for each threshold
	for c in contours:
	 	#Same loop setup as before, looking for discs
		M = cv2.moments(c)
	
		Area = cv2.contourArea(c) * pxtomm
		circumfrence = cv2.arcLength(c, True)*pxtomm
		AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
		CircR = circumfrence/(math.pi*2)
		diametermm = AreaR*2
		
		if(M["m00"] != 0) and  (2/1) > M['mu20']/M['mu02'] > (1/2) and 500 < cv2.arcLength(c, True) < 1000 and 15000 < cv2.contourArea(c) < 30000 and math.fabs(AreaR - CircR) < 1:
				#Slightly more stringent qualifications for discs, as there will be more false positives and extreme thresholds
				v = v+1
				# counts the number of discs found
	index.append(v)
		#Records the discs found in each threshold, index[x] gives the number of discs found at threshold x

goodthresh = []
	#best thresh = best threshold number ... kinda self explanatory
bestthresh = 0
z = 0
for y in range (0, len(index)):
	if(index[y] > index[bestthresh]):
		bestthresh = y
for z in range (0, len(index)):
	if(index[z] == index[bestthresh]):
		goodthresh.append(z)
print(bestthresh,index[bestthresh])
	# Searches through the index to find the threshold with the right number of discs, i.e. the one equal to the discnum(inputted by the user)
	#Hope to un-hardcode at some point
bestthresh = int(np.mean(goodthresh))
if(bestthresh == 0):
	bestthresh = 127
	#Precautionary measure if the discs aren't found 127 is considered a good "default threshold"

print('bestthresh', bestthresh)
print(index)


cv2.imshow("Image", img)
cv2.waitKey(0)

	## This bit of the code is the same as the loop earlier, the difference is the threshold is the one with all the discs same tilde break is there.

ret3,otsuThresh = cv2.threshold(imggray,bestthresh,255,cv2.THRESH_BINARY)
invOtsuThresh = cv2.bitwise_not(otsuThresh)
cv2.imshow("Image", invOtsuThresh)
cv2.waitKey(0)

im3, contours, hierarchy = cv2.findContours(invOtsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Image", img)
print('test')
cv2.waitKey(0)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
v = 0

for c in contours:

	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	M = cv2.moments(c)
	
	Area = cv2.contourArea(c) * pxtomm
	circumfrence = cv2.arcLength(c, True)*pxtomm
	AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
	CircR = circumfrence/(math.pi*2)
	diametermm = AreaR*2
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	if(M["m00"] != 0) and  (1.2/1) > M['mu20']/M['mu02'] > (1/1.2) and 500 < cv2.arcLength(c, True) < 1000 and 15000 < cv2.contourArea(c) < 30000 and math.fabs(AreaR - CircR) < 1:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		discs[v+1] = cX, cY
		v = v + 1

		cv2.drawContours(imgclean, [c], -1, (0, 255, 0), 2)
		text = "diameter " + str(round(diametermm, 3)) + " mm "
			#cv2.putText(imgclean, text, (cX - 20, cY - 20),
			#	cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Makes sure the discs are actually found, troubleshooting measure

#truedges, contours, truehierch = cv2.findContours(threshphoto[bestthresh],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
retval, otsuThresh = cv2.threshold(imggray,np.mean(val)+(np.std(val)),255,cv2.THRESH_BINARY)
im, contours, hierarchy = cv2.findContours(inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

matchdiscs = 0
sharedzone = {}
# match discs records the number of discs found within a contour
#shared zone is used for large contours with multiple discs found within, typically present when ZOE's overlap
#imgdif = cv2.resize(imgblur, (0,0), fx = 0.1, fy = 0.1)
downfactor = 0.1
Ynew = int(cdY*0.1)
Xnew = int(cdX*0.1)
Cnew = int(dishrad*downfactor)
imgdif = cv2.resize(imgblur, (0,0), fx = downfactor, fy = downfactor)
height0, width0, channels0 = imgdif.shape
hdif = 0
wdif = 0
if (Ynew - Cnew > 0 and Xnew - Cnew > 0):
	cropimg = imgdif[Ynew-Cnew:Ynew+Cnew, Xnew-Cnew: Xnew+Cnew].copy()
	hdif = Ynew-Cnew
	wdif = Xnew-Cnew
elif (Ynew - Cnew > 0 and Xnew - Cnew < 0):
	cropimg = imgdif[Ynew-Cnew:Ynew+Cnew, 0: Xnew+Cnew].copy()
	hdif = Ynew-Cnew
elif (Ynew - Cnew < 0 and Xnew - Cnew > 0):
	cropimg = imgdif[0:Ynew+Cnew, Xnew-Cnew: Xnew+Cnew].copy()
	wdif = Xnew-Cnew
else:
	cropimg = imgdif[0:Ynew+Cnew, 0:Xnew+Cnew].copy()

print('differences', wdif, hdif)
height, width, channels = cropimg.shape
blank_image = np.zeros((height,width,3), np.uint8)
newblur = cv2.GaussianBlur(cropimg, (7,7), 0)
grayimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
smalldish = copy.deepcopy(blank_image)
cv2.circle(smalldish, (int(width/2), int(height/2)), Cnew, (255, 255, 255), 1 )

#cv2.imshow("Image", cropimg)
#cv2.waitKey(0)
ttests = {}
pvals = []
tvals = []
graddists = []
innervalues = []
outervalues = []
r = 0
mindishdist = 1000000000000000000000000000
dishcoord = (0,0)
disharr = np.transpose(np.where(smalldish==255))
disharr = np.delete(disharr, np.s_[2:], 1)
print(Xnew, Ynew, Cnew)
cv2.imshow("Image", grayimg)
cv2.waitKey(0)
overlap = False

newdiscs = {}
print(hdif, wdif)
print(discs)
for z in range(1, len(discs)+1):
	newdiscs[z] =((discs[z][0]*downfactor)-wdif, (discs[z][1]*downfactor)-hdif)
print(newdiscs)
print(discs)
for d in range(1, len(discs)+1):
	
	mindishdist = 1000000000000000000000000000
	for e in range (0, len(disharr)):
		xdist = disharr[e][1] - int(newdiscs[d][0])
		ydist = disharr[e][0] - int(newdiscs[d][1])
		totaldishdist = math.sqrt((math.pow(xdist, 2)) + (math.pow(ydist, 2)))
				#Same as the previous loop, but with the petri dish instead
		if(totaldishdist < mindishdist):
			mindishdist = totaldishdist
			dishcoord = (disharr[e][1],disharr[e][0])

	ttests['disc' + str(d)] = {}
	r = 0
	print('disc', d, mindishdist, dishcoord)
	for h in range(15, 80, 2):
		r = r + 1

		innervalues = []
		outervalues = []

		blank_image = np.zeros((height,width,3), np.uint8)
		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h, (255, 255, 255), 1)
		points = np.transpose(np.where(blank_image==255))
		innercircle = np.delete(points, np.s_[2:], 1)

		blank_image = np.zeros((height,width,3), np.uint8)
		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h+2, (255, 255, 255), 1)
		points = np.transpose(np.where(blank_image==255))
		outercircle = np.delete(points, np.s_[2:], 1)


		for i in range(0, len(innercircle)-1):
			xdist = dishcoord[0] - innercircle[i][1]
			ydist = dishcoord[1] - innercircle[i][0]
			bufferdist = int(math.sqrt((math.pow(xdist, 2)) + (math.pow(ydist, 2))))
			print(bufferdist)

			if (bufferdist > 20): 
				innervalues.append(grayimg[innercircle[i][0]][innercircle[i][1]])
		
		for j in range(0, len(outercircle)-1):
			xdist = dishcoord[0] - outercircle[j][1]
			ydist = dishcoord[1] - outercircle[j][0]
			bufferdist = int(math.sqrt((math.pow(xdist, 2)) + (math.pow(ydist, 2))))

			if (bufferdist > 20): 
				outervalues.append(grayimg[outercircle[j][0]][outercircle[j][1]])
			else:
				print('traitor')

		cv2.circle(blank_image, (int(width/2), int(height/2)), Cnew, (255, 255, 255), 1 )
		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), 5, (255, 255, 255), 1 )
		cv2.circle(blank_image, (dishcoord), 20, (255, 0, 255), 1 )
		cv2.imshow("Image", blank_image)
		cv2.waitKey(0)
				

		ttests['disc'+str(d)][r] = stats.ttest_ind(innervalues,outervalues)

print('BREATHE')
print(discs)

for a in range(1, len(discs)+1):
	tvals = []
	pvals = []

	for b in range(1, len(ttests['disc' + str(a)])):
		tvals.append(ttests['disc' + str(a)][b][0])
		pvals.append(ttests['disc' + str(a)][b][1])
	
	if(np.amax(tvals) > math.fabs(np.amin(tvals))):
		if(pvals[np.argmax(tvals)] < 0.005 and np.amax(tvals) > 3):
			graddists.append(((np.argmax(tvals)*2)+15))
		else:
			graddists.append(10)
		print(a, np.amax(tvals), np.amin(pvals))
	else:
		if(pvals[np.argmin(tvals)] < 0.005 and np.amin(tvals) < -3):
			graddists.append(((np.argmin(tvals)*2)+15))
		else:
			graddists.append(10)
		print(a, np.amin(tvals), np.amin(pvals))

print(graddists)
zonevals = []
hsvmask = cv2.cvtColor(imgdif, cv2.COLOR_BGR2HSV)
for k in range(1, len(discs)+1):
	zonearr = []
	blank_image = np.zeros((height,width,3), np.uint8)
	cv2.circle(blank_image, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(graddists[k-1]), (255, 255 , 255), -1)
	cv2.circle(blank_image, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(10), (0, 0 , 0), -1)
	#cv2.imshow("Image", blank_image)
	#cv2.waitKey(0)
	print('eskettit')
	zonearr = np.transpose(np.where(blank_image==255))
	zonearr = np.delete(zonearr, np.s_[2:], 1)
	for m in range(0, len(zonearr)):
		zonevals.append(hsvmask[zonearr[m][0]][zonearr[m][1]][2])
	print(np.mean(zonevals))
	if(np.mean(zonevals) > 150):
		cv2.circle(cropimg, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(graddists[k-1]), (0, 0 , 255), 1)
	else:
		cv2.circle(cropimg, (int(newdiscs[k][0]), int(newdiscs[k][1])), 10, (0, 0, 255), 1)

	cv2.imshow("Image", cropimg)
	cv2.waitKey(0)
cv2.imwrite("zones" + wordup, cropimg)
print(imgdif.shape)
print(imgdif.shape)
#
