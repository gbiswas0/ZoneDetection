#Import Libraries
import cv2
import numpy as np
import os.path
#from matplotlib import pyplot as plt
import math
#from franges import drange, frange
import copy
from scipy import stats
#FYI ZOE(s) = Zone of Inhibition(s)

#Take an input image and do pre-processing
wordup = 'condensation-image2018-07-04_19-40-09.jpg'
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
print(img.shape)
hue, sat, val = cv2.split(imghsv)
print('maxs', np.amax(hue), np.amax(sat), np.amax(val))
print('mins', np.amin(hue), np.amin(sat), np.amin(val))
#The image has to be split into hsv channels to get the value of all the pixels
width, height = img.shape[:2]
dishblank = np.zeros((width,height, 3), np.uint8)


discnum = 6
retval, otsuThresh = cv2.threshold(imggray,np.mean(val)+(np.std(val)),255,cv2.THRESH_BINARY)
#The mean value of all the pixels + the standard deviation is used to determine the usual threshold optimal for finding discs on a dish
retrival, dishimg = cv2.threshold(imggray,np.mean(val)-(np.std(val)),255,cv2.THRESH_BINARY)
#Mean value minus standard deviation is ideal for finding the plate, the edges of the plate must be found
#, as they often interfere with the zone of inhibition; more elaboration later on
inv = cv2.bitwise_not(otsuThresh)
# Invert the image

#cv2.imshow('Image', imgblur)
#cv2.waitKey(0)
print('test'+'quote'+',' + str(discnum))
#cv2.imshow('Image', otsuThresh)
#cv2.waitKey(0)
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
#cv2.imshow('Image' , dishblank)
print(cdX,cdY, dishrad)
print(int(cdX*0.1), int(cdY*0.1))
#cv2.waitKey(0)
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
#is a dictionary of each of the contour arrays that correspond to a certain threshold
index = []
#Index that stores the number of discs found per threshold


#Creates a different image for each threshold possible and stores it at threshphoto[x] where x is the threshold of given iamge
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


for x in range(0, 256):
	retval3,threshphoto = cv2.threshold(imggray,x,255,cv2.THRESH_BINARY)
	edgephoto, contours, hierarchy = cv2.findContours(threshphoto,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
		
		if(M["m00"] != 0) and  (2/1) > M['mu20']/M['mu02'] > (1/2) and 500 < cv2.arcLength(c, True) < 1000 and 15000 < cv2.contourArea(c) < 30000:
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


#cv2.imshow("Image", img)
#cv2.waitKey(0)

	## This bit of the code is the same as the loop earlier, the difference is the threshold is the one with all the discs same tilde break is there.

ret3,otsuThresh = cv2.threshold(imggray,bestthresh,255,cv2.THRESH_BINARY)
#cv2.imshow("Image", invOtsuThresh)
#cv2.waitKey(0)

im3, contours, hierarchy = cv2.findContours(otsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#cv2.imshow("Image", img)
print('test')
#cv2.waitKey(0)
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

	if(M["m00"] != 0) and  (1.2/1) > M['mu20']/M['mu02'] > (1/1.2) and 500 < cv2.arcLength(c, True) < 1000 and 15000 < cv2.contourArea(c) < 30000:
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
clahe = cv2.createCLAHE(clipLimit=10., tileGridSize=(10,10))
print('differences', wdif, hdif)
height, width, channels = cropimg.shape
blank_image = np.zeros((height,width,3), np.uint8)
blurimg = cv2.GaussianBlur(cropimg, (3,3), 0)
grayimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
smalldish = copy.deepcopy(blank_image)
cv2.circle(smalldish, (int(width/2), int(height/2)), Cnew-5, (255, 255, 255), -1 )

#cv2.imshow("Image", cropimg)
#cv2.waitKey(0)
valttests = {}
satttests = {}
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
overlap = False

newdiscs = {}

print(hdif, wdif)
print(discs)
hsvmask = cv2.cvtColor(cropimg, cv2.COLOR_BGR2HSV)
backtobasics = cv2.cvtColor(hsvmask, cv2.COLOR_HSV2BGR)
#cv2.imshow("Image", hsvmask)
#cv2.waitKey(0)
print(hsvmask)
#cv2.imshow("Image", backtobasics)
#cv2.waitKey(0)
labmask = cv2.cvtColor(cropimg, cv2.COLOR_BGR2LAB)
alldish = np.zeros(shape=(1,2), dtype=int)
background = np.zeros(shape=(1,2), dtype = int)
print(alldish)
for z in range(1, len(discs)+1):
	newdiscs[z] =(int(discs[z][0]*downfactor)-wdif, int(discs[z][1]*downfactor)-hdif)
	cv2.circle(smalldish, (newdiscs[z]), 10, (0, 0, 0), -1)

for x in range(0, height):
	for y in range(0, width):
		if (smalldish[x][y][0] == 255 and smalldish[x][y][1] == 255 and smalldish[x][y][2] == 255):
			alldish = np.append(alldish, [[x,y]], axis = 0)
		else:
			background = np.append(background, [[x,y]], axis = 0)
Ycenter = height0 / 2
Xcenter = width0 / 2
toprite = 0
topleft = 0
botrite = 0
botleft = 0
#for q in (0, background.shape[0]):
#	if(background[q][0] < Xcenter and background[q][1] < Ycenter):
#		topleft.append((background[q][0],background[q][1]))
#	elif(background[q][0] > Xcenter and background[q][1] < Ycenter):
#		toprite.append((background[q][0],background[q][1]))
#	elif(background[q][0] > Xcenter and background[q][1] > Ycenter):
#		botrite.append((background[q][0],background[q][1]))
#	elif(background[q][0] > Xcenter and background[q][1] < Ycenter):
#		botleft.append((background[q][0],background[q][1]))


print('background', background.shape, alldish.shape, cropimg.shape, cropimg.size)
print(background, alldish)

dishvalues = np.array([])
dishsaturations = np.array([])
dishues = np.array([])

print('help', grayimg.shape)

#for f in range(0, len(alldish)):
	#dishinf[f, 0] = hsvmask[alldish[f][0]][alldish[f][1]][0]
	#dishinf[f, 1] = hsvmask[alldish[f][0]][alldish[f][1]][1]
	#dishinf[f, 2] = hsvmask[alldish[f][0]][alldish[f][1]][2]

for d in range(1, len(discs)+1):
	
	valttests['disc' + str(d)] = {}
	satttests['disc' + str(d)] = {}
	r = 0
	for h in range(15, 100, 2):
		r = r + 1

		innervalues = np.array([])
		outervalues = np.array([])
		innerhues = np.array([])
		outerhues = np.array([])
		innersats = np.array([])
		outersats = np.array([])

		blank_image = np.zeros((height,width,3), np.uint8)
		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h, (255, 255, 255), 1)
		s = 0
		for s  in range(0, background.shape[0]):
			blank_image[background[s][0]][background[s][1]][0] = 0
			blank_image[background[s][0]][background[s][1]][1] = 0
			blank_image[background[s][0]][background[s][1]][2] = 0
		for x in range(0, height):
			for y in range(0, width):
				if (int(blank_image[x][y][0]) == 255 and int(blank_image[x][y][1]) == 255 and int(blank_image[x][y][2]) == 255):
					innervalues = np.append(grayimg[x][y], innervalues)
					innersats = np.append(hsvmask[x][y][1], innersats)

		blank_image = np.zeros((height,width,3), np.uint8)
		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h+4, (255, 255, 255), 1)
		s = 0
		for s  in range(0, background.shape[0]):
			blank_image[background[s][0]][background[s][1]][0] = 0
			blank_image[background[s][0]][background[s][1]][1] = 0
			blank_image[background[s][0]][background[s][1]][2] = 0
		for x in range(0, height):
			for y in range(0, width):
				if (int(blank_image[x][y][0]) == 255 and int(blank_image[x][y][1]) == 255 and int(blank_image[x][y][2]) == 255):
					outervalues = np.append(grayimg[x][y], outervalues)
					outersats = np.append(hsvmask[x][y][1], outersats)

		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h, (255, 255, 0), 1)
		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h+4, (255, 0 , 0), 1)
		for s  in range(0, background.shape[0]):
			blank_image[background[s][0]][background[s][1]][0] = 0
			blank_image[background[s][0]][background[s][1]][1] = 0
			blank_image[background[s][0]][background[s][1]][2] = 0
		cv2.circle(blank_image, (int(width/2), int(height/2)), Cnew, (255, 255, 255), 1 )
		for n in range(1,len(newdiscs)+1):
			cv2.circle(blank_image, (int(newdiscs[n][0]), int(newdiscs[n][1])), 8, (0, 0, 255), 1 )
		cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), 8, (255, 0, 255), 1 )
		cv2.imshow("Image", blank_image)
		cv2.waitKey(0)
	
		p = 3
		innervalmed = np.median(innervalues)
		innervalstd = np.std(innervalues)
		outervalmed = np.median(outervalues)
		outervalstd = np.std(outervalues)
		innervalmean = np.mean(innervalues)
		outervalmean = np.mean(outervalues)

		innersatmed = np.median(innersats)
		innersatstd = np.std(innersats)
		outersatmed = np.median(outersats)
		outersatstd = np.std(outersats)
		innersatmean = np.mean(innersats)
		outersatmean = np.mean(outersats)

		#print(innerq1, innerq3, outerq1, outerq3)
		#print(innervalues, outervalues)
		innersats = innersats[abs(innersats - innersatmean) < (p * innersatstd)]
		outersats = outersats[abs(outersats - outersatmean) < (p * outersatstd)]

		innervalues = innervalues[abs(innervalues - innervalmean) < (p * innervalstd)]
		outervalues = outervalues[abs(outervalues - outervalmean) < (p * outervalstd)]

		toppercentile = 90
		botpercentile = 10
		innervalq1 = np.percentile(innervalues, botpercentile)
		innervalq3 = np.percentile(outervalues, toppercentile)
		outervalq1 = np.percentile(outervalues, botpercentile)
		outervalq3 = np.percentile(outervalues, toppercentile)
		innervalues = innervalues[abs(innervalues) < innervalq3]
		outervalues = outervalues[abs(outervalues) < outervalq3]
		innervalues = innervalues[abs(innervalues) > innervalq1]
		outervalues = outervalues[abs(outervalues) > outervalq1]


		valttests['disc'+str(d)][r] = stats.ttest_ind(innervalues,outervalues, equal_var=False)
		satttests['disc'+str(d)][r] = stats.ttest_ind(innersats,outersats, equal_var=False)

print('BREATHE')

for a in range(1, len(discs)+1):
	valuetvals = []
	valuepvals = []
	saturationtvals = []
	saturationpvals = []
	bestsatdistances = []
	bestvaldistances = []
	mostsignificant = []

	for b in range(1, len(valttests['disc' + str(a)])):
		valuetvals.append(valttests['disc' + str(a)][b][0])
		valuepvals.append(valttests['disc' + str(a)][b][1])
		saturationtvals.append(satttests['disc' + str(a)][b][0])
		saturationpvals.append(satttests['disc' + str(a)][b][1])

	#if(np.amax(tvals) > math.fabs(np.amin(tvals))):
	#if(np.amax(tvals)>3 and pvals[np.argmax(tvals)] / 2 < 0.05):

	for e in range(1, len(valuetvals)):
		if(saturationpvals[e]*0.5 < 0.005 and saturationtvals[e] < -2):
			bestsatdistances.append((e*2)+15)
		if(valuepvals[e]*0.5 < 0.005 and valuetvals[e] > 2):
			bestvaldistances.append((e*2)+15)
		if(saturationpvals[e]*0.5 < 0.001 and valuepvals[e]*0.5 < 0.01 and saturationtvals[e] < -2 and valuetvals[e] > 2):
			mostsignificant.append((e*2)+15)

	if(len(mostsignificant) == 0):
		graddists.append(10)
	else:
		graddists.append(np.median(mostsignificant))

	print(a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a)
	print('bestsatdistances', bestsatdistances)
	print('bestvaldistances', bestvaldistances)
	print('mostsignificant', mostsignificant)
	print('minimum saturation')
	print('saturation', (np.argmin(saturationtvals)*2)+15, 't ', np.amin(saturationtvals), 'p ', saturationpvals[np.argmin(saturationtvals)])
	print('value', (np.argmin(saturationtvals)*2)+15, 't', valuetvals[np.argmin(saturationtvals)], 'p', valuepvals[np.argmin(saturationtvals)])


	print('maximum value')
	print('value',((np.argmax(valuetvals)*2)+15), 't', np.amax(valuetvals), 'p', valuepvals[np.argmax(valuetvals)])
	print('saturation', ((np.argmax(valuetvals)*2)+15), 't', saturationtvals[np.argmax(valuetvals)], 'p', saturationpvals[np.argmax(valuetvals)])
	print(a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a)
	#else:
	#	graddists.append(10)
	#	print(a, np.amax(tvals), np.amin(pvals))
	#else:
	#	graddists.append(((np.argmin(tvals)*2)+15))
	#	print(a, np.amin(tvals), np.amin(pvals))

print(graddists)

zonesats = np.array([])
zonevals = np.array([])
zonehues = np.array([])
  # convert from BGR to LAB color space
file = ""
for k in range(1, len(discs)+1):
	zonearr = []
	blank_image = np.zeros((height,width,3), np.uint8)
	cv2.circle(blank_image, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(graddists[k-1]), (255, 255 , 255), -1)
	cv2.circle(blank_image, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(10), (0, 0 , 0), -1)

	for x in range(0, height):
			for y in range(0, width):
				if (int(blank_image[x][y][0]) == 255 and int(blank_image[x][y][1]) == 255 and int(blank_image[x][y][2]) == 255):
					zonesats = np.append(hsvmask[x][y][1], zonesats)
					zonevals = np.append(hsvmask[x][y][2], zonevals)
					zonehues = np.append(hsvmask[x][y][0], zonehues)
	if(k == 5):
		print(zonesats, zonehues, zonevals)
	
	m = 3
	toppercentile = 80
	botpercentile = 20

	if(graddists[k-1] != 10):
		valstd = np.std(zonevals)
		satstd = np.std(zonesats)
		satavg = np.mean(zonesats)
		valavg = np.mean(zonevals)
		satmed = np.median(zonesats)
		valmed = np.median(zonevals)
		zonevals = zonevals[abs(zonevals - valmed) < (p * valstd)]
		zonesats = zonesats[abs(zonesats - satmed) < (p * satstd)]

		satq1 = np.percentile(zonesats, botpercentile)
		satq3 = np.percentile(zonesats, toppercentile)
		valq1 = np.percentile(zonevals, botpercentile)
		valq3 = np.percentile(zonevals, toppercentile)
		#print(satq1, satq3, valq1, valq3)
		zonesats = zonesats[abs(zonesats) < satq3]
		zonevals = zonevals[abs(zonevals) < valq3]
		zonesats = zonesats[abs(zonesats) > satq1]
		zonevals = zonevals[abs(zonevals) > valq1]


		print(k, 'sat median: ', np.median(zonesats), np.mean(zonesats), satstd) 
		print(k, 'val median: ', np.mean(zonevals), valstd)
		#cv2.imshow("Image", blank_image)
		#cv2.waitKey(0)
		if(np.mean(zonesats) < 35 and np.median(zonesats) < 35):
			cv2.circle(cropimg, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(graddists[k-1]), (0, 0 , 255), 1)
			file = file + "Disc " + str(k) + ' : ' + str(graddists[k-1]*pxtomm*(1/downfactor)*2) + " mm, " 
			#cv2.imshow("Image", cropimg)
			#cv2.waitKey(0)
		else:
			cv2.circle(cropimg, (int(newdiscs[k][0]), int(newdiscs[k][1])), 10, (0, 0, 255), 1)
			file = file + "Disc " + str(k) + ' : ' + str(10*pxtomm*(1/downfactor)*2) + " mm, "
			#cv2.imshow("Image", cropimg)
			#cv2.waitKey(0)
	else:
		cv2.circle(cropimg, (int(newdiscs[k][0]), int(newdiscs[k][1])), 10, (0, 0, 255), 1)
		file = file + "Disc " + str(k) + ' : ' + str(10*pxtomm*(1/downfactor)*2) + " mm, "

cv2.imshow("Image", cropimg)
cv2.waitKey(0)
print(file)
text_file = open("Output.txt", "w")
text_file.write(file)
text_file.close()
cv2.imwrite('Zones'+wordup, cropimg)
print(imgdif.shape)
#
