#Import Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from franges import drange, frange

#FYI ZOE(s) = Zone of Inhibition(s)

#Take an input image and do pre-processing
wordup = 'image2018-06-20_074607.jpg'
img = cv2.imread(wordup)
imgblur = cv2.GaussianBlur(img, (15,15), 0)
imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
imghsv = cv2.cvtColor(imgblur, cv2.COLOR_BGR2HSV)
#turn it to B/W and blur it to eliminate noise
hue, sat, val = cv2.split(imghsv)
#The image has to be split into hsv channels to get the value of all the pixels

discnum = 5
retval, otsuThresh = cv2.threshold(imggray,np.mean(val)+(np.std(val)),255,cv2.THRESH_BINARY)
#The mean value of all the pixels + the standard deviation is used to determine the usual threshold optimal for finding discs on a dish
retrival, dishimg = cv2.threshold(imggray,np.mean(val)-(np.std(val)),255,cv2.THRESH_BINARY)
#Mean value minus standard deviation is ideal for finding the plate, the edges of the plate must be found
#, as they often interfere with the zone of inhibition; more elaboration later on
inv = cv2.bitwise_not(otsuThresh)


cv2.namedWindow('Image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 600,600)
cv2.imshow('Image', inv)
cv2.waitKey(0)
#Make a window and test that the program has the correct image

im, contours, hierarchy = cv2.findContours(inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Finds the contours of the discs 
im2, dishape, hierarch = cv2.findContours(dishimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Finds the contour of the petri dish

petriDishSize = 100
width, height = img.shape[:2]
petriDishConversion = petriDishSize / height
#Records the converson rate from px to inches/mm

discs = {}
dindex = []
# Arrays made to store information about the discs, as they are the central feature used to find ZOEs

maxarea = dishape[0]
#The array to be used to store the contour that represents the petri dish
for d in dishape:

	if cv2.contourArea(maxarea) < cv2.contourArea(d):
		maxarea = d
	

v = 0

for c in contours:

	M = cv2.moments(c)
	
	Area = cv2.contourArea(c)
	Areamm = int(Area * petriDishConversion * petriDishConversion) 
	diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)
	
	if(M["m00"] != 0) and (5 > diametermm > 2) and 1.15 > M["mu20"]/M["mu02"] > 0.85:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		v = v+1
		discs[v] = cX, cY
		dindex.append(c)

		cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

print(len(discs))
if (len(discs) != discnum):

	discs={}

	imgclean = cv2.imread(wordup)

	shapes={}

	points={}

	threshphoto={}

	edgephoto={}

	index = []

	petriDishSize = 100
	width, height = img.shape[:2]
	petriDishConversion = petriDishSize / height

	for x in range(0, 256):

		retval3,threshphoto[x] = cv2.threshold(imggray,x,255,cv2.THRESH_BINARY)

	for x in range(0, 256):
		edgephoto[x], contours, hierarchy = cv2.findContours(threshphoto[x],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		v = 0
	
		for c in contours:

			M = cv2.moments(c)
	
			Area = cv2.contourArea(c)
			Areamm = int(Area * petriDishConversion * petriDishConversion) 
			diametermm = int((((math.pi * Area)/4) ** 0.5) *  petriDishConversion)
		
		
			if(M["m00"] != 0) and diametermm > 3 and 5 > diametermm and (2/1) > M['mu20']/M['mu02'] > (1/2) and cv2.arcLength(c, True) < 1000:
				if(x == 159):
					print('value', M['mu20']/M['mu02'])
					print(cv2.arcLength(c, True))
				v = v+1

		index.append(v)

	bestthresh = 0

	for y in range (0, len(index)):
		if(index[y] == discnum):
			bestthresh = y
	if(bestthresh == 0):
		bestthresh = 127
	
	print('bestthresh', bestthresh)
	print(index)

	cv2.imshow("Image", img)
	cv2.waitKey(0)

	ret3,otsuThresh = cv2.threshold(imggray,bestthresh,255,cv2.THRESH_BINARY)
	invOtsuThresh = cv2.bitwise_not(otsuThresh)
	cv2.imshow("Image", invOtsuThresh)
	
	cv2.waitKey(0)

	im3, contours, hierarchy = cv2.findContours(invOtsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

	petriDishSize = 100
	width, height = img.shape[:2]
	petriDishConversion = petriDishSize / height

	v = 0

	for c in contours:

		M = cv2.moments(c)
	
		Area = cv2.contourArea(c)
		Areamm = int(Area * petriDishConversion * petriDishConversion) 
		diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)

		if(M["m00"] != 0) and (5 > diametermm > 3) and 6 > M['mu20']/M['mu02'] > (1/6):
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			discs[v+1] = cX, cY
			v = v + 1

			cv2.drawContours(imgclean, [c], -1, (0, 255, 0), 2)
			text = "diameter " + str(diametermm) + " mm "
			cv2.putText(imgclean, text, (cX - 20, cY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	 
	cv2.imshow("Image", imgclean)
	cv2.waitKey(0)


matchdiscs = 0
sharedzone = {}
for c in contours:

	inzone = []
	M = cv2.moments(c)

	Area = cv2.contourArea(c)
	Areamm = int(Area * petriDishConversion * petriDishConversion) 
	diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)

	if(M["m00"] != 0 and (40 > diametermm > 5)):
		cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
		topx = 0
		topy = 0
		botx = 100000000000000
		boty = 100000000000000
		for a in range(0, len(c)):
				if(c[a][0][0] > topx):
					topx = c[a][0][0]
				if(c[a][0][1] > topy):
					topy = c[a][0][1]
				if(c[a][0][0] < botx):
					botx = c[a][0][0]
				if(c[a][0][1] < boty):
					boty = c[a][0][1]
		print(topx, topy, botx, boty)
		for b in range(1, len(discs)+1):
			dist = cv2.pointPolygonTest(c,(discs[b]),False)
			if(dist > 0):
				inzone.append(b)
		print(discs)

	matchdiscs = matchdiscs + len(inzone)
		
	if(len(inzone) == 1):

		minzonedist = 10000000000000000000000000000000000
		mindishdist = minzonedist
		radii = []
		avgradius = 0
		for f in range(0, len(c)):
			xdist = c[f][0][0] - discs[inzone[0]][0]
			ydist = c[f][0][1] - discs[inzone[0]][1]
			totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
			radii.append(totaldist)
			if(totaldist < minzonedist):
				minzonedist = totaldist
				zonecoord = (c[f][0][0], c[f][0][1])

		for g in range(0, len(maxarea)):
			xddist = maxarea[g][0][0] - discs[inzone[0]][0]
			yddist = maxarea[g][0][1] - discs[inzone[0]][1]
			totaldishdist = math.sqrt((xddist*xddist) + (yddist*yddist))
			if(totaldishdist < mindishdist):
				mindishdist = totaldishdist
				dishcoord = (maxarea[g][0][0], maxarea[g][0][1])
			
		coord = zonecoord

		mindishdist = mindishdist 
		if math.fabs(mindishdist-minzonedist) < 200:
			minzonedist = 10000000000000000000000000000000000
			radii = []
			for f in range(0, len(c)):
				xdist = c[f][0][0] - discs[inzone[0]][0]
				ydist = c[f][0][1] - discs[inzone[0]][1]
				totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
				if totaldist+80 > mindishdist:
					radii.append(totaldist)
					if(totaldist < minzonedist):
						minzonedist = totaldist
						zonecoord = (c[f][0][0], c[f][0][1])
		



		avgradius = np.mean(radii)
		radleg = int(math.sqrt((avgradius*avgradius)/2))
		coord = (discs[inzone[0]][0]+radleg, discs[inzone[0]][1]-radleg)
		cv2.line(img,(discs[inzone[0]]),(coord),(0, 0, 0), 4)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
		cv2.circle(img, discs[inzone[0]], int(avgradius), (255, 0, 0), 7)
		text = "diameter " + str(avgradius) + " mm "
		cv2.putText(img, text, (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
	elif(len(inzone)>1):
		sharedzone = {}
		coord = {}
		dishedges = []
		dishindex = 0

		for e in range(0, len(inzone)):
			minzonedist = 10000000000000000000000000000000000
			mindishdist = minzonedist
			for f in range(0, len(c)):
				xdist = c[f][0][0] - discs[inzone[e]][0]
				ydist = c[f][0][1] - discs[inzone[e]][1]
				totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
				if(totaldist < minzonedist):
					minzonedist = totaldist
					zonecoord = (c[f][0][0], c[f][0][1])
				
			for g in range(0, len(maxarea)):
				xddist = maxarea[g][0][0] - discs[inzone[e]][0]
				yddist = maxarea[g][0][1] - discs[inzone[e]][1]
				totaldishdist = math.sqrt((xddist*xddist) + (yddist*yddist))
				if(totaldishdist < mindishdist):
					mindishdist = totaldishdist
					dishcoord = (maxarea[g][0][0], maxarea[g][0][1])
					sharedzone[e+1] = mindishdist
			

			coord[e] = zonecoord
			sharedzone[e+1] = minzonedist

			if math.fabs(mindishdist-minzonedist) < 250:
				minzonedist = 10000000000000000000000000000000000
				for f in range(0, len(c)):
					xdist = c[f][0][0] - discs[inzone[e]][0]
					ydist = c[f][0][1] - discs[inzone[e]][1]
					totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
					if(totaldist+80 > mindishdist):
						if(totaldist < minzonedist):
							minzonedist = totaldist
							zonecoord = (c[f][0][0], c[f][0][1])
					else:
						dishedges.append((c[f][0][0], c[f][0][1]))


				coord[e] = zonecoord
			coord[e] = zonecoord
			sharedzone[e+1] = minzonedist
		dishpoints = []
		for x in range(0, len(c)):
			if (c[x][0][0],c[x][0][1]) not in dishedges:
				dishpoints.append((c[x][0][0],c[x][0][1]))

		distsums = []
		pointnum = []
		pointindex = np.array([0,0,0,0,0,0])
		sumindex = np.array([0,0,0,0,0,0])
		for h in range(0, len(dishpoints)):
			distances = []
			for i in range(0, len(inzone)):
				xdist = dishpoints[h][0] - discs[inzone[i]][0]
				ydist = dishpoints[h][1] - discs[inzone[i]][1]
				totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
				distances.append(totaldist)
			distsums.append(np.amin(distances))
			pointnum.append(np.argmin(distances))
		
		for j in range(0, len(dishpoints)-1):

			sumindex[pointnum[j]] = sumindex[pointnum[j]] + distsums[j]
			pointindex[pointnum[j]] = pointindex[pointnum[j]] + 1

		print(type(sumindex))
		averages = np.divide(sumindex, pointindex)

		for k in range(0, len(inzone)):

			radleg = int(math.sqrt((averages[k]*averages[k])/2))
			coord = (discs[inzone[k]][0]+radleg, discs[inzone[k]][1]-radleg)
			cv2.line(img,(discs[inzone[k]]),(coord),(0, 0, 0), 4)
			cv2.circle(img, discs[inzone[k]], int(averages[k]), (255, 0, 0), 10)
			cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
			text = "diameter " + str(averages[k]) + " mm "
			cv2.putText(img, text, (discs[inzone[k]][0] - 20, discs[inzone[k]][1] - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


print('matchdiscs', matchdiscs)
if matchdiscs == len(discs):
	
	cv2.imshow("Image", img)
	cv2.waitKey(0)

else:

	print('test14090-', discs)
	imgnew = cv2.imread(wordup)
	imgnewblur = cv2.GaussianBlur(imgnew, (15,15), 0)
	imgnewgray = cv2.cvtColor(imgnewblur, cv2.COLOR_BGR2GRAY)
	testcase = cv2.cvtColor(imgnewblur, cv2.COLOR_BGR2GRAY)

	shapes={}

#individual edges found in the image
	points={}

#image at a given threshold value
	threshphoto={}

#the contour photo
	edgephoto={}

	v=0
	index = []

#dictionary to hold the information about the 255 images

	petriDishSize = 100
	width, height = img.shape[:2]
	petriDishConversion = petriDishSize / height
	print(len(discs))
#Conversion Constants
#Loop to Iterate through multiple Binary Threshold Functions
	for x in range(0, 256):
	#Makes an image for each threshold value and stores it in the corresponding thresh index
		retval3,threshphoto[x] = cv2.threshold(imgnewgray,x,255,cv2.THRESH_BINARY)
#Loop to store all the points and shapes of the hiercarchy.
	for x in range(0, 256):
		edgephoto[x], contours, hierarchy = cv2.findContours(threshphoto[x],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		v = 0
	#Contour approximation
		for c in contours:

		#compute the center of the contour
			inzone = []

			M = cv2.moments(c)
			# compute the area of each contour
			Area = cv2.contourArea(c)
			Areamm = int(Area * petriDishConversion * petriDishConversion) 
			diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)
			AreaR = math.sqrt(Area/math.pi)
			CircR = cv2.arcLength(c, True)/(math.pi*2)
			radii = []
			minzonedist = 1000000000000000000000
			maxzonedist = 0
			if(M["m00"] != 0 and (50 > diametermm > 4)):
				topx = 0
				topy = 0
				botx = 100000000000000
				boty = 100000000000000
				for a in range(0, len(c)):
						if(c[a][0][0] > topx):
							topx = c[a][0][0]
						if(c[a][0][1] > topy):
							topy = c[a][0][1]
						if(c[a][0][0] < botx):
							botx = c[a][0][0]
						if(c[a][0][1] < boty):
							boty = c[a][0][1]
				
				for b in range(1, len(discs)+1):
					dist = cv2.pointPolygonTest(c,(discs[b]),False)
					if(dist > 0.0):
						inzone.append(b)
				if(len(inzone) == 1): 
					if (1.25/1) > M['mu20']/M['mu02'] > (1/1.25):
						v = v + 1
				else: 
					v = v + len(inzone)
					if(x == 146):
						print('sup', inzone, len(inzone))

			# Checks the area, makes sure it's "disk sized", then compares it's circularity, by making sure it has even varianc
		
		index.append(v)
	
	print(np.amax(index))
	print(index)
	bestthresh = np.argmax(index)
		#Looks at what threshold has the most contours

	if(bestthresh == 0):
		bestthresh = 127
	print('look i work', bestthresh)

	ret3,otsuThresh = cv2.threshold(imgnewgray,bestthresh,255,cv2.THRESH_BINARY)
	invOtsuThresh = cv2.bitwise_not(otsuThresh)
	ret0,TestThresh = cv2.threshold(testcase,bestthresh,255,cv2.THRESH_BINARY)

#Step 3 - Contour detection and display
	im3, contours, hierarchy = cv2.findContours(invOtsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im0, contours0, hierarchy0 = cv2.findContours(TestThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Find the corners
	lst_intensities = []

	cv2.drawContours(imgnew, contours, -1, (0, 255, 0), 2)
	cv2.imshow("Image", invOtsuThresh)
	cv2.waitKey(0)
# For each list of contour points...
	
#Commented out for now - this is to split apart the shapes

# Petri Dish pixel to mm conversion
	a = 0
#Step 4 - loop over the contours, finding the moments
	for c in contours:
		inzone = []
		a = a + 1
		#compute the center of the contour
		M = cv2.moments(c)
			# compute the area of each contour
		Area = cv2.contourArea(c)
		Areamm = int(Area * petriDishConversion * petriDishConversion) 
		diametermm = int((((3.14 * Area)/4) ** 0.5) * petriDishConversion)


		if(M["m00"] != 0 and (50 > diametermm > 4)):
			#print('functioning', lst_intensities[a])
			topx = 0
			topy = 0
			botx = 100000000000000
			boty = 100000000000000
			for a in range(0, len(c)):
				if(c[a][0][0] > topx):
					topx = c[a][0][0]
				if(c[a][0][1] > topy):
					topy = c[a][0][1]
				if(c[a][0][0] < botx):
						botx = c[a][0][0]
				if(c[a][0][1] < boty):
					boty = c[a][0][1]
			for b in range(1, len(discs)+1):
				dist = cv2.pointPolygonTest(c,(discs[b]),False)
				if(dist > 0):
					inzone.append(b)
				#if(topx > discs[b][0] and discs[b][0] > botx and topy > discs[b][1] and discs[b][1] > boty):
				#	inzone.append(b)
	
				
			# Checks the area, makes sure it's "disk sized", then compares it's circularity, by making sure it has even variance

		matchdiscs = matchdiscs + len(inzone)

		if(len(inzone) == 1):

			minzonedist = 10000000000000000000000000000000000
			mindishdist = minzonedist
			radii = []
			avgradius = 0
			for f in range(0, len(c)):
				xdist = c[f][0][0] - discs[inzone[0]][0]
				ydist = c[f][0][1] - discs[inzone[0]][1]
				totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
				radii.append(totaldist)
				if(totaldist < minzonedist):
					minzonedist = totaldist
					zonecoord = (c[f][0][0], c[f][0][1])

			for g in range(0, len(maxarea)):
				xddist = maxarea[g][0][0] - discs[inzone[0]][0]
				yddist = maxarea[g][0][1] - discs[inzone[0]][1]
				totaldishdist = math.sqrt((xddist*xddist) + (yddist*yddist))
				if(totaldishdist < mindishdist):
					mindishdist = totaldishdist
					dishcoord = (maxarea[g][0][0], maxarea[g][0][1])
					

			coord = zonecoord

			if math.fabs(mindishdist-minzonedist) < 200:
				minzonedist = 10000000000000000000000000000000000
				radii = []
			for f in range(0, len(c)):
				xdist = c[f][0][0] - discs[inzone[0]][0]
				ydist = c[f][0][1] - discs[inzone[0]][1]
				totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
				if totaldist+60 > mindishdist:
					radii.append(totaldist)
					if(totaldist < minzonedist):
						minzonedist = totaldist
						zonecoord = (c[f][0][0], c[f][0][1])

			if(np.max(radii)-np.min(radii) < 100):
				avgradius = np.mean(radii)
				radleg = int(math.sqrt((avgradius*avgradius)/2))
				coord = (discs[inzone[0]][0]+radleg, discs[inzone[0]][1]-radleg)
				cv2.line(img,(discs[inzone[0]]),(coord),(0, 0, 0), 4)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				cv2.drawContours(imgnew, [c], -1, (0, 255, 0), 2)
				cv2.circle(imgnew, discs[inzone[0]], int(avgradius), (255, 0, 0), 7)
				text = "diameter " + str(avgradius) + " mm "
				cv2.putText(imgnew, text, (cX - 20, cY - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

		elif(len(inzone)>1):
			sharedzone = {}
			coord = {}
			dishedges = []
			dishindex = 0

			for e in range(0, len(inzone)):
				minzonedist = 10000000000000000000000000000000000
				mindishdist = minzonedist
				for f in range(0, len(c)):
					xdist = c[f][0][0] - discs[inzone[e]][0]
					ydist = c[f][0][1] - discs[inzone[e]][1]
					totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
					if(totaldist < minzonedist):
						minzonedist = totaldist
						zonecoord = (c[f][0][0], c[f][0][1])
						
				for g in range(0, len(maxarea)):
					xddist = maxarea[g][0][0] - discs[inzone[e]][0]
					yddist = maxarea[g][0][1] - discs[inzone[e]][1]
					totaldishdist = math.sqrt((xddist*xddist) + (yddist*yddist))
					if(totaldishdist < mindishdist):
						mindishdist = totaldishdist
						dishcoord = (maxarea[g][0][0], maxarea[g][0][1])
						sharedzone[e+1] = mindishdist
					

				coord[e] = zonecoord
				sharedzone[e+1] = minzonedist

				if math.fabs(mindishdist-minzonedist) < 250:
					minzonedist = 10000000000000000000000000000000000
					for f in range(0, len(c)):
						xdist = c[f][0][0] - discs[inzone[e]][0]
						ydist = c[f][0][1] - discs[inzone[e]][1]
						totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
						if(totaldist+90 > mindishdist):
							if(totaldist < minzonedist):
								minzonedist = totaldist
								zonecoord = (c[f][0][0], c[f][0][1])
						else:
							dishedges.append((c[f][0][0], c[f][0][1]))
								#dishindex = dishindex + 1

					coord[e] = zonecoord
				coord[e] = zonecoord
				sharedzone[e+1] = minzonedist
			
			dishpoints = []
			for x in range(0, len(c)):
				if (c[x][0][0],c[x][0][1]) not in dishedges:
					dishpoints.append((c[x][0][0],c[x][0][1]))

			distsums = []
			pointnum = []
			pointindex = np.array([1,1,1,1,1,1])
			sumindex = np.array([1,1,1,1,1,1])
			for h in range(0, len(dishpoints)):
				distances = []
				for i in range(0, len(inzone)):
					xdist = dishpoints[h][0] - discs[inzone[i]][0]
					ydist = dishpoints[h][1] - discs[inzone[i]][1]
					totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
					distances.append(totaldist)
	
				distsums.append(np.amin(distances))
				pointnum.append(np.argmin(distances))
				
			for j in range(0, len(dishpoints)-1):

				sumindex[pointnum[j]] = sumindex[pointnum[j]] + distsums[j]
				pointindex[pointnum[j]] = pointindex[pointnum[j]] + 1

			averages = np.divide(sumindex, pointindex)

			for k in range(0, len(inzone)):

				radleg = int(math.sqrt((averages[k]*averages[k])/2))
				coord = (discs[inzone[k]][0]+radleg, discs[inzone[k]][1]-radleg)
				cv2.line(imgnew,(discs[inzone[k]]),(coord),(0, 0, 0), 4)
				cv2.circle(imgnew, discs[inzone[k]], int(averages[k]), (255, 0, 0), 10)
				cv2.drawContours(imgnew, [c], -1, (0, 255, 0), 2)
				text = "diameter " + str(averages[k]) + " mm "
				cv2.putText(imgnew, text, (discs[inzone[k]][0] - 20, discs[inzone[k]][1] - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

	cv2.imshow("Image", imgnew)
	cv2.waitKey(0)