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
wordup = 'image2018-06-21_054256.jpg'
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

#The array to be used to store the contour that represents the petri dish
for d in dishape:

	if cv2.contourArea(maxarea) < cv2.contourArea(d):
		maxarea = d
#Occasionally there are spurious contours found in the thresholded image primarily used to find the dish. 
#This makes sure "maxarea" represents the petri dish



# line means the code is repeated, the character chosen is arbitrary, but same character almost the same code	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
v = 0
# V represents the number of discs

for c in contours:

	M = cv2.moments(c)
	
	Area = cv2.contourArea(c) * pxtomm
	circumfrence = cv2.arcLength(c, True)*pxtomm
	AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
	CircR = circumfrence/(math.pi*2)
	diametermm = AreaR*2
	#Pixel to mm co
	if(M["m00"] != 0) and (7 >= diametermm >= 5) and (1.5/1) > M['mu20']/M['mu02'] > (1/1.5):
	# Checks that the contours found are of the right size
		print('arearadius', Area, cv2.contourArea(c))
		print('circumfrence', CircR)
		print('areaarea', AreaR)
		#The M["mu02"] and M["mu20"] are the standard deviation of the points in the x and y direction respectively. 
	#They should be about equal for any circle. This is another check to prevent finding spurrious contours
		
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	#Finds the center point of the circle using moments
	
		v = v+1
		discs[v] = cX, cY
		dindex.append(c)
	#Records that a disc has been found, and records data about it. It's coordinates are stored in the  disc dictionary
	#It's contour information is stored in the dindex array

		cv2.drawContours(img, [c], -1, (0, 255, 0), 5)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cv2.imshow('Image', img)
cv2.waitKey(0)

bestthresh = np.mean(val) + np.std(val)
if (len(discs) != discnum):
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
		
			if(M["m00"] != 0) and diametermm > 5 and 10 > diametermm and (2/1) > M['mu20']/M['mu02'] > (1/2) and cv2.arcLength(c, True) < 1000:
				#Slightly more stringent qualifications for discs, as there will be more false positives and extreme thresholds
				if(x == 199):
					print('value', diametermm)
					print(cv2.arcLength(c, True))
				v = v+1
				# counts the number of discs found
		index.append(v)
		#Records the discs found in each threshold, index[x] gives the number of discs found at threshold x

	goodthresh = []
	#best thresh = best threshold number ... kinda self explanatory
	bestthresh = 0
	z = 0
	for y in range (0, len(index)):
		if(index[y] == discnum):
			goodthresh.append(y)
			bestthresh = y
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

	cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

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

		if(M["m00"] != 0) and (7 >= diametermm >= 5 ) and 6 > M['mu20']/M['mu02'] > (1/6):
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

matchdiscs = 0
sharedzone = {}
# match discs records the number of discs found within a contour
#shared zone is used for large contours with multiple discs found within, typically present when ZOE's overlap

#

for c in contours:

	inzone = []
	#Records the discs found within the contour

	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	M = cv2.moments(c)
	
	Area = cv2.contourArea(c) * pxtomm
	circumfrence = cv2.arcLength(c, True)*pxtomm
	AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
	CircR = circumfrence/(math.pi*2)
	diametermm = AreaR*2	
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	if(M["m00"] != 0 and (75 > diametermm > 7)):
	# Checks that there's a contour of the right size
		for b in range(1, len(discs)+1):
			dist = cv2.pointPolygonTest(c,(discs[b]),False)
			if(dist > 0):
				inzone.append(b)
		print(discs)
	#Sees what discs are contained in the contour

	matchdiscs = matchdiscs + len(inzone)
	# Records the numbers of discs stored in the contours
	#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

	if(len(inzone) > 0):
	#Checks to see if the contours have a disc within

		print(inzone)
		sharedzone = {}
		coord = {}
		dishedges = []
	# Sharedzone resets for each contour
	#coord stores the coordinates for the average the point on the contour the average distance of the radius
	#dishedges holds the points on the contour that overlap or are near the petri dish
		for e in range(0, len(inzone)):
		#Loops for every disc found within a contour

			minzonedist = 10000000000000000000000000000000000
			mindishdist = minzonedist
			#Min distances start large, Minimum zone distance is the minimum distance from the center of the disc to the ege of the contour
			#min dish dist is the minimum distance from the center of the disc to the edge of the petri dish

		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			for f in range(0, len(c)):
				#Loops through every point of the contour
				xdist = c[f][0][0] - discs[inzone[e]][0]
				ydist = c[f][0][1] - discs[inzone[e]][1]
				totaldist = math.sqrt(math.pow(xdist, 2) + math.pow(ydist, 2))
				#Finds x and y distance from both points, then uses pythagoras theorum to find the distance between the points

				if(totaldist < minzonedist):
					minzonedist = totaldist
				#Checks to see if the distance from the center of the disc is closer than the minimum, if it is it becomes the new mindist
		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


		#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++		
			for g in range(0, len(maxarea)):
				#Loops through every point of the contour representing the petri dish
				xddist = maxarea[g][0][0] - discs[inzone[e]][0]
				yddist = maxarea[g][0][1] - discs[inzone[e]][1]
				totaldishdist = math.sqrt((xddist*xddist) + (yddist*yddist))
				#Same as the previous loop, but with the petri dish instead
				if(totaldishdist < mindishdist):
					mindishdist = totaldishdist
					sharedzone[e+1] = mindishdist
				#If a new minimum distance from the petri dish is found it's recorded and put in shared zone.
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	

			print(minzonedist, mindishdist, 'look at me')
			sharedzone[e+1] = minzonedist

			if math.fabs(mindishdist-minzonedist) < 200:
			#Occasinally the contours that are cut of by the petri dish don't overlap with the petri dish so a buffer is needed
				minzonedist = 10000000000000000000000000000000000

			#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
				for f in range(0, len(c)):
					xdist = c[f][0][0] - discs[inzone[e]][0]
					ydist = c[f][0][1] - discs[inzone[e]][1]
					totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
					if(totaldist + 40 > mindishdist):
						if(totaldist < minzonedist):
						#The difference when finding the minimum distance from the contour to the disc is that the contour points that are overlapping
						#with the petri dish are disregarded, with an 80 pixel buffer, currently this buffer is hardcoded, this can change
							minzonedist = totaldist
			#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
					else:
						dishedges.append((c[f][0][0], c[f][0][1]))
					#contour points too near the petri dish edge are added to dishedges and considered inaccurate to the reading of the disc.

			sharedzone[e+1] = minzonedist
		dishpoints = []

		for x in range(0, len(c)):
			if (c[x][0][0],c[x][0][1]) not in dishedges:		
				dishpoints.append((c[x][0][0],c[x][0][1]))
		#This loop goes through the contours and creates an array that holds all the points that aren't parts of the zone cut of by the edge ofthe dish

		distsums = []
		pointnum = []
		pointindex = np.array([1,1,1,1,1,1,1,1,1])
		sumindex = np.array([1,1,1,1,1,1,1,1,1])
		#distsums is an array sum of the minimum distances from each disc to the edge of the contour
		#pointnum contains the disc number corresponding to the disc. i.e. in distsum[x] there is a pixel value like 400 
		# and in pointnum[x] it will say 3 indicating that discs[3] has a point that is closest to it that is 400 pixels away
		#pointindex and sumindex are arrays that contain the number of points each disc has and the sum of those points respectively
		#pointindex and sumindex are hardcoded, I don't like that, but making them dictionary caused python to through a fit, fix later

		for h in range(0, len(dishpoints)):
		#goes through each point of a contour
			distances = []
			#creates an array to hold the distances from each disc to the contour

		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			for i in range(0, len(inzone)):
				xdist = dishpoints[h][0] - discs[inzone[i]][0]
				ydist = dishpoints[h][1] - discs[inzone[i]][1]
				totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))

		#Instead of there being 1 minimum point found each is recorded in their respective array, see explanation of arrays previously stated
				distances.append(totaldist)
			distsums.append(np.amin(distances))
			pointnum.append(np.argmin(distances))
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		for j in range(0, len(dishpoints)-1):
			sumindex[pointnum[j]] = sumindex[pointnum[j]] + distsums[j]
			pointindex[pointnum[j]] = pointindex[pointnum[j]] + 1
		#Makes it so that sum and point index have the values they are supposed to have
		#Sum index has the sum of the the distances of each point in a contour that are closest to each disc
		#pointindex has the number of contourpoints closest at each disc at each point.

		averages = np.divide(sumindex, pointindex)
		# averages contains the average distance of the contour point for each disc i.e. the average radius of the ZOE for each disc 
		#So  discs[x] average ZOE is held at averages[x]
		
		for k in range(0, len(inzone)):
		#goes through all the discs found in this respective contour, and draws them 

			print(averages, print(inzone))
			radleg = int(math.sqrt((averages[k]*averages[k])/2))
			#conputes the x/y distances of each leg of the radius to draw the line
			coord = (discs[inzone[k]][0]+radleg, discs[inzone[k]][1]-radleg)
			#A coordinate that is the average distance away from the disc given its contour information
			cv2.line(img,(discs[inzone[k]]),(coord),(0, 0, 0), 4)
			#draws a line from the center of a disc to a coordinate that is average distance away up and to the right
			cv2.circle(img, discs[inzone[k]], int(averages[k]), (255, 0, 0), 10)
			# draws a cricle representing the ZOE
			cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
			#shows contours found for proofreading
			text = "D" + str(round(averages[k]*pxtomm*2, 3)) + " mm "
			cv2.putText(img, text, (discs[inzone[k]][0] - 20, discs[inzone[k]][1] - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
			#Prints distance of ZOE, currently in px and not mm must fix

#


print('matchdiscs', matchdiscs)
if matchdiscs == len(discs):
#If a circle was drawn for each disc, it's happy, sometimes it's tricked when two contours surround a disc and one disc is missed
#similar problem later
	cv2.imshow("Image", img)
	cv2.waitKey(0)
#Shows the imaage if the total number of contours found with discs within them matches the number of discs

else:
#Graph search through all thresholds

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	print('test14090-', discs)
	imgnew = cv2.imread(wordup)
	imgnewblur = cv2.GaussianBlur(imgnew, (15,15), 0)
	imgnewgray = cv2.cvtColor(imgnewblur, cv2.COLOR_BGR2GRAY)
	testcase = cv2.cvtColor(imgnewblur, cv2.COLOR_BGR2GRAY)


#image at a given threshold value
	threshphoto={}

#the contour photo
	edgephoto={}

	index = []

#dictionary to hold the information about the 255 images


#Loop to Iterate through multiple Binary Threshold Functions
	for x in range(0, 256):
	#Makes an image for each threshold value and stores it in the corresponding thresh index
		retval3,threshphoto[x] = cv2.threshold(imgnewgray,x,255,cv2.THRESH_BINARY)
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	

	incdiscs = []
	v=0
	w=0
	
	for x in range(0, 256):
		edgephoto[x], contours, hierarchy = cv2.findContours(threshphoto[x],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		w = 0
		v = 0
		checkdisc = copy.deepcopy(discs)
		for c in contours:
		
		#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>		
			inzone = []
			M = cv2.moments(c)
			
			Area = cv2.contourArea(c) * pxtomm
			circumfrence = cv2.arcLength(c, True)*pxtomm
			AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
			CircR = circumfrence/(math.pi*2)
			diametermm = AreaR*2
		#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			
			radii = []
			minzonedist = 1000000000000000000000
			maxzonedist = 0
			#AreaR and CircR are Area radius and circumfrence if the contour is a circle they should be similar

		#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
			if(M["m00"] != 0 and (75 > diametermm > 6)):
			#checks to see if the contour is the right size
				for b in range(1, len(discs)+1):
					dist = cv2.pointPolygonTest(c,(discs[b]),False)
					if(dist > 0.0):
						inzone.append(b)
						checkdisc[b] = 1
					#checks to see the discs within the contour

				if(len(inzone) == 1): 
					if (1.25/1) > M['mu20']/M['mu02'] > (1/1.25):
						v = v + 1

				# If the contour only contains one disc, It's good to make sure it's circular, not doing so produces many errant contours

				else: 
					v = v + len(inzone)
					if(x == 146):
						print('sup', inzone, len(inzone))
				# No additional chekcs in place for contours containing multiple, may implement more soon, more complicated this way
		for d in range(1, len(checkdisc)+1):
			if(checkdisc[d] == 1):
				w = w+1

		if(w == len(checkdisc)):
			incdiscs.append(1)
		else:
			incdiscs.append(0)

			#v still functions to count discs per threshold
		#for d in range(0, len(discs)+1):

		index.append(v)
		print(v)
	#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	print(len(checkdisc))
	print(w)
	print('checkdisc', incdiscs)
	print(np.amax(index))
	print(index)
	bestthresh = 0
	#Index plays the same role
	goodthresh = []
	bestthresh = np.argmax(index)
	for x in range(0, len(index)):
		if(index[x] == np.amax(index)):
			goodthresh.append(x)
			bestthresh = x

	bestthresh = int(np.mean(goodthresh))
	if(bestthresh == 0):
		bestthresh = 127
	print('look i work', bestthresh)
	#bestthresh is the one threshold that produces the most contours that contains discs. There needs to be a better way
	#Most contours that contains more discs != finding a contour for every disc, does find double innoculation zones though
	# needs work

	ret3,otsuThresh = cv2.threshold(imgnewgray,bestthresh,255,cv2.THRESH_BINARY)
	invOtsuThresh = cv2.bitwise_not(otsuThresh)
	ret0,TestThresh = cv2.threshold(testcase,bestthresh,255,cv2.THRESH_BINARY)
	#Thresholding based on new best thresh and using a new clean imaged

	im3, contours, hierarchy = cv2.findContours(invOtsuThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im0, contours0, hierarchy0 = cv2.findContours(TestThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# other image, used for some experimenting, may delete later

	cv2.drawContours(imgnew, contours, -1, (0, 255, 0), 2)
	#CHeck to see the image, quality check 



	a = 0

#

	for c in contours:
		
		a = a + 1
		#A just counts the loop iteration number

		#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		inzone = []
		M = cv2.moments(c)
		
		Area = cv2.contourArea(c) * pxtomm
		circumfrence = cv2.arcLength(c, True)*pxtomm
		AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
		CircR = circumfrence/(math.pi*2)
		diametermm = AreaR*2
		#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


		#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		if(M["m00"] != 0 and (100 > diametermm > 6)):
			for b in range(1, len(discs)+1):
				dist = cv2.pointPolygonTest(c,(discs[b]),False)
				if(dist > 0):
					inzone.append(b)

		matchdiscs = matchdiscs + len(inzone)
		#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

		
		if(len(inzone) > 0):
			sharedzone = {}
			coord = {}
			dishedges = []

			for e in range(0, len(inzone)):
				minzonedist = 10000000000000000000000000000000000
				mindishdist = minzonedist

			#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
				for f in range(0, len(c)):
					xdist = c[f][0][0] - discs[inzone[e]][0]
					ydist = c[f][0][1] - discs[inzone[e]][1]
					totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
					if(totaldist < minzonedist):
						minzonedist = totaldist
			#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
							

			#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
				for g in range(0, len(maxarea)):
					xddist = maxarea[g][0][0] - discs[inzone[e]][0]
					yddist = maxarea[g][0][1] - discs[inzone[e]][1]
					totaldishdist = math.sqrt((xddist*xddist) + (yddist*yddist))
					if(totaldishdist < mindishdist):
						mindishdist = totaldishdist
						sharedzone[e+1] = mindishdist
			#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

				sharedzone[e+1] = minzonedist

				if math.fabs(mindishdist-minzonedist) < 250:
					minzonedist = 10000000000000000000000000000000000

			#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
					for f in range(0, len(c)):
						xdist = c[f][0][0] - discs[inzone[e]][0]
						ydist = c[f][0][1] - discs[inzone[e]][1]
						totaldist = math.sqrt((xdist*xdist) + (ydist*ydist))
						if(totaldist + 40 > mindishdist):
							if(totaldist < minzonedist):
								minzonedist = totaldist
			#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
						else:
							dishedges.append((c[f][0][0], c[f][0][1]))

				sharedzone[e+1] = minzonedist
			
			dishpoints = []
			for x in range(0, len(c)):
				if (c[x][0][0],c[x][0][1]) not in dishedges:
					dishpoints.append((c[x][0][0],c[x][0][1]))

			distsums = []
			pointnum = []
			pointindex = np.array([1,1,1,1,1,1,1,1,1])
			sumindex = np.array([1,1,1,1,1,1,1,1,1])
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
				cv2.circle(imgnew, discs[inzone[k]], int(averages[k]), (255, 0, 0), 20)
				cv2.drawContours(imgnew, [c], -1, (0, 255, 0), 2)
				text = "D" + str(round(averages[k]*pxtomm*2, 3)) + " mm "
				cv2.putText(imgnew, text, (inzone[k][0] - 20, discs[inzone[k]][1] - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)

#




	cv2.imshow("Image", imgnew)
	cv2.waitKey(0)
#Shows final image