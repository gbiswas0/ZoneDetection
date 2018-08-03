import os

import cv2
import numpy as np
import os.path
import math
import copy
from scipy import stats
import numpy as np
from scipy import signal
import pickle

file = 'image2018-07-04_19-40-09.jpg'

def zonefinder(filepath):
    # This file takes a file path for a petri dish and finds the Zones of inhibition for that dish, and draws them
    # It writes an image to the disc of the petri dish with zones drawns in the same location as the input image, but with zonesfound before the original image name
    # It returns the filepath to this new image, as well as an array of diameters that correspond to the image, i.e. actualdists[0] gives the diameter
    # of the zone of inhibition to disc 1 on the image, actualdiscs[1] gives disc 2, etc.

    # Import these libraries
    # import cv2
    # import numpy as np
    # import os.path
    # import math
    # import copy
    # from scipy import stats
    # FYI ZOE(s) = Zone of Inhibition(s)

    # Take an input image and do pre-processing
    wordup = filepath
    img = cv2.imread(wordup)
    imgblur = cv2.GaussianBlur(img, (15, 15), 0)
    imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # The image has to be split into hsv channels to get the value of all the pixels
    imgbgr = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
    # turn it to B/W and blur it to eliminate noise


    hue, sat, val = cv2.split(imghsv)
    width, height = img.shape[:2]
    dishblank = np.zeros((width, height, 3), np.uint8)
    # image made for the isolated dish
    pxtomm = 0.0307692
    mmtopx = 1 / pxtomm
    # conversionfactor

    retrival, dishimg = cv2.threshold(imggray, np.mean(val) - (np.std(val)), 255, cv2.THRESH_BINARY)
    # The mean value of all the pixels + the standard deviation is used to determine the usual threshold optimal for finding discs on a dish
    # Mean value minus standard deviation is ideal for finding the plate, the edges of the plate must be found
    # , as they often interfere with the zone of inhibition; more elaboration later o
    # Finds the contours of the discs
    im2, dishape, hierarch = cv2.findContours(dishimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Finds the contour of the petri dish


    # Arrays made to store information about the discs, as they are the central feature used to find ZOEs
    maxarea = dishape[0]
    M = cv2.moments(dishape[0])
    if (M["m00"] == 0):
        M["m00"] = 1
    cdX = int(M["m10"] / M["m00"])
    cdY = int(M["m01"] / M["m00"])
    dishrad = math.sqrt(cv2.contourArea(dishape[0]) / math.pi)
    # Setting defaults about the petri dish


    # The array to be used to store the contour that represents the petri dish
    for d in dishape:

        M = cv2.moments(d)

        if cv2.contourArea(maxarea) < cv2.contourArea(d):
            maxarea = d
            cdX = int(M["m10"] / M["m00"])
            cdY = int(M["m01"] / M["m00"])
            dishrad = math.sqrt(cv2.contourArea(d) / math.pi)

    cv2.circle(dishblank, (int(cdX), int(cdY)), int(dishrad), (255, 255, 255), 1)
    # Makes an image with the petri dish in white and everything else in black. Useful later to make sure the program doesn't take the dish into account
    # when finding zones


    v = 0
    discs = {}
    index = []
    imgclean = cv2.imread(wordup)
    # V represents the discs
    # discs is a dictionary to store the discs' location in the image
    # imgclean is a new img to work with

    for x in range(0, 256):
        retval3, threshphoto = cv2.threshold(imggray, x, 255, cv2.THRESH_BINARY)
        edgephoto, contours, hierarchy = cv2.findContours(threshphoto, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Thresholds and finds the contours of the image given threshold x
        v = 0
        # v is to count discs at a given threshold
        for c in contours:
            # loops through contours to finds any ones that fit the parameter of a disc

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # INFORMATION THAT MAY BE USEFUL TO BE MORE ACCURATE IN FINDING DISCS USE AT YOUR OWN DISCRETION
            # Area = cv2.contourArea(c) * pxtomm
            # circumfrence = cv2.arcLength(c, True)*pxtomm
            # AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
            # CircR = circumfrence/(math.pi*2)
            # diametermm = AreaR*2
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


            M = cv2.moments(c)
            # Establishes a moment, they're like moments in physics gives a snapshot of the image
            # Long story short they help you understand information about the contour
            if (M["m00"] != 0) and (1.2 / 1) > M['mu20'] / M['mu02'] > (1 / 1.2) and 500 < cv2.arcLength(c,
                                                                                                         True) < 800 and 15000 < cv2.contourArea(
                c) < 30000:
                # These checks need to past to find a disc, mu02 and mu20 are moments that describe the standard deviation of the contour in the x and y directions
                # m00 is the moment's size
                # These make sure the disc is the right size, has a circumfrence that seems right given the size, and is circular enough
                v = v + 1
                # counts the number of discs found
        index.append(v)
    # Records the discs found in each threshold, index[x] gives the number of discs found at threshold x



    goodthresh = []
    # best thresh = best threshold number ... kinda self explanatory
    bestthresh = 0
    z = 0
    for y in range(0, len(index)):
        if (index[y] > index[bestthresh]):
            bestthresh = y
    for z in range(0, len(index)):
        if (index[z] == index[bestthresh]):
            goodthresh.append(z)
    bestthresh = int(np.mean(goodthresh))
    # Searches through the index to find the threshold with the right number of discs, i.e. the one equal to the discnum(inputted by the user)
    # Hope to un-hardcode at some point
    if (bestthresh == 0):
        bestthresh = 127
    # Precautionary measure if the discs aren't found 127 is considered a good "default threshold"

    ret3, otsuThresh = cv2.threshold(imggray, bestthresh, 255, cv2.THRESH_BINARY)
    im3, contours, hierarchy = cv2.findContours(otsuThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = 0
    # V Serves the same purpose


    for c in contours:

        M = cv2.moments(c)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # INFORMATION THAT MAY BE USEFUL TO BE MORE ACCURATE IN FINDING DISCS USE AT YOUR OWN DISCRETION
        # Area = cv2.contourArea(c) * pxtomm
        # circumfrence = cv2.arcLength(c, True)*pxtomm
        # AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
        # CircR = circumfrence/(math.pi*2)
        # diametermm = AreaR*2
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Same contour function as before, looks for discs at the threshold that finds the most discs

        if (M["m00"] != 0) and (1.2 / 1) > M['mu20'] / M['mu02'] > (1 / 1.2) and 500 < cv2.arcLength(c,
                                                                                                     True) < 800 and 15000 < cv2.contourArea(
            c) < 30000:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            v = v + 1
            discs[v] = cX, cY
            # This time the discs are recorded in the disc disctionary at their points

    downfactor = 0.1
    Ynew = int(cdY * downfactor)
    Xnew = int(cdX * downfactor)
    Cnew = int(dishrad * downfactor)
    imgdif = cv2.resize(imgblur, (0, 0), fx=downfactor, fy=downfactor)
    height0, width0, channels0 = imgdif.shape
    hdif = 0
    wdif = 0
    # The images taken with the camera are really big, so they need to be shrunk downfactor is the downsampling factor
    # these new variables are to work with this smaller version of the image

    if (Ynew - Cnew > 0 and Xnew - Cnew > 0):
        cropimg = imgdif[Ynew - Cnew:Ynew + Cnew, Xnew - Cnew: Xnew + Cnew].copy()
        hdif = Ynew - Cnew
        wdif = Xnew - Cnew
    elif (Ynew - Cnew > 0 and Xnew - Cnew < 0):
        cropimg = imgdif[Ynew - Cnew:Ynew + Cnew, 0: Xnew + Cnew].copy()
        hdif = Ynew - Cnew
    elif (Ynew - Cnew < 0 and Xnew - Cnew > 0):
        cropimg = imgdif[0:Ynew + Cnew, Xnew - Cnew: Xnew + Cnew].copy()
        wdif = Xnew - Cnew
    else:
        cropimg = imgdif[0:Ynew + Cnew, 0:Xnew + Cnew].copy()
    # Crop image works to isolate the dish, depending on where it is in the original image.
    # this series of elif statements make sure the dish isn't hitting against an edge of the image and crop the image accordingly
    # cropping the image makes the program faster on the pi, and a better image for visualization


    hsvmask = cv2.cvtColor(cropimg, cv2.COLOR_BGR2HSV)
    # labmask = cv2.cvtColor(cropimg, cv2.COLOR_BGR2LAB)
    grayimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
    height, width, channels = cropimg.shape
    blank_image = np.zeros((height, width, 3), np.uint8)
    smalldish = copy.deepcopy(blank_image)
    # Creating new images based on the small image
    # Others may be useful in the future
    cv2.circle(smalldish, (int(width / 2), int(height / 2)), Cnew - 5, (255, 255, 255), -1)
    # Draws an image with a circle representing the petri dish on a smaller image, using the information of the shape of the petri dish based on the
    # petri dish locater at the beggining of the program



    alldish = np.zeros(shape=(1, 2), dtype=int)
    background = np.zeros(shape=(1, 2), dtype=int)
    # Arrays that represent the points where the petri dish is or isn't

    newdiscs = {}
    for z in range(1, len(discs) + 1):
        newdiscs[z] = (int(discs[z][0] * downfactor) - wdif, int(discs[z][1] * downfactor) - hdif)
        cv2.circle(smalldish, (newdiscs[z]), 10, (0, 0, 0), -1)
    # finds the new locations of the disc on this downsampled image
    # draws them on the dish

    for x in range(0, height):
        for y in range(0, width):
            if (smalldish[x][y][0] == 255 and smalldish[x][y][1] == 255 and smalldish[x][y][2] == 255):
                alldish = np.append(alldish, [[x, y]], axis=0)
            else:
                background = np.append(background, [[x, y]], axis=0)
    # Looks through the image with a white dish with black background and records those points in two arrays,
    # If a check happens outside the dish it's discounted, because it's not actually part of what would be the zone
    # Other discs are also not used to factor a theoretical zoe



    valttests = {}
    satttests = {}
    # Arrays to store t tests  of value and saturation
    r = 0
    # r is to count iteration number, it is useful when storing t tests
    for d in range(1, len(discs) + 1):
        # loops through all d discs

        valttests['disc' + str(d)] = {}
        satttests['disc' + str(d)] = {}
        # establishes  that the t tests for value and saturation will all be stored as a dictionary at a certian point at an index at point
        # in a dictionary corresponding to their disc
        r = 0
        # resetting r

        for h in range(15, 100, 2):
            r = r + 1
            # Counts up 1 at a time

            innervalues = np.array([])
            outervalues = np.array([])
            innersats = np.array([])
            outersats = np.array([])
            # Establishes arrays of an inner circle and outer circles
            # These will store the saturation and value of a two circles

            blank_image = np.zeros((height, width, 3), np.uint8)
            cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h, (255, 255, 255), 1)
            # Creates a circles of radius h of white on a c
            s = 0
            for s in range(0, background.shape[0]):
                blank_image[background[s][0]][background[s][1]][0] = 0
                blank_image[background[s][0]][background[s][1]][1] = 0
                blank_image[background[s][0]][background[s][1]][2] = 0
            # Then any point on that circle that is part of a disc, or the edge of the petri dish is written over
            # Those points do not count in finding the zoes

            for x in range(0, height):
                for y in range(0, width):
                    if (int(blank_image[x][y][0]) == 255 and int(blank_image[x][y][1]) == 255 and int(
                            blank_image[x][y][2]) == 255):
                        innervalues = np.append(grayimg[x][y], innervalues)
                        innersats = np.append(hsvmask[x][y][1], innersats)
            # This goes through the image and finds the points of the blank image that are white, and recored the saturation and value of those points


            # This next portion of the code does the same thing as the previous portion of code, but with a slightly larger circle
            blank_image = np.zeros((height, width, 3), np.uint8)
            cv2.circle(blank_image, (int(newdiscs[d][0]), int(newdiscs[d][1])), h + 4, (255, 255, 255), 1)
            s = 0
            for s in range(0, background.shape[0]):
                blank_image[background[s][0]][background[s][1]][0] = 0
                blank_image[background[s][0]][background[s][1]][1] = 0
                blank_image[background[s][0]][background[s][1]][2] = 0
            for x in range(0, height):
                for y in range(0, width):
                    if (int(blank_image[x][y][0]) == 255 and int(blank_image[x][y][1]) == 255 and int(
                            blank_image[x][y][2]) == 255):
                        outervalues = np.append(grayimg[x][y], outervalues)
                        outersats = np.append(hsvmask[x][y][1], outersats)
                        # The outer circle values and saturations are then recorded

            # Now there are four arrays innervalues and inner saturations contain the values of saturation of a small
            #  circle outervalues and outersats contain the saturation and values of a slightly bigger circle

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
            # Establishing statistics of these arrays

            innersats = innersats[abs(innersats - innersatmean) < (p * innersatstd)]
            outersats = outersats[abs(outersats - outersatmean) < (p * outersatstd)]

            innervalues = innervalues[abs(innervalues - innervalmean) < (p * innervalstd)]
            outervalues = outervalues[abs(outervalues - outervalmean) < (p * outervalstd)]
            # Removing outliers for saturation and value


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
            # Furter data processing, only for value though.
            # The saturation values are very similar, given a small size, making percentile filtering not fesable given the image size


            valttests['disc' + str(d)][r] = stats.ttest_ind(innervalues, outervalues, equal_var=False)
            satttests['disc' + str(d)][r] = stats.ttest_ind(innersats, outersats, equal_var=False)
            # Does a t test of the value and saturation of the inner circle vs the outer circle
            # The more statistically significant the, higer likelyhood the zone border is there
            # This finds the t statistic and p value for the value and saturation for a theoretical zone for 100 px

    graddists = []
    # distances of the gradient
    resizedists = []
    # array that holds the pixel distances needed for the full size image
    actualdists = []
    # holds the mm distances for each disc
    # Establishing arrays

    for a in range(1, len(discs) + 1):
        # looping through all a discs

        valuetvals = []
        valuepvals = []
        saturationtvals = []
        saturationpvals = []
        mostsignificant = []
        # more arrays, names should be self explanatory

        for b in range(1, len(valttests['disc' + str(a)])):
            valuetvals.append(valttests['disc' + str(a)][b][0])
            valuepvals.append(valttests['disc' + str(a)][b][1])
            saturationtvals.append(satttests['disc' + str(a)][b][0])
            saturationpvals.append(satttests['disc' + str(a)][b][1])
        # separates all the t and p values for a given disc

        for e in range(1, len(valuetvals)):
            if (saturationpvals[e] * 0.5 < 0.001 and valuepvals[e] * 0.5 < 0.01 and saturationtvals[e] < -2 and
                        valuetvals[e] > 2):
                mostsignificant.append((e * 2) + 15)
                # loops through t and p values for every possible zone of inhibition and then goes to find the ones that are statistically significant
                # It looks for a significant change in value and saturation i.e. where the white and lawn interact, or the ZOE

        if (len(mostsignificant) == 0):
            graddists.append(10)
            resizedists.append(int(graddists[a - 1] * (1 / downfactor)))
            actualdists.append(3 * mmtopx)
        # If there are no significant p values or t values, there is no zone of inhibition beyond the disc itself and the zone is 6 mm
        else:
            graddists.append(np.median(mostsignificant))
            resizedists.append(int(graddists[a - 1] * (1 / downfactor)))
            actualdists.append(graddists[a - 1] * (1 / downfactor))
            # Of all the significant distances the median is taken, to avoid outliers throwing of the mean, the most significant distances for a zoe
            # should be next to each other

    zonesats = np.array([])
    zonevals = np.array([])
    zonehues = np.array([])
    # More arrays to check the zone the program arrived at

    finalimg = img.copy()
    # The final image to be output
    for k in range(1, len(discs) + 1):
        # loops through discs
        X = discs[k][0]
        Y = discs[k][1]
        # Location of the disk k

        blank_image = np.zeros((height, width, 3), np.uint8)
        cv2.circle(blank_image, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(graddists[k - 1]), (255, 255, 255), -1)
        cv2.circle(blank_image, (int(newdiscs[k][0]), int(newdiscs[k][1])), int(10), (0, 0, 0), -1)
        # Draws the zone the program found on a blank image in white

        for x in range(0, height):
            for y in range(0, width):
                if (int(blank_image[x][y][0]) == 255 and int(blank_image[x][y][1]) == 255 and int(
                        blank_image[x][y][2]) == 255):
                    zonesats = np.append(hsvmask[x][y][1], zonesats)
                    zonevals = np.append(hsvmask[x][y][2], zonevals)
                    zonehues = np.append(hsvmask[x][y][0], zonehues)
        # Finds the hues saturation, and values of that zone

        m = 3
        toppercentile = 80
        botpercentile = 20

        if (graddists[k - 1] != 10):
            # If there's no zone no need to check
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
            # print(satq1, satq3, valq1, valq3)
            zonesats = zonesats[abs(zonesats) < satq3]
            zonevals = zonevals[abs(zonevals) < valq3]
            zonesats = zonesats[abs(zonesats) > satq1]
            zonevals = zonevals[abs(zonevals) > valq1]
            # more data processing of the values and saturation of this zone, taking out outliers

            if (np.mean(zonesats) < 35 and np.median(zonesats) < 35):
                # Makes sure the zone's saturation is not to high, meaning that the zones are white and not filled with something
                cv2.circle(finalimg, (X, Y), resizedists[k - 1], (0, 0, 0), 20)
                cv2.circle(finalimg, (X, Y), resizedists[k - 1], (0, 0, 255), 5)
                text = "Disc " + str(k) + ': ' + str(round((actualdists[k - 1]*pxtomm*2), 1)) + ' mm'
                cv2.putText(finalimg, text, (X - 140, Y - 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 20)
                cv2.putText(finalimg, text, (X - 140, Y - 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            # Draws circles and puts text of the zones diameter, does it twice to get a clear readable red on black text
            else:
                cv2.circle(finalimg, (X, Y), int(10 * (1 / downfactor)), (0, 0, 0), 20)
                cv2.circle(finalimg, (X, Y), int(10 * (1 / downfactor)), (0, 0, 255), 5)
                actualdists[k - 1] = (3*mmtopx)
                text = "Disc " + str(k) + ': ' + str(6) + ' mm'
                cv2.putText(finalimg, text, (X - 140, Y - 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 20)
                cv2.putText(finalimg, text, (X - 140, Y - 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                # Draws circles and puts text of the zones diameter, does it twice to get a clear readable red on black text
                # If the zone has too much stuff inside there is no zone drawn and it become 6 mm (a disc)
        else:
            cv2.circle(finalimg, (X, Y), int(10 * (1 / downfactor)), (0, 0, 0), 20)
            cv2.circle(finalimg, (X, Y), resizedists[k - 1], (0, 0, 255), 5)
            text = "Disc " + str(k) + ': ' + str(round((actualdists[k - 1]*pxtomm*2), 1))  + ' mm'
            cv2.putText(finalimg, text, (X - 140, Y - 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 20)
            cv2.putText(finalimg, text, (X - 140, Y - 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            # Draws circles and puts text of the zones diameter, does it twice to get a clear readable red on black text
            # If the zone has too much stuff inside there is no zone drawn and it become 6 mm (a disc)
        shrunkimg = cv2.resize(finalimg, (0, 0), fx=0.2, fy=0.2)
        print(actualdists)

    head, tail = os.path.split(filepath)
    if (len(head) > 0):
        # Makes it so that if you use a univeral path it writes correctly as a the split function doesn't add a slash at the end of the head string
        cv2.imwrite(head + '/zonesfound' + tail, shrunkimg)
        return ((head + '/zonesfound' + tail), actualdists, discs)
    # writes a file of an image with zones and returns the filepath along with the zone diameters in order of the zones labeled on the image
    else:
        cv2.imwrite(head + 'zonesfound' + tail, shrunkimg)
        return ((head + 'zonesfound' + tail), actualdists, discs)


# Import Libraries
def locatediscs(filepath):
    # This function takes in an image file path, finds the discs on that image writes a new image to the disc with the same path, but with 'discsfound'
    # before the title of the orignal image, and returns the file path to the new image

    # Import These Libraries
    # import cv2
    # import numpy as np
    # import os
    # import math
    # import copy
    # FYI ZOE(s) = Zone of Inhibition(s)

    # Take an input image and do pre-processing
    wordup = filepath
    # No current way to set multiple images to each other this string is used multiple points in the code for the program to see what image to read

    img = cv2.imread(wordup)
    imgblur = cv2.GaussianBlur(img, (15, 15), 0)
    imggray = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgbgr = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
    # turn it to B/W and blur it to eliminate noise
    hue, sat, val = cv2.split(imghsv)
    # The image has to be split into hsv channels to get the value of all the pixels
    width, height = img.shape[:2]
    dishblank = np.zeros((width, height, 3), np.uint8)
    pxtomm = 0.0307692
    mmtopx = 1 / pxtomm

    v = 0
    discs = {}
    discimgs = {}
    discmask = {}
    index = []
    imgclean = cv2.imread(wordup)
    verifyimg = copy.deepcopy(img)
    # V represents the discs
    # discs is a dictionary to store the discs' location in the image
    # imgclean is a new img to work with

    for x in range(0, 256):
        retval3, threshphoto = cv2.threshold(imggray, x, 255, cv2.THRESH_BINARY)
        edgephoto, contours, hierarchy = cv2.findContours(threshphoto, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Thresholds and finds the contours of the image given threshold x
        v = 0
        # v is to count discs at a given threshold
        for c in contours:
            # loops through contours to finds any ones that fit the parameter of a disc

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # INFORMATION THAT MAY BE USEFUL TO BE MORE ACCURATE IN FINDING DISCS USE AT YOUR OWN DISCRETION
            # Area = cv2.contourArea(c) * pxtomm
            # circumfrence = cv2.arcLength(c, True)*pxtomm
            # AreaR = math.sqrt(cv2.contourArea(c)/math.pi)*pxtomm
            # CircR = circumfrence/(math.pi*2)
            # diametermm = AreaR*2
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


            M = cv2.moments(c)
            # Establishes a moment, they're like moments in physics gives a snapshot of the image
            # Long story short they help you understand information about the contour
            if (M["m00"] != 0) and (1.2 / 1) > M['mu20'] / M['mu02'] > (1 / 1.2) and 500 < cv2.arcLength(c,
                                                                                                         True) < 800 and 15000 < cv2.contourArea(
                c) < 30000:
                # These checks need to past to find a disc, mu02 and mu20 are moments that describe the standard deviation of the contour in the x and y directions
                # m00 is the moment's size
                # These make sure the disc is the right size, has a circumfrence that seems right given the size, and is circular enough
                v = v + 1
                # counts the number of discs found
        index.append(v)
    # Records the discs found in each threshold, index[x] gives the number of discs found at threshold x



    goodthresh = []
    # best thresh = best threshold number ... kinda self explanatory
    bestthresh = 0
    z = 0
    for y in range(0, len(index)):
        if (index[y] > index[bestthresh]):
            bestthresh = y
    for z in range(0, len(index)):
        if (index[z] == index[bestthresh]):
            goodthresh.append(z)
    bestthresh = int(np.mean(goodthresh))
    # Searches through the index to find the threshold with the right number of discs, i.e. the one equal to the discnum(inputted by the user)
    # Hope to un-hardcode at some point
    if (bestthresh == 0):
        bestthresh = 127
    # Precautionary measure if the discs aren't found 127 is considered a good "default threshold"

    ret3, otsuThresh = cv2.threshold(imggray, bestthresh, 255, cv2.THRESH_BINARY)
    im3, contours, hierarchy = cv2.findContours(otsuThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    v = 0
    # V Serves the same purpose


    for c in contours:

        M = cv2.moments(c)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # INFORMATION THAT MAY BE USEFUL TO BE MORE ACCURATE IN FINDING DISCS USE AT YOUR OWN DISCRETION
        Area = cv2.contourArea(c) * pxtomm
        circumfrence = cv2.arcLength(c, True) * pxtomm
        AreaR = math.sqrt(cv2.contourArea(c) / math.pi) * pxtomm
        CircR = circumfrence / (math.pi * 2)
        diametermm = AreaR * 2
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Same contour function as before, looks for discs at the threshold that finds the most discs

        if (M["m00"] != 0) and (1.2 / 1) > M['mu20'] / M['mu02'] > (1 / 1.2) and 500 < cv2.arcLength(c,
                                                                                                     True) < 800 and 15000 < cv2.contourArea(
            c) < 30000:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            v = v + 1
            discs[v] = cX, cY
            text = "Disc " + str(v)
            discimgs[v] = imgclean[int(cY - (CircR * mmtopx)):int(cY + (CircR * mmtopx)),
                          int(cX - (CircR * mmtopx)): int(cX + (CircR * mmtopx))].copy()
            # Isolating the disc in a small array, which is saved to a dictionary
            cv2.drawContours(imgclean, [c], -1, (0, 0, 0), -1)
            discmask[v] = imgclean[int(cY - (CircR * mmtopx)):int(cY + (CircR * mmtopx)),
                          int(cX - (CircR * mmtopx)): int(cX + (CircR * mmtopx))].copy()
            cv2.drawContours(verifyimg, [c], -1, (0, 0, 255), 5)
            cv2.putText(verifyimg, text, (cX - 140, cY - 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    head, tail = os.path.split(filepath)
    # splits file path
    # /home/documents/zone/dish.jpeg -> head = '/home/documents/zone and' tail = 'dish.jpeg'
    # writes
    if (len(head) > 0):
        # Makes it so that if you use a univeral path it writes correctly as a the split function doesn't add a slash at the end of the head string or
        # start of the tail string
        cv2.imwrite(head + '/discsfound' + tail, verifyimg)
        return (head + '/discsfound' + tail, discs, discimgs, discmask)
    else:
        cv2.imwrite(head + 'discsfound' + tail, verifyimg)
        return (head + 'discsfound' + tail, discs, discimgs, discmask)


def zoneadjuster(zonedists, disc, increasing, magnitude):
    newdists = []
    pxtomm = 0.0307692
    mmtopx = 1 / 0.0307692

    # Takes All the zone RADII as zonedists as an array that starts from zero
    # It takes the zones as mm
    # The discs and zones whould always correspond with the exception of being in array or dictionary
    # Zonefinder returns  an array of zone radii with their index being the disc number, so you can feed that programs output into this one
    # Takes the disc's zone you want to add or subtract from
    # Increasing = true means you want to make the zone bigger
    # magnitude is how much you want to change the circle by
    if (increasing == True):
        direction = 1
    else:
        direction = -1
    # Which direction

    for x in range(0, len(zonedists)):
        if (x == disc - 1):
            # Disc nnumber is 1 higher in the array since disc 1 is stored at zonedists[0]
            newdists.append(((direction * magnitude * mmtopx)*0.51) + zonedists[x])
        else:
            newdists.append(zonedists[x])
            # converts from mm to px for circle drawer

    # Return
    return newdists


# circledrawer('imagepath', zoneadjuster(actualdists, 4, increasing=True, 1), disc_coordinates)


def circledrawer(cleanimg, distances, coordinates):
    # This function takes an petri dish image with nothing on it, and draws it's zones, given the distances and coordinates, are the same and correspond
    # If you are using my previous programs they should

    pxtomm = 0.0307692
    mmtopx = 1 / 0.0307692
    downfactor = 0.2
    # Take an input image and do pre-processing
    img = cv2.imread(str(cleanimg))
    # Cleanimg is an unaltered image
    # distances are the zone dists
    # coordinates are the disc coords in the image
    # coordinates[x] should match distances[x]

    for Z in range(0, len(distances)):
        # loops through
        X = int(coordinates[Z + 1][0])
        Y = int(coordinates[Z + 1][1])
        print(coordinates)
        pxdist = int(distances[Z])
        print(pxdist)
        # xy coords and the pixel distance established
        cv2.circle(img, (X, Y), int(pxdist), (0, 0, 0), 20)
        cv2.circle(img, (X, Y), int(pxdist), (0, 0, 255), 5)
        text = 'Disc ' + str(Z + 1) + ': ' + str(round((pxtomm * pxdist * 2), 1)) + ' mm'
        cv2.putText(img, text, (X - 140, Y - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 20)
        cv2.putText(img, text, (X - 140, Y - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    # convert back into mm and put text on the disc
    smallimg = imgdif = cv2.resize(img, (0, 0), fx=downfactor, fy=downfactor)
    head, tail = os.path.split(cleanimg)
    if (len(head) > 0):
        # Makes it so that if you use a univeral path it writes correctly as a the split function doesn't add a slash at the end of the head string or
        # start of the tail string
        cv2.imwrite(head + '/newzones' + tail, smallimg)
        return (head + '/newzones' + tail)
    else:
        cv2.imwrite(head + 'newzones' + tail, smallimg)
        return (head + 'newzones' + tail)


def featurefinder(discimgs, discmask):
    lonediscs = {}
    graydiscs = {}
    textdiscs = {}

    # Loops throuh the discs its found, and isolates them, making sure that only the disc itself is viewable
    for d in range(1, len(discmask) + 1):
        width, height = discimgs[d].shape[:2]
        graydiscs[d] = cv2.cvtColor(discimgs[d], cv2.COLOR_BGR2GRAY)
        lonediscs[d] = np.zeros((height, width, 3), np.uint8)
        # making the discs B/W, and a new disc to assign these values to
        for y in range(0, width - 1):
            for x in range(0, height - 1):
                if (discmask[d][x][y][0] == 0 and discmask[d][x][y][1] == 0 and discmask[d][x][y][2] == 0):
                    lonediscs[d][x][y][0] = graydiscs[d][x][y]
                    lonediscs[d][x][y][1] = graydiscs[d][x][y]
                    lonediscs[d][x][y][2] = graydiscs[d][x][y]
                # looping over the mask and getting rid of the background
                else:
                    lonediscs[d][x][y][0] = 255
                    lonediscs[d][x][y][0] = 255
                    lonediscs[d][x][y][0] = 255

        imghsv = cv2.cvtColor(discimgs[d], cv2.COLOR_BGR2HSV)

        hue, sat, val = cv2.split(imghsv)
        # Converting to hsv to get the value
        flattenedvalues = val.flatten()
        # Getting the value as a 1 d array for data processing
        reorganizer = np.zeros(256, np.uint8)
        # Making another array to reorganize the data
        for a in range(0, len(flattenedvalues)):
            reorganizer[flattenedvalues[a]] = reorganizer[flattenedvalues[a]] + 1
        # Reorganizing the data for the scipy peaks function
        peaks = signal.find_peaks_cwt(reorganizer, np.arange(2, 10), noise_perc=40.0)
        # finding the peaks

        if (len(peaks) != 0):
            # Finding the peaks that are low, i.e. the peaks that represent the darkness of the text
            thresh = int(np.amin(peaks) + 2)
        # Adding two so it doesn't accidentally cut of any text
        else:
            thresh = int(np.mean(flattenedvalues - np.std(flattenedvalues)))
        # If no peaks are found then proceed to take the mean - the standard deviation, this means the text is very strange

        retval, textdiscs[d] = cv2.threshold(graydiscs[d], thresh, 255, cv2.THRESH_BINARY)
    # Threshold the discs to get a binary image


    orb = cv2.ORB_create(200, scaleFactor=1.1, nlevels=20, edgeThreshold=10, firstLevel=0, WTA_K=2, patchSize=50)
    # Orb creates a descriptor
    features = {}
    keypoints = {}
    # Dictionary of features and keypoints
    for t in range(1, len(textdiscs) + 1):
        keypoints[t], features[t] = orb.detectAndCompute(textdiscs[t], None)
    # making a dictionary of features and key points, 1 for each disc
    return (features)

path, dists, coords = zonefinder(file)
newzones = zoneadjuster(dists, 1, True, 1)
print(newzones)
circledrawer(file, newzones, coords)

# return features



def discsearcher(features):
    cursor = dbh.cursor()
    # connect to database
    # execute the SQL query using execute() method.
    cursor.execute("select abx_descriptor from antibiotics")
    data = cursor.fetchall()
    # get antibiotic descriptors
    cursor.execute("select abx_name from antibiotics")
    prevdiscnames = cursor.fetchall()
    # get antibiotic names
    notfoundmessage = 'No matching antibiotic Discs found'
    # message for when nothing is found
    discnames = {}
    if (len(features) == 0):
        for d in range(1, len(features) + 1):
            discnames[d] = notfoundmessage
    else:
        # If the database is empty say nothing is found
        # If the database is empty say nothing is found

        discalignments = np.zeros(len(data), np.uint8)
        bestmatches = np.zeros(len(data), np.uint8)
        discdatapoint = np.zeros(len(features), np.uint8)
        discmatchnums = np.zeros(len(features), np.uint8)
        for e in range(0, len(data)):
            if (data[e][0] != ''):
                # loops through every known features for discs
                matchcomparison = np.array([])
                # array to store the matches
                discfile = open(data[e][0].rstrip(), "rb")
                # open the filepath to the pickle that contains the eth disc
                seendisc = pickle.load(discfile)
                # loads a file to a dictionary
                for h in range(0, len(seendisc)):
                    # loops through the dictionary containing feature set(s) for each previous disc
                    for b in range(1, len(features) + 1):
                        # loops through the features of all 6 discs given by the input
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        # matcher object used by orb
                        aligner = matcher.match(seendisc[h], features[b])
                        # matched features
                        truematches = []
                        distances = []
                        # arrays to store the matches and sdistances there of
                        for f in range(0, len(aligner)):
                            distances.append(aligner[f].distance)
                            if (aligner[f].distance < 25):
                                truematches.append(aligner[f].distance)
                        # stores the distances that pass a cutoff, i.e. true matches
                        print('stds', np.std(distances))

                        matchcomparison = np.append(matchcomparison, len(truematches))
                        # records all the matches, and number of true matches between the 6 discs
                index = np.argmax(matchcomparison)
                # gets the index of the best match
                basic = math.floor(index / len(features)) * len(features)
                matchindex = index - basic
                # find what disc it belongs to
                if (np.amax(matchcomparison) > discmatchnums[matchindex]):
                    # if this is the best match for this disc, then record it, and then put it as the candidate to beat for best match
                    discmatchnums[matchindex] = np.amax(matchcomparison)
                    discdatapoint[matchindex] = e
        print(discdatapoint)
        print(discmatchnums)

        for g in range(0, len(discdatapoint)):
            # looks through the best matches found for each disc
            if (discmatchnums[g] != 0):
                # If a match was found find it's name in the database and asign it to a dictionary at the index of the disc at the same position
                discnames[g + 1] = prevdiscnames[discdatapoint[g]][0]
            else:
                # if nothing was found say so at that index
                discnames[g + 1] = notfoundmessage
        # return a dictionary of disc names starting at one
        return (discnames)


def discwriter(discs, descriptors, filepath, ids, dosages, content):
    # cnx = mysql.connector.connect(user='root', database='incubator', password = 'Letmein!')
    # connect to database
    cursor = dbh.cursor()
    cursor.execute("select abx_name from antibiotics")
    # get names
    discnames = cursor.fetchall()
    cursor.execute("select abx_descriptor from antibiotics")
    # get descriptors
    knownfeatures = cursor.fetchall()

    cursor.execute("select abx_id from antibiotics")
    ids = cursor.fetchall()

    existance = np.zeros(len(discs), np.uint8)
    head, tail = os.path.split(filepath)
    # makes sure all discs are known
    for d in range(0, len(discs)):
        for n in range(0, len(discnames)):
            if (discs[d + 1] == discnames[n]):
                if (knownfeatures[n] != ''):
                    # if a discs name matches one found in the database
                    features = open(knownfeatures[n][0], "wb")
                    pairdisc = pickle.load(features)
                    # access its descriptors
                    pairdisc[len(pairdisc)] = descriptors[d]
                    # open the dictionary and add the new descriptor to it at len(descriptors) since it starts at zero
                    features.close()
                    # close it
                    features = open(knownfeatures[n][0], "wb")
                    # dump it at the same point
                    pickle.dump(pairdisc, features)
                    features.close()
                    existance[d] = 1
                else:
                    entirepath = ''
                    if (len(filepath) > 0):
                        entirepath = head + '/' + discs[d + 1] + '.pkl'
                    # make a file at a given folder
                    else:
                        entirepath = discs[d + 1] + '.pkl'
                    tempdict = {}
                    features = open(entirepath, "wb")
                    tempdict[0] = descriptors[d + 1]
                    # make a new dictionary and dump it in that file
                    pickle.dump(tempdict, features)
                    features.close()
                    cursor.execute("update antibiotics set abx_descriptor = " + entirepath + "where abx_id = " str(abx_id[n]))


                    # record that you found the same path

    for a in range(1, len(discs) + 1):
        if (existance[a - 1] != 1):
            entirepath = ''
            # If you don't find a matching antibiotic
            if (len(filepath) > 0):
                entirepath = head + '/' + discs[a] + '.pkl'
            # make a file at a given folder
            else:
                entirepath = discs[a] + '.pkl'
            # make a pickle file with those descriptors in a dictionary
            tempdict = {}
            features = open(entirepath, "wb")
            tempdict[0] = descriptors[a]
            # make a new dictionary and dump it in that file
            pickle.dump(tempdict, features)
            features.close()
            # push it to the database.
            cursor.execute("insert into `antibiotics`  VALUES (" + str(ids[a]) + ", " + str(discs[a]) + ", " + str(
                content[a]) + ", " + dosages[a] + ", " + entirepath + ")")


            # testimg = 'discsfoundimage2018-07-03_07-36-04-staph-a.jpg'

            # zoneimage, zonedists = zonefinder(testimg)
            # print(zoneimage, zonedists)
            # newimg, disc_coordinates, discimages, discmasks = locatediscs(testimg)
            # newpath = circledrawer(testimg, zonedists, disc_coordinates)
            # features = featurefinder(discimages, discmasks)
            # foundones = discsearcher(features)


            # Methods declaration

