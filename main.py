import cv2
import numpy as np
import os

path = os.getcwd()
for filename in os.listdir(path + '/input'):
   #_____imgOriginal
   imgOriginal = cv2.imread('input/'+filename)

   #_____imgHSV
   imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

   imgHSV2 = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

   #_____imgThreshLow
   lowerRed = np.array([0, 135, 135])
   upperRed = np.array([15, 255, 255])
   imgThreshLow = cv2.inRange(imgHSV, lowerRed, upperRed)
   #_____imgThreshHigh
   lowerRed = np.array([159, 135, 135])
   upperRed = np.array([179, 255, 255])
   imgThreshHigh = cv2.inRange(imgHSV, lowerRed, upperRed)

   #_____imgThresh
   imgThresh = cv2.add(imgThreshLow, imgThreshHigh)

   #_____imgThreshSmoothed
   kernel = np.ones((3, 3), np.uint8)

   imgEroded = cv2.erode(imgThresh, kernel, iterations=1)
   imgDilated = cv2.dilate(imgEroded, kernel, iterations=1)
   imgThreshSmoothed = cv2.GaussianBlur(imgDilated, (3, 3), 0)

   def translate(value, leftMin, leftMax, rightMin, rightMax):
      # Figure out how 'wide' each range is
      leftSpan = leftMax - leftMin
      rightSpan = rightMax - rightMin

      # Convert the left range into a 0-1 range (float)
      valueScaled = float(value - leftMin) / float(leftSpan)

      # Convert the 0-1 range into a value in the right range.
      return rightMin + (valueScaled * rightSpan)

   kernel_size = 69
   kernel_treshold = 0.5
   kernel = np.zeros((kernel_size, kernel_size), np.uint8)
   kernel_max = (kernel_size//2)*((2)**(1/2))
   for i in range(kernel_size):
      for j in range(kernel_size):
         i_diff = (i - kernel_size//2)**2
         j_diff = (j - kernel_size//2)**2
         val = max(translate((i_diff+j_diff)**(1/2), 0, kernel_max, 1, 0), 0)
         if(val > kernel_treshold):
            kernel[i,j] = 1
         else:
            kernel[i,j] = 0
   #print(kernel)
   imgDilated2 = cv2.dilate(imgThresh, kernel, iterations=1)
   #imgEroded = cv2.erode(imgEroded, kernel, iterations=1)
   #imgEroded = cv2.dilate(imgEroded, kernel, iterations=8)
   #imgThreshSmoothed = cv2.erode(imgEroded, kernel, iterations=1)

   """
   #_____imgCanny
   imgCanny = cv2.Canny(imgThreshSmoothed, 80, 160)

   #_____imgContours
   contours, _ = cv2.findContours(np.array(imgCanny), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   img_Contours = np.zeros_like(imgCanny)
   cv2.drawContours(img_Contours, contours, -1, (255, 255, 255), 1)

   approxContours = []

   for c in contours:
       approx = cv2.approxPolyDP(c, 10, closed=True)
       approxContours.append(approx)

   imgContours = np.zeros_like(imgCanny)
   cv2.drawContours(imgContours, approxContours, -1, (255, 255, 255), 1)

   #_____imgAllConvexHulls
   allConvexHulls = []

   for ac in approxContours:
       allConvexHulls.append(cv2.convexHull(ac))

   imgAllConvexHulls = np.zeros_like(imgCanny)
   cv2.drawContours(imgAllConvexHulls, allConvexHulls, -1, (255, 255, 255), 2)

   #_____imgConvexHulls3To10
   convexHull3To10 = []

   for ch in allConvexHulls:
       if 3 <= len(ch) <= 10:
           convexHull3To10.append(cv2.convexHull(ch))

   imgConvexHulls3To10 = np.zeros_like(imgCanny)
   cv2.drawContours(imgConvexHulls3To10, convexHull3To10, -1, (255, 255, 255), 2)

   #imgTrafficCones

   def convexHullPointingUp(ch):
       pointsAboveCenter, poinstBelowCenter = [], []

       x, y, w, h = cv2.boundingRect(ch)
       aspectRatio = w / h

       if aspectRatio < 0.8:
           verticalCenter = y + h / 2

           for point in ch:
               if point[0][1] < verticalCenter:
                   pointsAboveCenter.append(point)
               elif point[0][1] >= verticalCenter:
                   poinstBelowCenter.append(point)

           leftX = poinstBelowCenter[0][0][0]
           rightX = poinstBelowCenter[0][0][0]
           for point in poinstBelowCenter:
               if point[0][0] < leftX:
                   leftX = point[0][0]
               if point[0][0] > rightX:
                   rightX = point[0][0]

           for point in pointsAboveCenter:
               if (point[0][0] < leftX) or (point[0][0] > rightX):
                   return False

       else:
           return False

       return True

   cones = []
   bounding_Rects = []

   for ch in convexHull3To10:
       if convexHullPointingUp(ch):
           cones.append(ch)
           rect = cv2.boundingRect(ch)
           bounding_Rects.append(rect)

   imgTrafficCones = np.zeros_like(imgCanny)
   cv2.drawContours(imgTrafficCones, cones, -1, (255, 255, 255), 2)

   #imgTrafficConesWithOverlapsRemoved
   imgTrafficConesWithOverlapsRemoved = imgOriginal.copy()
   #cv2.drawContours(imgTrafficConesWithOverlapsRemoved, cones, -1, (255, 255, 255), 2)

   for rect in bounding_Rects:
       cv2.rectangle(imgTrafficConesWithOverlapsRemoved, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 2)

   #Image scaling
   imgOriginalSmall = cv2.resize(imgOriginal, (0, 0), fx=0.5, fy=0.5)
   imgHSVSmall = cv2.resize(imgHSV, (0, 0), fx=0.5, fy=0.5)
   imgThreshLowSmall = cv2.resize(imgThreshLow, (0, 0), fx=0.5, fy=0.5)
   imgThreshHighSmall = cv2.resize(imgThreshHigh, (0, 0), fx=0.5, fy=0.5)
   imgThreshHighSmall = cv2.resize(imgThreshHigh, (0, 0), fx=0.5, fy=0.5)
   imgErodedSmall = cv2.resize(imgEroded, (0, 0), fx=0.5, fy=0.5)
   #imgThreshLowSmall2 = cv2.resize(imgThreshLow2, (0, 0), fx=0.5, fy=0.5)
   #imgThreshHighSmall2 = cv2.resize(imgThreshHigh2, (0, 0), fx=0.5, fy=0.5)
   imgThreshSmall = cv2.resize(imgThresh, (0, 0), fx=0.5, fy=0.5)
   #imgThreshSmall2 = cv2.resize(imgThresh2, (0, 0), fx=0.5, fy=0.5)
   imgThreshSmoothedSmall = cv2.resize(imgThreshSmoothed, (0, 0), fx=0.5, fy=0.5)
   imgCannySmall = cv2.resize(imgCanny, (0, 0), fx=0.5, fy=0.5)
   imgContoursSmall = cv2.resize(imgContours, (0, 0), fx=0.5, fy=0.5)
   imgAllConvexHullsSmall = cv2.resize(imgAllConvexHulls, (0, 0), fx=0.5, fy=0.5)
   imgConvexHulls3To10Small = cv2.resize(imgConvexHulls3To10, (0, 0), fx=0.5, fy=0.5)
   imgTrafficConesSmall = cv2.resize(imgTrafficCones, (0, 0), fx=0.5, fy=0.5)
   imgTrafficConesWithOverlapsRemovedSmall = cv2.resize(imgTrafficConesWithOverlapsRemoved, (0, 0), fx=0.5, fy=0.5)

   #Image displaying
   cv2.imshow('imgOriginal', imgOriginalSmall)
   cv2.imshow('imgHSV', imgHSVSmall)
   cv2.imshow('imgThreshLow', imgThreshLowSmall)
   cv2.imshow('imgThreshHigh', imgThreshHighSmall)
   cv2.imshow('imgEroded', imgErodedSmall)
   #cv2.imshow('imgThreshLow2', imgThreshLowSmall2)
   #cv2.imshow('imgThreshHigh2', imgThreshHighSmall2)
   cv2.imshow('imgThresh', imgThreshSmall)
   #cv2.imshow('imgThresh2', imgThreshSmall2)
   cv2.imshow('imgThreshSmoothed', imgThreshSmoothedSmall)
   cv2.imshow('imgCanny', imgCannySmall)
   cv2.imshow('imgContours', imgContoursSmall)
   cv2.imshow('imgAllConvexHulls', imgAllConvexHullsSmall)
   cv2.imshow('imgConvexHulls3To10', imgConvexHulls3To10Small)
   cv2.imshow('imgTrafficCones', imgTrafficConesSmall)
   cv2.imshow('imgTrafficConesWithOverlapsRemoved', imgTrafficConesWithOverlapsRemovedSmall)
   """

   cv2.imwrite(os.path.join(path , "output/smoothed_"+filename), imgThreshSmoothed)
   cv2.imwrite(os.path.join(path , "output/dilated_"+filename), imgDilated2)
   cv2.imwrite(os.path.join(path , "output/"+filename), imgThresh)


   cv2.waitKey()

