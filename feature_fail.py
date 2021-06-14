#!/usr/bin/env python2
import numpy as np
import cv2 as cv
import time
import os
import matplotlib.pyplot as plt
global c,query_count,a,b,d,left ,safe,right

MIN_MATCH_COUNT = 10
DIR = 'img_query/'
img2 = cv.VideoCapture('1008.avi') #trainImage
count = 0
av_1 = 0
av_2 = 0
text_count = 0
first_flag = False
time_set = 4
img_count = 0
h = 385
w = 660
count = [0,0,0,0]
pixel_value = [[0,h/2,0,w/2],[h/2,h,0,w/2],[0,h/2,w/2,w],[h/2,h,w/2,w]]
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted
while (True):
	img1 = cv.imread('9.PNG',0)
	img1 = translate(img1, -20, 0)
	for fuckthat in range(4):
		ret, frame = img2.read()
		#frame = cv.imread('83.PNG',cv.IMREAD_COLOR)
		
		print(pixel_value[fuckthat][0],pixel_value[fuckthat][1],pixel_value[fuckthat][2],pixel_value[fuckthat][3])
		frame = frame[pixel_value[fuckthat][0] : pixel_value[fuckthat][1] , pixel_value[fuckthat][2] : pixel_value[fuckthat][3]]		
		#frame = frame[385/5:385*3/5,660/4:660*3/4]
		img_count = img_count + 1
		
		cv.imwrite('image_test/'+str(img_count) + '.PNG',frame)
		frame = cv.imread('image_test/'+str(img_count) + '.PNG',cv.IMREAD_GRAYSCALE)
		# Initiate SIFT detector
		sift = cv.xfeatures2d.SIFT_create()
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(frame,None)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		# BFMatcher with default params
		flann = cv.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1,des2,k=2)
		good = []
		good_without_list = []
		for m,n in matches:
			if m.distance < 0.55*n.distance:
				good.append([m])
				good_without_list.append(m)
			
		#RANSAC
		if len(good)>MIN_MATCH_COUNT and len(good) != 0:
			src_pts = np.float32([kp1[mat.queryIdx].pt for mat in good_without_list]).reshape(-1,1,2)
			dst_pts = np.float32([kp2[mat.trainIdx].pt for mat in good_without_list]).reshape(-1,1,2)
			#print(src_pts)
			M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
			if M is None:
				pass
			
			matchesMask = mask.ravel().tolist()
			#print(matchesMask)
			h,w = img1.shape
			pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
			dst = cv.perspectiveTransform(pts,M)
			list_kp1 = []
			delta_x = []
			delta_y = []
			#list_kp1 = [kp1[mat.queryIdx].pt for mat in good_without_list]
			av_x = 0
			av_y = 0
			src_pts = src_pts.tolist()
			for j in range(len(matchesMask)):
				if matchesMask[j] == 1:
					list_kp1.append(src_pts[j][0])
				
			print(list_kp1)
			for i in range(len(list_kp1)):
				x = list_kp1[i][0]
				y = list_kp1[i][1]
				delta_x.append(x-409)
				delta_y.append(y-237)
			if len(list_kp1) == 0 or len(delta_x)==0 or len(delta_y)==0:
				print("NO FEATURE POINT")
				continue
			for cnt in delta_x:
				av_x = av_x + cnt
		
			else:
				av_x = av_x/len(delta_x)
				av_y = av_y/len(delta_y)		
				if av_x >35:
					print("<----")
					text = "<----"
				elif av_x<35 and av_x >-35 and av_x !=0:
					print("Inside target area")
					text = "Inside target area"
				elif av_x <-35:
					print("---->")
					text = "---->"
			print(fuckthat)
			count[fuckthat] = count[fuckthat] + 1
		else:
			print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			matchesMask = None
        		
	
	
		#cv.drawMatchesKnn expects list of lists as matches.
		#if len(good) >0:
		draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,
	flags = 2)
		#else:
			#continue
		img3 = cv.drawMatches(img1,kp1,frame,kp2,good_without_list,None,**draw_params)
		'''if len(good)>MIN_MATCH_COUNT:
			#cv.putText(img3, text, (30, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)
		else:
			#text = "Not enough matches are found"
			#cv.putText(img3, text, (30, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)'''
		print(count)
		cv.namedWindow("image",0)
		t = time.time()
		while time.time() -t<=1:
			cv.imshow('image',img3)
		cv.waitKey(1)
	#cv.destroyAllWindows()
	
