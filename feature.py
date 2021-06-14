#!/usr/bin/env python2
#coding=utf-8
import numpy as np
import cv2 as cv
import time
import os
import matplotlib.pyplot as plt
global c,query_count,a,b,d,left ,safe,right

MIN_MATCH_COUNT = 10
DIR = 'img_query/'
img2 = cv.VideoCapture('2.avi') #trainImage
#fourcc = cv.VideoWriter_fourcc('X','V','I','D')
#out = cv.VideoWriter('record.avi',fourcc,30.0,(660,385))
count = 0
av_1 = 0
av_2 = 0
text_count = 0
first_flag = False
time_set = 4
img_count = 0
h = 385
w = 660
img_cnt = 0
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

while (img2.isOpened()):
	if img_count <= 750:
		img1 = cv.imread('1417.PNG',0)
		tar_x = 375
		tar_y = 252
		'''img1 = cv.imread('9.PNG',0)
		tar_x = 420
		tar_y = 212'''
		img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
		cv.rectangle(img1,(tar_x-40,tar_y-40),(tar_x+40,tar_y+40),(0, 0, 255),2)
	elif img_cnt > 750 and img_cnt <= 1400:
		img1 = cv.imread('17.PNG',0)
		tar_x = 450
		tar_y = 241
		img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
		cv.rectangle(img1,(tar_x-40,tar_y-40),(tar_x+40,tar_y+40),(0, 0, 255),2)
	elif img_cnt > 1400:
		img1 = cv.imread('1417.PNG',0)
		tar_x = 375
		tar_y = 252
		img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
		cv.rectangle(img1,(tar_x-40,tar_y-40),(tar_x+40,tar_y+40),(0, 0, 255),2)
	
	print(img_cnt)
	img1 = translate(img1, -20, 0)
	ret, frame = img2.read()
	img_count = img_count + 1
	cv.imwrite('image_test/'+str(img_count) + '.PNG',frame)
	frame = cv.imread('image_test/'+str(img_count) + '.PNG',cv.IMREAD_GRAYSCALE)
	#img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
	#cv.rectangle(img1,(tar_x-40,tar_y-40),(tar_x+40,tar_y+40),(0, 0, 255),2)
	if (img_cnt<=750 and img_cnt>=0) or (img_cnt<=1019 and img_cnt>750) or (img_cnt<=1420 and img_cnt>1400):
		frame = frame[h/5:h*4/5,w/4:w*3/4]

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
			M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
			if M is None:
				continue
			else:
				matchesMask = mask.ravel().tolist()
				#print(matchesMask)
				h,w ,_= img1.shape
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
				
			#print(list_kp1)
			for i in range(len(list_kp1)):
				x = list_kp1[i][0]
				y = list_kp1[i][1]
				delta_x.append(x-tar_x)
				delta_y.append(y-tar_y)
			if len(list_kp1) == 0 or len(delta_x)==0 or len(delta_y)==0:
				print("NO FEATURE POINT")
				continue
			for cnt in delta_x:
				av_x = av_x + cnt
			for cnt1 in delta_y:
				av_y = av_y + cnt1
			else:
				av_x = av_x/len(delta_x)
				av_y = av_y/len(delta_y)
				print(av_x,av_y)		
				if av_x > 40:
					text = "<----"
				elif av_x < 40 and av_x > -40 and av_x != 0:
					
					text = "Inside target area"
				elif av_x < -40:
					text = "---->"
			
		else:
			#print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
			matchesMask = None

		img_cnt = img_cnt +1
		#cv.drawMatchesKnn expects list of lists as matches.
		#if len(good) >0:
		draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,
	flags = 2)
		#else:
			#continue
		#if (img_cnt<=350 and img_cnt>0) or (img_cnt<=1019 and img_cnt>750) or (img_cnt<=1435 and img_cnt>1370):
		img3 = cv.drawMatches(img1,kp1,frame,kp2,good_without_list,None,**draw_params)
		if len(good)>MIN_MATCH_COUNT:
			cv.putText(img3, text, (30, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)
		else:
			text = "Not enough matches are found"
			cv.putText(img3, text, (30, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)
	else:
		if img_cnt >= 1420:
			text = "landing"
			img3 = frame
			img3 = cv.cvtColor(img3,cv.COLOR_GRAY2BGR)
			cv.putText(img3, text, (30, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)
		else:
			text = "forward"
			img3 = frame
			img3 = cv.cvtColor(img3,cv.COLOR_GRAY2BGR)
			#img3 = cv.drawMatches(img1,None,frame,None,None,None)
			cv.putText(img3, text, (30, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 1)
		img_cnt = img_cnt +1
	cv.namedWindow("image",0)
	cv.imshow('image',img3)
	cv.waitKey(1)
	#cv.destroyAllWindows()
	
