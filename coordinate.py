import cv2  
import numpy as np  
  
img = cv2.imread('1417.PNG',cv2.IMREAD_COLOR)  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

img2 = cv2.VideoCapture('20200226_realsense_Trim.avi',0) #trainImage
#SIFT  
detector = cv2.xfeatures2d.SIFT_create()
keypoints = detector.detect(gray,None)  
cv2.drawKeypoints(gray,keypoints,img)  

points2f = cv2.KeyPoint_convert(keypoints)
cv2.imshow('img',img)
cv2.waitKey(0)

img2.release()
cv2.destroyAllWindows()


