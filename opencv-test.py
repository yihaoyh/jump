# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


# calculate the size of image
def getImgSize(image):
    height, width, channels = image.shape
    return height, width


# find the pos of actor
def findActor(srcImage):
    template = cv2.imread('actor1.png', 1)
    w, h, channel = template.shape[::-1]
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #
    methods = ['cv2.TM_CCOEFF_NORMED']
    for meth in methods:
        img = srcImage.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        print "actor position ", top_left, bottom_right;
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        if(False):
            plt.subplot(121), plt.imshow(res, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()
    return top_left, bottom_right


# get the color of image's background
def getBackgroundColor(srcImage):
    img = srcImage.copy()
    global img_hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 取图片中间左边的像素作为背景色
    return img_hsv[image_size[0]/2, 1]

def getNextPosition(img, actor_side):
    offset = 5
    pos1 = [(pos_actor[0][0] + pos_actor[1][0])/2, pos_actor[1][1]]
    result = None
    if(actor_side == 'left'):
        image_right = image_size[1]-1
        pos2 = [image_right, math.ceil(-(image_right - pos1[0])/1.732 + pos1[1]) + offset]
        for i in range(0, (image_right-pos1[0])):
            pixel = img[int(pos2[1] + i)][int(math.floor(pos2[0] - 1.732*i))]
            if(deltaOfUint(pixel[2], bg_hsv[2]) > 20):
                result = math.floor(pos2[0] - 1.732*i), pos2[1] + i
                print result
                break
            pixel = img[int(pos2[1] + i)][int(math.ceil(pos2[0] - 1.732 * i))]
            if(deltaOfUint(pixel[2], bg_hsv[2]) > 20):
                result = math.ceil(pos2[0] - 1.732*i), pos2[1] + i
                print result
                break
        result = result[0] - 25, result[1]

    if(actor_side == 'right'):
        image_left = 0
        pos2 = [image_left, math.ceil(-pos1[0]/1.732 + pos1[1]) + offset]
        for i in range(0, pos1[0]):
            pixel = img[int(pos2[1] + i)][int(math.floor(1.732*i))]
            if(deltaOfUint(pixel[2], bg_hsv[2]) > 20):
                result = math.floor(1.732*i), pos2[1] + i
                print result
                break
            pixel = img[int(pos2[1] + i)][int(math.ceil(1.732 * i))]
            if(deltaOfUint(pixel[2], bg_hsv[2]) > 20):
                result = math.ceil(1.732*i), pos2[1] + i
                print result
                break
        result = result[0] + 25, result[1]
    return result

def deltaOfUint(val1, val2):
    return abs(int(val1) - int(val2))

image_size = None
bg_hsv = None
pos_actor = None
img_hsv = None
# Load an color image in grayscale
# img = cv2.imread('IMG_1629.PNG',0)
# print 'test'
images = ['IMG_1629.PNG', 'IMG_1630.PNG', 'IMG_1634.PNG']
# images = ['IMG_1629.PNG']
for path in images:
    img = cv2.imread(path, 1)
    # cv2.imshow('image', img)
    image_size = getImgSize(img)
    print "image size is ", image_size

    pos_actor = findActor(img)
    print pos_actor

    bg_hsv = getBackgroundColor(img);
    print "bg color in hsv is ", bg_hsv

    if(pos_actor[0][0] < image_size[1]/2):
        print "actor is in left side"
        nextPos = getNextPosition(img_hsv, 'left')
        print "scale ",(nextPos[0] - pos_actor[0][0])/image_size[0]
    else:
        print "actor is in right side"
        nextPos = getNextPosition(img_hsv, 'right')
        print "scale ", (pos_actor[0][0] - nextPos[0])/image_size[0]


# images = [img, thresh1]
# for i in xrange(1):
# plt.subplot(1, 2, i+1),plt.imshow(images[i],'gray')
# cv2.imshow('image', img)

# ret,thresh1 = cv2.threshold(img,160,255,cv2.THRESH_BINARY)

# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                             cv2.THRESH_BINARY, 11, 2)
# cv2.imshow('image', th3)

# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# img = th3

#### find circle
# circles = cv2.HoughCircles(th3, cv2.HOUGH_GRADIENT,1,50,
#                             param1=20,param2=21,minRadius=30,maxRadius=100)
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
#######    

# closed = cv2.erode(th3, None, iterations=1)
# # closed = cv2.dilate(closed, None, iterations=2)
# ret, binary = cv2.threshold(th3, 127, 255, cv2.THRESH_BINARY)

# image ,contours,hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print (len(contours))  

# cv2.drawContours(img, contours, -1, (255,0,0),3)  


# cv2.imshow('image', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# def processCircel():
# (x,y),radius = cv2.minEnclosingCircle(contours[0])
# center = (int(x),int(y))
# radius = int(radius)
# print radius
# img = cv2.circle(img,center,radius,(0,255,0),2)
