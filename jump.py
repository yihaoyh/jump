# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
import time
# 斜45度投影宽高比
SCALE_XY = 3 ** 0.5

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

# 计算下一个跳跃的目标点
# 如果actor在左边，则从actor原点处到到图片最右边做一条直线，
# 设交点为pos2，则再以pos2为起点，向actor方向读取像素，如果
# 读到和背景色不一样的点，则认为找到了下一个跳跃目标的物体。
def getNextPosition(img, actor_side):
    offset = 20
    delta = 5
    result = None
    pos1 = [(pos_actor[0][0] + pos_actor[1][0]) / 2, pos_actor[1][1]]
    if(actor_side == 'left'):
        result = find_right_border(img, pos1)
        # result = result[0] - 60, result[1]

    if(actor_side == 'right'):
        result = find_left_border(img, pos1)
        # result = result[0] + 60, result[1]
    return result

def isColorDifferent(val1, val2):
    # print "color delta ", abs(int(val1) - int(val2))
    if(abs(int(val1[0]) - int(val2[0])) > 10 or abs(int(val1[1]) - int(val2[1])) > 10 or abs(int(val1[2]) - int(val2[2])) > 10):
        return True
    else:
        return False

def pull_screenshot():
    os.system('adb shell screencap -p /sdcard/autojump.png')
    os.system('adb pull /sdcard/autojump.png .')

def jump(distance):
    press_time = distance * 3000
    press_time = int(press_time)
    cmd = 'adb shell input swipe 320 410 320 410 ' + str(press_time)
    print(cmd)
    os.system(cmd)

def find_right_border(img, begin_pos):
    result = None
    offset = 40
    image_right = image_size[1] - 1
    pos2 = [image_right, math.ceil(-(image_right - begin_pos[0]) / SCALE_XY + begin_pos[1]) + offset]
    print "pos2 ", pos2
    for i in range(0, (image_right - begin_pos[0])):
        y = int(pos2[1] + i)
        x = int(math.floor(pos2[0] - SCALE_XY * i))
        pixel = img[y][x - 1]
        print "pixel ", x, y, pixel
        if (isColorDifferent(pixel, bg_hsv)):
            result = math.floor(pos2[0] - SCALE_XY * i), pos2[1] + i
            print result
            break
        pixel = img[y][x]
        if (isColorDifferent(pixel, bg_hsv)):
            result = math.ceil(pos2[0] - SCALE_XY * i), pos2[1] + i
            print result
            break
    return result

def find_left_border(img, begin_pos):
    result = None
    offset = 40
    image_left = 0
    pos2 = [image_left, math.ceil(-begin_pos[0] / SCALE_XY + begin_pos[1]) + offset]
    print "pos2 ", pos2
    for i in range(0, begin_pos[0]):
        pixel = img[int(pos2[1] + i)][int(math.floor(SCALE_XY * i))]
        if (isColorDifferent(pixel, bg_hsv)):
            result = int(math.floor(SCALE_XY * i)), pos2[1] + i
            print result
            break
        pixel = img[int(pos2[1] + i)][int(math.ceil(SCALE_XY * i))]
        if (isColorDifferent(pixel, bg_hsv)):
            result = int(math.ceil(SCALE_XY * i)), pos2[1] + i
            print result
            break
    return result

# 从内部寻找边界点
def find_border_from_inner(img, begin_pos, orientation):
    foot_length = image_size[0]/2
    result = None
    for i in range(0, foot_length, 1):
        if(orientation == 'left'):
            X = int(begin_pos[0] - i * SCALE_XY)
            Y = int(begin_pos[1] - i)
            if(not isColorDifferent(img[Y][X], bg_hsv)):
                result = [X, Y]
                break
        elif(orientation == 'right'):
            X = int(begin_pos[0] + i * SCALE_XY)
            Y = int(begin_pos[1] - i)
            if(not isColorDifferent(img[Y][X], bg_hsv)):
                result = [X, Y]
                break
    return result

image_size = None
bg_hsv = None
pos_actor = None
img_hsv = None
# Load an color image in grayscale
# img = cv2.imread('IMG_1629.PNG',0)
# print 'test'
# images = ['IMG_1629.PNG', 'IMG_1630.PNG', 'IMG_1634.PNG']
while(True):
    pull_screenshot()
    images = ['autojump.png']
    # images = ['fail3.png']
    for path in images:
        img = cv2.imread(path, 1)
        # cv2.imshow('image', img)
        image_size = getImgSize(img)
        print "image size is ", image_size

        pos_actor = findActor(img)
        print pos_actor

        bg_hsv = getBackgroundColor(img);
        print "bg color in hsv is ", bg_hsv
        nextPos = None
        scale = None
        if(pos_actor[0][0] < image_size[1]/2):
            print "actor is in left side"
            begin = getNextPosition(img_hsv, 'left')
            print "next begin ", begin
            # 向左边探测中点
            for i in range(10, image_size[1]/2):
                x = int(begin[0] - i * SCALE_XY)
                y = begin[1] + i
                border = find_border_from_inner(img_hsv, [x, y], "left")
                print "sub ", x * 2 - begin[0] - border[0]
                if(math.fabs(x * 2 - begin[0] - border[0]) < 4):
                    nextPos = [x, y]
                    scale = (nextPos[0] - pos_actor[0][0]) / float(image_size[0]);
                    break;
        else:
            print "actor is in right side"
            begin = getNextPosition(img_hsv, 'right')
            print "next begin ", begin
            # 向右边边探测中点
            for i in range(10, image_size[1]/2):
                x = int(begin[0] + i * SCALE_XY)
                y = begin[1] + i
                border = find_border_from_inner(img_hsv, [x, y], "right")
                print "sub ", x * 2 - begin[0] - border[0]
                if(math.fabs(x * 2 - begin[0] - border[0]) < 4):
                    nextPos = [x, y]
                    scale = (pos_actor[0][0] - nextPos[0]) / float(image_size[0])
                    break;
        print "scale ", scale
        jump(scale)
        if(False):
            cv2.rectangle(img, (int(nextPos[0]), int(nextPos[1])), (int(nextPos[0] + 5), int(nextPos[1] + 5)), 255, 2)
            plt.subplot(122), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.show()
        time.sleep(3)

cv2.waitKey(0)
cv2.destroyAllWindows()


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


# def processCircel():
# (x,y),radius = cv2.minEnclosingCircle(contours[0])
# center = (int(x),int(y))
# radius = int(radius)
# print radius
# img = cv2.circle(img,center,radius,(0,255,0),2)
