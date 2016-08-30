import cv2
import cv2.cv as cv
import facedetect
import re
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import random

def OSTU(histogram):
    globalSum = sum(histogram)
    globalAver = float(globalSum) / len(histogram)
    count  = len(histogram)
    maxPos, maxValue = -1, -1
    for i in range(1, count - 1):
        backgroundAver = float(sum(histogram[:i])) / i
        foregroundAver = float(sum(histogram[i:])) / (count - i)
        value = i * (backgroundAver - globalAver) ** 2 + (count - i) * (foregroundAver - globalAver) ** 2
        if value > maxValue:
            maxValue = value
            maxPos = i
    return maxPos

def getNeighbor(position, limit):
    h, w = position
    max_height, max_width = limit
    neighbor = []
    if h > 0:
        neighbor.append((h - 1, w))
        if w < max_width:
            neighbor.append((h - 1, w + 1))
    if w > 0:
        neighbor.append((h, w - 1))
        if h < max_height:
            neighbor.append((h + 1, w - 1))
    if h < max_height:
        neighbor.append((h + 1, w))
        if w > 0:
            neighbor.append((h + 1, w - 1))
    if w < max_width:
        neighbor.append((h, w + 1))
        if h > 0:
            neighbor.append((h - 1, w + 1))
    return neighbor

def countTemperature(channelDict):
    valueSum, channelSum = 0.0, 0.0
    for channel in channelDict:
        valueSum += channelDict[channel]['value'] * channelDict[channel]['weight']
        channelSum += channelDict[channel]['valueMax'] * channelDict[channel]['weight']
    return valueSum / channelSum

def countHueDistance(hue1, hue2):
    if hue2 < hue1:
        hue1, hue2 = hue2, hue1
    return min(hue2 - hue1, 180 + hue1 - hue2)

def getTemperatureImg(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = img.shape[0], img.shape[1]
    temperatureImg = np.zeros((height, width), dtype='uint8')
    channelDict = {'saturation':{'value':0, 'valueMax':255, 'weight':0.1},
    'value':{'value':0, 'valueMax':255, 'weight':1}}
    for h in range(height):
        for w in range(width):
            channelDict['saturation']['value'] = img[h][w][1]
            channelDict['value']['value'] = img[h][w][2]
            temperatureImg[h][w] = countTemperature(channelDict) * 255
    return temperatureImg

def Segmentation(image):
    height, width = image.shape[0], image.shape[1]
    temperatureImg = getTemperatureImg(image)
    averTemperature = temperatureImg.sum() / (height * width)
    data = cv2.calcHist([temperatureImg], [0], None, [256], [0, 255])
    threshold = OSTU(data)
    _, img = cv2.threshold(temperatureImg, 1.2 * threshold, 255, cv2.THRESH_BINARY)
    fg = cv2.erode(img, None, iterations = 5)
    _, img = cv2.threshold(temperatureImg, 0.8 * threshold, 255, cv2.THRESH_BINARY)
    _, bg = cv2.threshold(cv2.dilate(img, None, iterations = 3), 1, 128, cv2.THRESH_BINARY_INV)
    marker = cv2.add(fg, bg)
    markers = np.int32(marker)
    # in markers 255 is colorfulcolor, 128 is other, -1 is boundary
    tpImg = np.rollaxis(np.array([temperatureImg] * 3), 0, 3)
    cv2.watershed(np.array(tpImg, dtype='uint8'), markers)
    return (markers, temperatureImg)

def TransformInLab(image, marker, temperatureImg):
    marker = np.array(marker, dtype='uint8')
    marker = cv2.dilate(marker, None, iterations = 1)
    labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    height, width = image.shape[0], image.shape[1]
    l, a, b = cv2.split(labImage)
    targetL, targetA, targetB = l[marker == 255], a[marker == 255], b[marker == 255]
    averL = targetL.sum() / len(targetL)
    globalAverL = (l.sum() - targetL.sum()) / (height * width - len(targetL))
    DIF = max(averL - globalAverL, 0)
    print (averL, globalAverL)
    min_l, max_l = targetL.min(), targetL.max()
    min_a, max_a = targetA.min(), targetA.max()
    min_b, max_b = targetB.min(), targetB.max()
    threadL = min_l + 1 * (max_l - min_l) / 3.0
    threadA = min_a + 1 * (max_a - min_a) / 3.0
    threadB = min_b + 1 * (max_b - min_b) / 3.0
    for h in range(height):
        for w in range(width):
            if marker[h][w] != 255:
                continue
            if l[h][w] > threadL:
                labImage[h][w][0] -= DIF * float(l[h][w] - threadL) / (max_l - threadL)
            if threadA > 128 and a[h][w] > threadA and l[h][w] > 50:
                #labImage[h][w][1] -= DIF * float(a[h][w] - threadA) / (max_a - threadA)
                labImage[h][w][1] = 128
            if threadB > 128 and b[h][w] > threadB and l[h][w] > 50:
                #labImage[h][w][2] -= DIF * float(b[h][w] - threadB) / (max_b - threadB)
                labImage[h][w][2] = 108

    cv2.imshow('image', cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR))
    #cv2.imshow('image', cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR))
    cv2.waitKey()

if __name__ == '__main__':
    img = cv2.imread("5.jpg")
    img = cv2.resize(img, (img.shape[1]/3, img.shape[0]/3))
    marker, temperatureImg = Segmentation(img)
    TransformInLab(img, marker, temperatureImg)
