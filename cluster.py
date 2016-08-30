import cv2
import cv2.cv as cv
import facedetect
import re
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import random

def colorKMeans(img):
    img = cv2.resize(img, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    data = []
    height, width = img.shape[0], img.shape[1]
    for h in range(height):
        for w in range(width):
            data.append(img[h][w])
    data = np.array(data)
    print data.shape
    n_clusters = 6
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=1)
    kmeans.fit(data)
    result = kmeans.predict(data)
    centers = kmeans.cluster_centers_
    
    for h in range(height):
        for w in range(width):
            if img[h][w][1] < 100 or img[h][w][2] < 100:
                continue
            idx = width * h + w
            img[h][w] = centers[result[idx]]

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('img', img)
    cv2.waitKey()
    print result

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
    cv2.imshow('foreground', fg)
    cv2.waitKey()
    _, img = cv2.threshold(temperatureImg, 0.8 * threshold, 255, cv2.THRESH_BINARY)
    _, bg = cv2.threshold(cv2.dilate(img, None, iterations = 3), 1, 128, cv2.THRESH_BINARY_INV)
    cv2.imshow('background', bg)
    cv2.waitKey()
    marker = cv2.add(fg, bg)
    cv2.imshow('maker_before', marker)
    cv2.waitKey()
    markers = np.int32(marker)
    cv2.watershed(image, markers)
    m = cv2.convertScaleAbs(markers)
    cv2.imshow('marker', m)
    cv2.waitKey()
    return m

def ValueTemperature(image):
    #image = cv2.resize(image, (300, 300))
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    img = hsvImage.copy()
    height, width = img.shape[0], img.shape[1]
    temperatureRange = 130
    temperatureSum = 0.0
    data = [0] * (temperatureRange + 1)
    channelDict = {'saturation':{'value':0, 'valueMax':255, 'weight':0.5},
    'value':{'value':0, 'valueMax':255, 'weight':0.5}}
    for h in range(height):
        for w in range(width):
            channelDict['saturation']['value'] = img[h][w][1]
            channelDict['value']['value'] = img[h][w][2]
            img[h][w][0] = (1 - countTemperature(channelDict)) * temperatureRange
            data[int(img[h][w][0])] += 1
            temperatureSum += img[h][w][0]
            img[h][w][1] = 240
            img[h][w][2] = 150
    averTemperature = int(temperatureSum / height / width)
    threadsold = OSTU(data[:averTemperature])
    # find major hue in high temperature part
    hueDict = {}
    maxHue, maxHueCount = -1, -1
    for h in range(height):
        for w in range(width):
            if img[h][w][0] < threadsold:
                hue = hsvImage[h][w][0]
                if hue not in hueDict:
                    hueDict[hue] = 1
                else:
                    hueDict[hue] += 1
                if hueDict[hue] > maxHueCount:
                    maxHueCount = hueDict[hue]
                    maxHue = hue

    # get the target pixel by both threadsold and hue
    targetPixel = {}
    hsvSum = np.array([0.0] * 3)
    for h in range(height):
        for w in range(width):
            if img[h][w][0] < threadsold + 5 and hsvImage[h][w][0] in hueDict and hueDict[hsvImage[h][w][0]] > maxHueCount * 0.3:
                targetPixel[(h, w)] = 1
                hsvSum += np.array(hsvImage[h][w])

    hasVisit = targetPixel.copy()
    limit = (height - 1, width - 1)
    neighborAdd = {}
    pixelQueue = targetPixel.keys()
    pos = 0
    while pos < len(pixelQueue):
        neighbor = getNeighbor(pixelQueue[pos], limit)
        for n in neighbor:
            if n in hasVisit:
                continue
            else:
                h, w = n
                hasVisit[n] = 1
                if countHueDistance(int(hsvImage[h][w][0]), maxHue) < 5:
                    neighborAdd[n] = 1
                    pixelQueue += getNeighbor(n, limit)
        hasVisit[pixelQueue[pos]] = 1
        pos += 1

    # count the average hue/saturation/value of the target part
    hsvAver = hsvSum / len(targetPixel)
    # sample should get from another part of the current image
    select_h = random.randint(0, height)
    select_w = random.randint(0, width)
    while (select_h, select_w) in targetPixel:
        print 're-select'
        select_h = random.randint(0, height)
        select_w = random.randint(0, width)
    #hsvSample = np.array([108, 235, 231])
    hsvSample = np.array(hsvImage[select_h][select_w])
    hsvChange = hsvAver - hsvSample
    for target in targetPixel.keys() + neighborAdd.keys():
        h, w = target[0], target[1]
        hh = int(hsvImage[h][w][0]) - hsvChange[0]
        ss = int(hsvImage[h][w][1]) - hsvChange[1]
        vv = int(hsvImage[h][w][2]) - hsvChange[2]
        hsvImage[h][w][0] = max(min(hh, 180), 0)
        hsvImage[h][w][1] = max(min(ss, 255), 0)
        hsvImage[h][w][2] = max(min(vv, 255), 0)

    cv2.imshow('image', cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR))
    #cv2.imshow('image', cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR))
    cv2.waitKey()
    #plt.bar(np.arange(len(data)), data, alpha = 0.5)
    #plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.imwrite('output.jpg', cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR))

def minFilter(input_img, size):
    height, width = input_img.shape
    pW = size[0] / 2
    pH = size[1] / 2
    newWidth = width + pW * 2
    newHeight = height + pH * 2
    # padding
    # 0 will lead to white border
    # img = zeros((newHeight, newWidth), dtype = "double")
    img = np.array([255.0] * newHeight * newWidth).reshape(newHeight, newWidth)
    # center part
    img[pH:(pH + height), pW:(pW + width)] = input_img.copy()
    w = 0
    h = 0
    # min filter
    for domain in view_as_window(img, size):
        # get the minimun number of the domain
        input_img[h][w] = domain.min()
        h += 1
        if h == height:
            h = 0
            w += 1
    return input_img

# size is the size of patch
def getDarkChannel(img, size):
    r, g, b = cv2.split(img)
    height, width = r.shape
    # get the min pixel among r, g and b
    min_rgb = np.zeros((height, width), dtype = "uint8")
    for h in range(height):
        for w in range(width):
            #print (r[h][w], g[h][w], b[h][w])
            min_rgb[h][w] = min(r[h][w], g[h][w], b[h][w])
    # min filter
    img = minFilter(min_rgb, size)
    return img

# the yield version of view_as_window function, lazy 
def view_as_window(img, patch_size):
    height, width = img.shape
    pWidth = patch_size[0]
    pHeight = patch_size[1]
    for i in range(width - pWidth + 1):
        for j in range(height - pHeight + 1):
            # the RGB[i][j] is the left-top pixel of the patch
            domain = img[j:(j+pHeight), i:(i+pWidth)]
            yield domain

if __name__ == '__main__':
    #colorKMeans(cv2.imread("wo.jpg"))
    #colorThreadsold(cv2.imread("pink2.jpg"))
    #colorThreadsold(cv2.imread("pink3.jpg"))
    #colorThreadsold(cv2.imread("pink4.jpg"))
    #colorThreadsold(cv2.imread("yellow.jpg"))
    #cv2.imshow('dark channel', getDarkChannel(cv2.imread("5.jpg"), (15, 15)))
    #cv2.waitKey()
    #ValueTemperature(cv2.imread("5.jpg"))
    Segmentation(cv2.imread("5.jpg"))
    cv2.imshow('temperature', getTemperatureImg(cv2.imread("5.jpg")))
    cv2.waitKey()