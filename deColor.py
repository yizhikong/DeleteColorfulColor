import cv2
import cv2.cv as cv
import facedetect
import re
from sklearn.cluster import KMeans
import numpy as np

# draw rectangle in the image
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# spilt image into hei * wid block
def splitImg(img, wid = 3, hei = 2):
    height, width = img.shape[0], img.shape[1]
    widStep = width / wid
    heiStep = height / hei
    split_info = []
    for j in range(hei):
        for i in range(wid):
            split_info.append([j*heiStep, (j+1)*heiStep, i*widStep, (i+1)*widStep])
    return split_info

def getBlockImg(img, wid = 3, hei = 2):
    split_info = splitImg(img, wid, hei)
    rects = map(lambda x : (x[2], x[0], x[3], x[1]), split_info)
    vis = img.copy()
    idx = 0
    for rect in rects:
        draw_rects(vis, [rect], (128, 0, 0))
        border = min(rect[3] - rect[1], rect[2] - rect[0])
        cv2.putText(vis, str(idx), ((rect[0] + rect[2]) / 2, (rect[3] + rect[1]) / 2), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0 ,0), thickness = 8, lineType = 8) 
        idx += 1
    return vis

# de-color for the entire image
def deColor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    height, width = img.shape[0], img.shape[1]
    min_a, max_a = a.min(), a.max()
    min_b, max_b = b.min(), b.max()
    thread = min_a + 1 * (max_a - min_a) / 3.0
    for h in range(height):
        for w in range(width):
            if a[h][w] > thread and b[h][w] < 150 and l[h][w] > 50:
                img[h][w][1] -= 50 * float(a[h][w] - thread) / (max_a - thread)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img

def excludeDeColor(img, faceColor):
    rgb = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    height, width = img.shape[0], img.shape[1]
    min_a, max_a = a.min(), a.max()
    min_b, max_b = b.min(), b.max()
    thread = min_a + 1 * (max_a - min_a) / 3.0
    hue, s, v = cv2.split(hsv)
    for h in range(height):
        for w in range(width):
            #thread = 1.0 - 0.8 * v[h][w] / 255
            #if s[h][w] < thread:
            #    continue
            if v[h][w] < 25 or s[h][w] < 25:
                continue
            if abs(hue[h][w] - faceColor[0]) < 10:
                continue
            if a[h][w] > thread and b[h][w] < 150 and l[h][w] > 50:
                img[h][w][1] -= 50 * float(a[h][w] - thread) / (max_a - thread)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img

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
    kills, keeps = [], []
    for i in range(n_clusters):
        if centers[i][0] < 10 or centers[i][0] > 170:
            kills.append(i)
        else:
            keeps.append(i)
    print centers
    print kills
    for h in range(height):
        for w in range(width):
            idx = width * h + w
            #img[h][w] = centers[result[idx]]
            if result[idx] in kills:
                dis = centers[result[idx]] - [50, 180, 180]
                img[h][w][0] -= 80
                if img[h][w][0] < 0:
                    img[h][w][0] += 180
                #img[h][w][1] = img[h][w][1] - dis[1]
                #img[h][w][2] = img[h][w][2] - dis[2]
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('img', img)
    cv2.waitKey()
    print result

def process(Full_img, select = 0, wid = 3, hei = 2):
    split_info = splitImg(Full_img, wid, hei)
    hs, he, ws, we = split_info[select]
    img = Full_img[hs:he, ws:we]
    Full_img[hs:he, ws:we] = deColor(img)
    return Full_img

def getFaceColors(faceImg):
    faceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2LAB)
    num = faceImg.shape[0] * faceImg.shape[1]
    r = faceImg[:, :, 0].sum() / num
    g = faceImg[:, :, 1].sum() / num
    b = faceImg[:, :, 2].sum() / num
    return (r, g, b)

def excludeFaceProcess(imgUrl):
    faces = facedetect.microsoft_detect(imgUrl)
    imgName = re.sub('.*?img/', '', imgUrl)
    img = cv2.imread(imgName)
    res_img = deColor(img.copy())
    for face in faces:
        res_img[face[1]:face[3], face[0]:face[2]] = img[face[1]:face[3], face[0]:face[2]]
    return res_img

def excludeSkinProcess(imgUrl):
    faces = facedetect.microsoft_detect(imgUrl)
    imgName = re.sub('.*?img/', '', imgUrl)
    img = cv2.imread(imgName)
    origin_img = img.copy()
    if len(faces) > 0:
        face = faces[0]
        faceColor = getFaceColors(img[face[1]:face[3], face[0]:face[2]])
        img = excludeDeColor(img, faceColor)
    else:
        img = deColor(img)
    d = min(face[3] - face[1], face[2] - face[0]) / 6
    for face in faces:
        h1, h2 = face[1] + d, face[3] - d
        w1, w2 = face[0] + d, face[2] - d
        # img[face[1]:face[3], face[0]:face[2]] = img[face[1]:face[3], face[0]:face[2]]
        img[h1:h2, w1:w2] = origin_img[h1:h2, w1:w2]
    return img

if __name__ == '__main__':
    colorKMeans(cv2.imread("ogafIty9-HIQqCBrc9p3--79A_Pc_1462944445_temp.jpg"))
    imgUrl = 'http://139.129.131.50/img/ogafIty9-HIQqCBrc9p3--79A_Pc_1462944445_temp.jpg'
    #img = excludeSkinProcess(imgUrl)
    #cv2.imshow('img', img)
    #cv2.waitKey()
    '''
    status = 0
    fileName = ''
    while True:
        if status == 0:
            fileName = raw_input()
            img = getBlockImg(cv2.imread(fileName))
            cv2.imshow('img', img)
            cv2.waitKey()
            status = 1
            continue
        if status == 1:
            select = raw_input()
            try:
                select = int(select)
                img = process(cv2.imread(fileName), select)
                cv2.imshow('img', img)
                cv2.waitKey()
            except:
                status = 0
                img = getBlockImg(cv2.imread(select))
                cv2.imshow('img', img)
                cv2.waitKey()
                status = 1
    '''