import urllib2
import urllib
import cv2
import json

# draw rectangle in the image
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def face_rect_detect(url):
    api = 'https://apicloud-facerect.p.mashape.com/process-url.json?'
    request = urllib2.Request(api + urllib.urlencode({"url":url}))
    request.add_header('X-Mashape-Key', 'YOURKEY')
    request.add_header('Accept', 'application/json')
    response = urllib2.urlopen(request).read()
    # {"faces":[{"orientation":"frontal","x":638,"y":145,"width":251,"height":251},{"orientation":"profile-left","x":700,"y":746,"width":88,"height":150}],"image":{"width":1280,"height":960}}
    rects = []
    for face in json.loads(response)["faces"]:
        cvx, cvy = face["x"], face["y"]
        rect = (cvx, cvy, cvx + face["width"], cvy + face["height"])
        rects.append(rect)
    return rects

def microsoft_detect(imgUrl):
    api = 'https://api.projectoxford.ai/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=false'
    data = json.dumps({'url':imgUrl})
    request = urllib2.Request(api, data)
    request.add_header('Content-Type', 'application/json')
    request.add_header('Ocp-Apim-Subscription-Key', 'YOURKEY')
    response = urllib2.urlopen(request).read()
    # [{"faceId":"f1460bc3-174a-4315-baf2-0da8615ce94d","faceRectangle":{"top":172,"left":634,"width":257,"height":257}}]
    print response
    rects = []
    for face in json.loads(response):
        faceInfo = face["faceRectangle"]
        cvx = faceInfo["left"]
        cvy = faceInfo["top"]
        rect = (cvx, cvy, cvx + faceInfo["width"], cvy + faceInfo["height"])
        rects.append(rect)
    return rects

def microsoft_detect2(binary_img):
    api = 'https://api.projectoxford.ai/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=false'
    request = urllib2.Request(api, binary_img)
    request.add_header('Content-Type', 'application/octet-stream')
    request.add_header('Ocp-Apim-Subscription-Key', 'YOURKEY')
    response = urllib2.urlopen(request).read()
    # [{"faceId":"f1460bc3-174a-4315-baf2-0da8615ce94d","faceRectangle":{"top":172,"left":634,"width":257,"height":257}}]
    print response
    rects = []
    for face in json.loads(response):
        faceInfo = face["faceRectangle"]
        cvx = faceInfo["left"]
        cvy = faceInfo["top"]
        rect = (cvx, cvy, cvx + faceInfo["width"], cvy + faceInfo["height"])
        rects.append(rect)
    return rects

def api_test(func, imgUrl):
    imgName = "temp.jpg"
    rects = func(imgUrl)
    img = urllib2.urlopen(imgUrl).read()
    f = open(imgName, "wb")
    f.write(img)
    f.close()
    img = cv2.imread(imgName)
    draw_rects(img, rects, (255, 0, 0))
    cv2.imshow('img', img)
    cv2.waitKey()

if __name__ == '__main__':
    imgUrl = 'http://139.129.131.50/img/ogafIty9-HIQqCBrc9p3--79A_Pc_1462944445_temp.jpg'
    api_test(microsoft_detect, imgUrl)