import cv2 # 영상/이미지를 처리/제어하는 함수가 포함된 라이브러리입니다.
import numpy as np # 수학/과학 관련 함수들이 포함되어있는 라이브러리입니다.
from djitellopy import tello # Tello 드론을 제어하기 위한 함수가 포함된 라이브러리입니다.

me = tello.Tello()  # tello 드론을 사용하기 위해, 변수 me를 정의하여 tello 제어에 접근하고 사용하도록 설정합니다.
me.connect() # 접속(연결) 함수를 통해 tello와 통신 설정을 합니다.
print(me.get_battery())#print문을 통해 현재 드론의 배터리 값을 출력하여 확인할 수 있습니다.
me.streamon() # 드론이 카메라를 통해 수집하는 영상데이터를 확인/사용하도록 stream 모드 시작합니다.
me.takeoff() # 드론을 이륙하도록 함수 takeoff를 선언합니다

#cap = cv2.VideoCapture(0)

#HSV란, 색을 표현하는 방법 중 하나로 색상(H), 채도(S), 명도(V)의 차이를 통해 색상값을 표현하는 방식입니다. 
# 해당 방식은 자연적으로 보이는 색상을 거의 온전히 표현할 수 있다는 장점이 있으나 색상구조가 곡면기하임에 따라 RGB에 비해 어렵게 느껴진다는 단점이 있다.
# 아래에서 표현되는 Threshold란 색상을 구분할 때에 어느정도 값을 기준으로 값의 옳고 그름을 판별할 지 정하는 값입니다. 
#실제로 코드를 통해서 이미지를 처리하기위해 값을 출력해보면 각각의 픽셀에 대한 데이터값으로 이미지가 처리되는 것을 볼수있습니다.


hsvVals = [71, 51, 2, 140, 115, 255] #huemin, satmin , valmin , huemax , satmax , valmax 으로 인식하고자하는 색 범위를 지정합니다.
sensors = 3 # 3개의 센서를 이용할 예정이므로 sensors 변수를 3으로 선언합니다. 
threshold = 0.2 # threshold 값은 0.2로 선언합니다.
width, height = 480, 360 # 제어하고자하는 이미지의 해상도(pixel 크기) 너비와 높이 값을 각각 정의합니다.
senstivity = 3
weights = [-25, -15, 0, 15, 25]
curve = 0
fSpeed = 0

def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def getContours(imgThres, img):
    cx = 0
    cy = 0
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2
        cv2.drawContours(img, contours, -1, (255, 0, 255), 7)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return cx, cy


def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors) #수평 축으로 배열을 분할하는 함수를 통해 받아들인 이미지 값 배열을 각각 처리하기 용이하도록 분할합니다. 
    totalPixels = (img.shape[1] // sensors) * img.shape[0]
    senOut = []
    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)

        cv2.imshow(str(x), im)

    print(senOut)
    return senOut


def sendCommands(senOut, cx, cy):
    global curve
    # TRANSLATION
    lr =( cx - width // 2 ) // senstivity
    fb = (cy - height // 2) // senstivity

    #print("lr:",lr)

    lr = int(np.clip(lr, -20, 20))
    fb = int(np.clip(fb, -20, 20))

    #print("np.clip lr:",lr)



    if lr < 2 and lr > -2: lr = 0
    if fb < 2 and fb > -2: fb = 0

    # ROTATION
    if senOut == [1, 0, 0]:
        curve = weights[0] #-25
    elif senOut == [1, 1, 0]:
        curve = weights[1] #-15
    elif senOut == [0, 1, 0]:
        curve = weights[2] #0
    elif senOut == [0, 1, 1]:
        curve = weights[3] #15
    elif senOut == [0, 0, 1]:
        curve = weights[4] #25

    elif senOut == [0, 0, 0]:
        curve = weights[2] #0
    elif senOut == [1, 1, 1]:
        curve = weights[2] #0
    elif senOut == [1, 0, 1]:
        curve = weights[2] #0

    #me.send_rc_control(lr, fSpeed, 0, curve)
    me.send_rc_control(lr, fb, 0, 0)

    print("lr:",lr,"fspeed:",fSpeed,"curve:",curve)

while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)

    imgThres = thresholding(img)
    #cv2.imshow("imgThres", imgThres)

    cx, cy = getContours(imgThres, img)
    senOut = getSensorOutput(imgThres, sensors)
    sendCommands(senOut, cx, cy)
    #v2.imshow("output", imgThres)
    cv2.imshow("output", img)
    cv2.waitKey(1)
