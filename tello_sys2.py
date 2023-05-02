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
senstivity = 3 # 민감도를 설정하기 위해 변수를 선언하고 값은 3으로 설정합니다.
weights = [-25, -15, 0, 15, 25] #이미지를 처리하기 위한 가중치 값을 담은 배열을 선언합니다.
curve = 0 # 색상 값에 따라 회전 시의 회전 정도(커브 값)에 대한 변수를 선언합니다
fSpeed = 0 #드론의 속도 변수를 선언합니다.

#아래의 함수는 색상값을 변별(구별)하기 위한 절차를 담당합니다.
#cvtColor 함수는 현재 받아오는 색상값형태를 다른 형태로 바꿀 때 사용하는 함수로 BGR,hsv,gray,yCrab간의 변환이 가능하다.
#inRange함수의 경우 색상필터를 생성하여 특정 색상값만 표현한 이미지를 생성하는 함수로 (이미지, 최소값, 최대값)의 변수를 기입한 경우 해당 이미지에서 최소값과 최대값 영역내의 값으로 필터를 생성하여 해당 필터를 통과한 이미지를 반환합니다..

def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # RGB로 받아온 값을 HSV로 변환합니다
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]]) #위의 정의한 색상 값 중 min값들을 배열 lower에 정의합니다.
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]]) # 위의 정의한 색상 값 중에 max값들을 배열 upper에 정의합니다.
    mask = cv2.inRange(hsv, lower, upper)# inRange함수를 통해 사용하고자 하는 색상 영역의 값만 처리한 이미지를 mask로 정의합니다.
    return mask #위의 과정으로 생성한 이미지를 반환합니다.

#아래의 함수는 위 Thresholding 함수를 통해 색상 필터링한 이미지에서 이미지의 특징을 추출하기 위해 외곽선을 검출하는 함수입니다.을 표시합니다.. 이때 별도의 선언없이 기입한 이미지에 자동적으로 내용이 적용됩니다.
# findContours 함수는 (이미지,검출모드,외곽선 근사화 방법,...) 변수를 기입한 경우 검출한 외곽선의 좌표와 외곽선 계층을 반환합니다. 이때 외곽선의 계층이란 다중객체(ex.원 안 사각형)가 있는 경우 가장 외부의 도형을 부모,내부의 도형을 자식과 같이 분류하여 순위를 숫자로 매겨 표현합니다.
#drawContours 함수는 (이미지,좌표,외곽선인덱스,색상,굵기,선 타입) 변수를 기입한 경우 해당 좌표에 지정한 방식에 따라 외곽선을 표시합니다.. 이때 별도의 선언없이 기입한 이미지에 자동적으로 내용이 적용되며 외곽선인덱스의 경우 -1인 경우 모든 외곽선 표시를, 두께의 경우 <0 인 경우 내부 채우기를 의미합니다.
#boundingRect 함수는 도형을 변수로 기입하였을 때 해당 도형에 외접하는 사각형의 좌표값을 반환합니다. 사각형을 그리고자하는 경우 cv2.rectangle함수를 이용하여야합니다.

def getContours(imgThres, img):
    cx = 0
    cy = 0
    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #함수를 통해 필요한 영역 부분에 대한 외곽선의 좌표와 인덱스 값을 반환받습니다.
    if len(contours) != 0: #외곽선을 표시할 영역이 존재하는 경우
        biggest = max(contours, key=cv2.contourArea)# 가장 외부의 도형을 biggest 변수로 선언하여 값을 저장합니다.
        x, y, w, h = cv2.boundingRect(biggest) #함수를 통해 좌표값을 반환받습니다. 이때 xy는 사각형의 좌상단의 좌표값이며 w,h는 너비와 높이를 의미합니다.
        cx = x + w // 2 # 원의 중심좌표의 x값은 x에서 너비의 반만큼 이동한 값이므로 x＋w/2로 선업합니다
        cy = y + h // 2 #원의 중심좌표의 y값은 y에서 높이의 반만큼 이동한 값이므로 y＋h/2값으로 선언합니다.
        cv2.drawContours(img, contours, -1, (255, 0, 255), 7)#함수를 사용하여 검출하고자하는 내용에 적합한 외곽선을 그립니다. (*)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED) #계산한 cx,cy값을 적용하여 원을 그립니다. (*)
    return cx, cy #계산한 cx,cy값을 반환합니다.

#아래의 내용은 센서를 통해 받아들이는 값을 처리하는 함수입니다.
#shape() 함수는 일반적으로 배열의 형태를 알아보는 함수이지만 image에 대해서 shape()를 사용하는 경우 세로(height),가로(width),채널 값을 배열로 반환합니다.
#enumerate 함수는 각 값에 대해서 자동적으로 번호를 붙이는 함수입니다. 예를 들어 for 내용 in enumerate([a,b,c]) 의 경우 내용을 출력하면 (1,a) (2,b) (3,c)와 같이 출력됩니다.
# 또한 파이썬의 for 문의 경우 다중 내용에 대하여 다중 변수로 선언한 경우 자동적으로 분할하여 변수 값에 저장하고 호출할 수 있습니다. 
# countNonZero 함수는 이미지에서 값이 존재하는 영역의 갯수(픽셀)를 반환합니다. 일반적으로 이미지는 [0,0,0,25,48,26,22,8,0,0,0,....] 와 같이 값을 가진 행렬의 형태로 표현되며 이는 각 픽셀이 가진 고유한 색상값을 의미합니다. 따라서 countNonZero 값은 이러한 행렬 값에서 0을 제외한 숫자가 존재하는 갯수를 의미하며 이를 반환합니다.
def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors) #수평 축으로 배열을 분할하는 함수를 통해 받아들인 이미지를 각각 센서영역에 맞도록 3등분 분할합니다. 
    totalPixels = (img.shape[1] // sensors) * img.shape[0] # 처리할 픽셀을 계산하기위해 너비와 높이 값을 이용하여 각 센서 영역 별 픽셀 수를 구합니다.(너비/3 * 높이)
    senOut = []
    for x, im in enumerate(imgs): #enumerate 함수를 통해 이미지에 대한 값을 붙여 접근 및 호출이 쉽도록 정의합니다.
        pixelCount = cv2.countNonZero(im) # 이미지가 존재하는 픽셀의 갯수를 pixelCount로 정의합니다. (np.count_nonzero 이용하길 추천)
        if pixelCount > threshold * totalPixels: #센서를 통해 받아온 이미지가 전체 영역 중 유효 영역이 20프로(0.2) 초과인 경우 
            senOut.append(1) #센서 output 값에 1을 삽입(추가)합니다.
        else: # 만약 유효한 영역이 20프로 이하인 경우 
            senOut.append(0) #센서 output 값에 0을 삽입(추가)합니다.

        cv2.imshow(str(x), im) # 각각의 센서로 받아온 이미지를 각각의 숫자 값을 기준으로 화면에 띄웁니다. (**)

    print(senOut) #현재 인식되고있는 센서 output 배열 값을 출력합니다.
    return senOut

#드론을 제어하기위한 함수입니다.
#np.clip() 함수는 (배열,최소값, 최대값)을 기입하였을 때, 배열 내의 최소값보다 작은 값은 최소값으로, 최대값보다 큰 값은 최대값으로 변환하는 함수입니다. Ex)np.clip([1,2,3,4,5],2,4) -> [2,2,3,4,4]
def sendCommands(senOut, cx, cy):
    global curve
    # TRANSLATION
    lr =( cx - width // 2 ) // senstivity  #좌우 이동의 경우 기존 boundingRect에서 얻은 x값을 민감도로 나누어 속도 값을 정의합니다.
    fb = (cy - height // 2) // senstivity # 전후 이동의 경우 기존 boundingRect에서 얻은 y값을 민감도로 나누어 속도 값을 정의합니다

    #print("lr:",lr)

    lr = int(np.clip(lr, -20, 20)) # np.clip을 통해 좌우이동 속도의 값은 절댓값 20을 넘지않도록 지정합니다.
    fb = int(np.clip(fb, -20, 20)) # np.clip을 통해 전후이동 속도의 값은 절댓값 20을 넘지않도록 지정합니다.

    #print("np.clip lr:",lr)



    if lr < 2 and lr > -2: lr = 0 #만약 좌우 이동 값이 2> lr > -2인 경우 좌우 이동을 멈추도록 합니다. (절댓값 2 이내의 영역은 오차범위 및 선의 굵기에 의해 해당 영역에 도입되었다고 판단)
    if fb < 2 and fb > -2: fb = 0 #만약 전후 이동 값이 2> fb > -2인 경우 전후 이동을 멈추도록 합니다. 

    # ROTATION
    if senOut == [1, 0, 0]: #만일 첫번째 센서에서만 색상(객체 값)이 검출된 경우
        curve = weights[0]  # 회전 값 curve를 weight[0] 값인 25로 지정합니다..
    elif senOut == [1, 1, 0]: #만일 첫번째,두번째 센서에서 색상(객체 값)이 검출된 경우
        curve = weights[1]  # 회전 값 curve를 weight[1] 값인 -15로 지정합니다..
    elif senOut == [0, 1, 0]:#만일 두번째 센서에서만 색상(객체 값)이 검출된 경우
        curve = weights[2] # 회전 값 curve를 weight[2] 값인 0으로 지정합니다..
    elif senOut == [0, 1, 1]:#만일 두번째,세번째 센서에서 색상(객체 값)이 검출된 경우
        curve = weights[3] #회전 값 curve를 weight[3] 값인 15로 지정합니다..
    elif senOut == [0, 0, 1]: #만일 세번째 센서에서만 색상(객체 값)이 검출된 경우
        curve = weights[4] #회전 값 curve를 weight[4] 값인 25로 지정합니다..

    #만일 모든 영역에서 또는 아무 영역에서도 값이 검출되지않은 경우 회전 값 curve를 0으로 지정합니다.
    elif senOut == [0, 0, 0]:
        curve = weights[2] #0
    elif senOut == [1, 1, 1]:
        curve = weights[2] #0
    elif senOut == [1, 0, 1]: # 센서 인식 값이 양 극단에서만 검출되는 경우에도 동일하게 회전 값 curve를 0으로 지정합니다
        curve = weights[2] #0

    #me.send_rc_control(lr, fSpeed, 0, curve)
    me.send_rc_control(lr, fb, 0, 0) # 드론을 위에 인식 영역에 따라 정의한 값으로 조종하도록 send_rc_control 함수에 변수를 기입합니다.

    print("lr:",lr,"fspeed:",fSpeed,"curve:",curve)

#cv2.flip() 함수는 (이미지,상수)를 통해 이미지를 좌우,상하반전하는 함수입니다. 이때, 0은 상하반전을, 1은 좌우반전을 의미합니다.

while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame #드론을 통해 받아오는 이미지를 사용할 수 있도록 img 변수를 정의합니다.
    img = cv2.resize(img, (width, height))#현재 처리할 크기에 맞게 이미지 사이즈를 변환합니다.
    img = cv2.flip(img, 0) #받아들이는 이미지를 상하반전하도록 0을 넣습니다.

    imgThres = thresholding(img) # thresholding 함수를 사용하여 이미지에서 필요한 색상부분을 검출한 이미지를 imgThres 변수로 정의합니다.
    #cv2.imshow("imgThres", imgThres)

    cx, cy = getContours(imgThres, img) # getContours 함수를 사용하여 검출영역 윤곽선과 좌표 값을 반환받아 cx,cy에 저장합니다.
    senOut = getSensorOutput(imgThres, sensors) # getSensorOutput 함수를 사용하여 센서를 통한 처리 값을 반환받습니다. 

    sendCommands(senOut, cx, cy) # 위의 함수를 통해 처리한 변수들을 함수 sendCommands를 통해 드론에 적용하여 색상을 따라 주행하도록 합니다.
    #v2.imshow("output", imgThres)
    cv2.imshow("output", img) #현재 드론을 통해 입력되고있는 이미지를 화면에 출력합니다.
    cv2.waitKey(1) #이때 이미지 재생 간격은 1초단위로 변환되도록 설정합니다.
