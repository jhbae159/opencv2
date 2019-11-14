#carscade로 손검출
#손 위치 검출
#손 영역의 hsv값
#제스처 인식

import cv2
import numpy as np
import math


# 관심영역에서 z누르면 rgb를 hsv값으로 바꿔줌
def find_hsv_range(frame):
    rows, cols = frame.shape
    center_frame = frame[rows/2,cols/2]

    hsv = cv2.cvtColor(center_frame,cv2.COLOR_BGR2HSV)
    print(hsv)
    return hsv



cap = cv2.VideoCapture(0)


while (1):

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        # define region of interest
        roi = frame[100:300, 100:300]
        #make_frame = roi를 짤라 (x :80~120, y: 90~130)범위를 짜름
        make_frame = roi[90:130,80:120]
        #hhhsv = make_frame을 hsv화 시킴
        hhhsv  = cv2.cvtColor(make_frame,cv2.COLOR_BGR2HSV)
        rows, cols = hhhsv.shape[:2]
        ######### hhhsv[0][0] ~ hhhsv[40][40]의 평균 hsv를 구해보자#############
        ######문제 : hsv값 잡기가 힘들다


        print(hhhsv[20][20])
        cv2.imshow('hsv',hhhsv)

        h= hhhsv[20][20][0]
        s= hhhsv[20][20][1]
        v= hhhsv[20][20][2]

        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
  #      lower_skin = np.array([54, 131, 110], dtype=np.uint8)
   #     upper_skin = np.array([163, 157, 135], dtype=np.uint8)

       # lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        #upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        if v>220:
            max_v = 225
        else:
            max_v = v
        if h<20:
            h=20
            max_h = h+20
            min_h = 0
        if h>100:
            max_h = 180
            min_h = max_h-30
        if s<50:
            s=50
        lower_skin = np.array([min_h, s-30, v-50], dtype=np.uint8)
        upper_skin = np.array([max_h, s+30, max_v+30], dtype=np.uint8)

        # extract skin colur imagw
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        #mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=6)
        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find contour of max area(hand)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        cv2.drawContours(mask,cnt,-1,(255,0,0),5)
        #cnt = max(contours, key=lambda x: cv2.contourArea(x))
        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # make convex hull around hand
        hull = cv2.convexHull(cnt)

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

        # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        # l = no. of defects
        l = 0

        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'THUMBS UP', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                else:
                    cv2.putText(frame, '1(THUMBS UP)', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 3:

            if arearatio < 27:
                cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

