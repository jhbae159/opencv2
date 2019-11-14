import cv2
import numpy as np


def find_thumb(img):  # 检测并标识大拇指手势
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fist_cascade = cv2.CascadeClassifier("thumb.xml")
    fists = fist_cascade.detectMultiScale(gray, 1.1, 15, flags=cv2.CASCADE_SCALE_IMAGE)  # v20_20 1.1 30 #v6 v17_20

    return fists

def draw_thumb(list, img):
    for (x, y, w, h) in list:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) #画方块
    return img

cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    t_list = find_thumb(frame)
    frame = draw_thumb(t_list,frame)
    cv2.imshow('c',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
