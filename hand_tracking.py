from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time
from collections import Counter
import pandas as pd
from app.py import qIndex, answers, phase

symptom_list = []

app = Flask(__name__)
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

def findCommon(list1, list2):
    return list(set(list1).intersection(list2))


def get_thumb_status(wy, ty):
    differential = wy - ty
    if differential <= .1 and differential >= -.1:
        return 0
    elif differential > .1:
        return 1
    elif differential < -.1:
        return -1

df  = pd.read_csv('data/dataset.csv')
empty = []

for i in range(1, 17): 
    empty += df['Symptom_' + str(i)].tolist()

empty = {k: v for k, v in sorted(Counter(empty).items(), key=lambda item: item[1], reverse=True)}
del empty[list(empty.keys())[0]]
# [print(k,':',v) for k, v in empty.items()]

for key in empty:
    print(key)

counter = 0
thumb_status = 0
last_status = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)


    wrist_y = 0
    thumb_y = 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)

                if id == 0:
                    wrist_y = lm.y
                if id == 4:
                    thumb_y = lm.y

                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)

                cv2.circle(img, (cx,cy), 7, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    last_status = thumb_status
    thumb_status = get_thumb_status(wrist_y, thumb_y)

    if last_status == thumb_status and thumb_status != 0:
        counter += 1
    else:
        counter = 0

    cv2.putText(img,str(thumb_status), (10,140), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    # cv2.putText(img,str(wrist_y), (10,210), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    # cv2.putText(img,str(thumb_y), (10,280), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.putText(img,str(counter), (10,280), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

