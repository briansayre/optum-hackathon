from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time
from app.py import qIndex, answers, phase

app = Flask(__name__)
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

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

    thumb_status = "unknown"
    if wrist_y < thumb_y:
        thumb_status = "Thumb down"
    elif wrist_y > thumb_y:
        thumb_status = "Thumb up"

    cv2.putText(img,thumb_status, (10,140), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    # cv2.putText(img,str(wrist_y), (10,210), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    # cv2.putText(img,str(thumb_y), (10,280), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
