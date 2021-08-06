from flask import Flask, request, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import Counter
import pandas as pd

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)



def get_thumb_status(wy, ty):
    differential = wy - ty
    if differential <= .1 and differential >= -.1:
        return 0
    elif differential > .1:
        return 1
    elif differential < -.1:
        return -1


def gen_frames():  # generate frame by frame from camera
    
    counter = 0
    thumb_status = 0
    last_status = 0
    symptom_list = []
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

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


        last_status = thumb_status
        thumb_status = get_thumb_status(wrist_y, thumb_y)

        if last_status == thumb_status and thumb_status != 0:
            counter += 1
        else:
            counter = 0

        if counter >= 50:
            counter = 50

        # cv2.putText(img,"Thumb status: " + str(thumb_status), (10,15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
        # cv2.putText(img,"Counter: " + str(counter), (10,30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

        overlay = img.copy()
        center_coordinates = (75, 75)
        radius = 50
        color = (255, 255, 255)
        if thumb_status == 1:
            color = (20, 255, 20, .5)
        elif thumb_status == -1:
            color = (20, 20, 255)
        thickness = 2
        cv2.circle(overlay, center_coordinates, radius+2, (230,230,230), thickness)
        cv2.circle(overlay, center_coordinates, counter, color, -1)
        alpha = 0.4  
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result

questions = ["Is it painful to walk around?", "Do you have neck pain?"]
qIndex = 0
answers = []
phase = True

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', q=questions, i=qIndex, a=answers, p=phase)


@app.route('/form')
def form():
    """Video streaming home page."""
    return render_template('form.html')

@app.route('/response', methods=['POST'])
def response():
    fname = request.form.get("fname")
    note = request.form.get("note")
    return render_template("index.html", name=fname, note=note)

if __name__ == '__main__':
    app.run(debug=True)
