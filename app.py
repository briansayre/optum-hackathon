from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np

app = Flask(__name__)

camera = cv.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            hsvim = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            lower = np.array([0, 48, 80], dtype = "uint8")
            upper = np.array([20, 255, 255], dtype = "uint8")
            skinRegionHSV = cv.inRange(hsvim, lower, upper)
            blurred = cv.blur(skinRegionHSV, (2,2))
            ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
            cv.imshow("thresh", thresh)
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)