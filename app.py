'''
8 
use 7 , keep camera feed only to do test deployment

WORKING CODE - ONLY FOR TOPS 

BOTTOM OVERLAY - FAILED

CAMERA DISSAPPERING DUE TO BOTTOM IMAGE UPLOAD ISSUE - UNSOLVED
CORRECT POSITION OF CLOTH - OK

DO NOT CLICK UPLOAD BUTTTON FOR BOTTOMS --> CAMERA WILL BE BLACK
BOTOM OVERLAY CODE IS INCOMPLETE

'''



from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import cvzone

app = Flask(__name__)


apply_status_tops = False
apply_status_bottoms = False

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    mp_pose1 = mp.solutions.pose
    pose1 = mp_pose1.Pose()
    
    global apply_status_tops

    
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1) # get mirror view

                     
        if not success:
            break
        else: 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')






