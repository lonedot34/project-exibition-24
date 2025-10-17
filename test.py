
import warnings
warnings.filterwarnings('ignore')

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import random
import time
from datetime import datetime
import os
import face_recognition
from flask import Flask, render_template, Response, jsonify, send_file


challenges = ["Blink Your Eyes"]
current_challenge = None
challenge_start_time = 0
CHALLENGE_INTERVAL = 7
CHALLENGE_TIMEOUT = 5
liveness_proven = False

# Blink detection parameters
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

#Encoding known faces

path = "photos"
images = []
known_face_names = []
for cl in os.listdir(path):
    if cl.lower().endswith(('.png', '.jpg', '.jpeg')):
        curImg = cv2.imread(os.path.join(path, cl))
        if curImg is not None:
            images.append(curImg)
            known_face_names.append(os.path.splitext(cl)[0])

def findEncodings(images_list):
    encodeList = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print(f"Warning: No face found in one of the known images, skipping.")
    return encodeList

known_face_encodings = findEncodings(images)
print('[INFO] Encoding Complete.')


#for csv file

output_folder_path = './'
today_date = datetime.now().strftime('%Y-%m-%d')
filename = f'Attendance_{today_date}.csv'
daily_attendance_file = os.path.join(output_folder_path, filename)

if not os.path.exists(daily_attendance_file):
    with open(daily_attendance_file, 'w') as f:
        f.writelines('Name,Time')

def markAttendance(name, filename):
    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')





app = Flask(__name__)
cap = None
process_this_frame = True
last_recognized_name = ""
last_face_location = None
status_text = ""
status_color = (0, 0, 0)
camera_on = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def generate_frames():
    global cap, process_this_frame, last_recognized_name, last_face_location, status_text, status_color, liveness_proven, current_challenge, challenge_start_time, COUNTER

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while camera_on:
        success, img = cap.read()
        if not success:
            break

        if process_this_frame:
            last_recognized_name = ""
            last_face_location = None

            if current_challenge is None or time.time() - challenge_start_time > CHALLENGE_INTERVAL:
                current_challenge = random.choice(challenges)
                challenge_start_time = time.time()
                liveness_proven = False

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            
            if len(rects) == 0:
                liveness_proven = False

            for rect in rects:
                landmarks = predictor(gray, rect)
                landmarks_np = np.zeros((68, 2), dtype=np.int32)
                for i in range(0, 68):
                    landmarks_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
                
                leftEye = landmarks_np[lStart:lEnd]
                rightEye = landmarks_np[rStart:rEnd]
                
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                
                ear = (leftEAR + rightEAR) / 2.0
                
                if not liveness_proven and current_challenge == "Blink Your Eyes" and time.time() - challenge_start_time < CHALLENGE_TIMEOUT:
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            liveness_proven = True
                            print("Blink detected! Liveness proven.")
                        COUNTER = 0

            if liveness_proven:
                face_locations = face_recognition.face_locations(img)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(img, face_locations)
                    for encodeFace, faceLoc in zip(face_encodings, face_locations):
                        matches = face_recognition.compare_faces(known_face_encodings, encodeFace)
                        if True in matches:
                            matchIndex = matches.index(True)
                            name = known_face_names[matchIndex].upper()
                            markAttendance(name, daily_attendance_file)
                            
                            last_recognized_name = name
                            last_face_location = faceLoc

        process_this_frame = not process_this_frame

        if liveness_proven:
            status_text = "Liveness Proven"
            status_color = (0, 255, 0)
            if last_recognized_name and last_face_location:
                y1, x2, y2, x1 = last_face_location
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, last_recognized_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        elif time.time() - challenge_start_time > CHALLENGE_TIMEOUT:
            status_text = "Challenge Failed. New challenge shortly."
            status_color = (0, 0, 255)
        else:
            status_text = f"Challenge: {current_challenge}"
            status_color = (0, 255, 255)
            
        cv2.putText(img, status_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if cap:
        cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global camera_on, cap
    camera_on = True
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return jsonify({"success": True, "message": "Camera started."})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global camera_on, cap
    camera_on = False
    if cap:
        cap.release()
        cv2.destroyAllWindows()
    cap = None
    return jsonify({"success": True, "message": "Camera stopped."})

@app.route('/api/attendance')
def get_attendance_data():
    try:
        with open(daily_attendance_file, 'r') as f:
            lines = f.readlines()[1:]
            attendance_list = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    attendance_list.append({"name": parts[0], "time": parts[1]})
            return jsonify(attendance_list)
    except FileNotFoundError:
        return jsonify([])



#download purpose
@app.route('/download_attendance')
def download_attendance():
    try:
        return send_file(daily_attendance_file, as_attachment=True)
    except FileNotFoundError:
        return "Attendance file not found.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)