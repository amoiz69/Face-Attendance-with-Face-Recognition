from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Define paths
IMAGES_PATH = 'Images'
os.makedirs(IMAGES_PATH, exist_ok=True)
ATTENDANCE_FILE = 'Attendance.csv'

# Load known images and names
images = []
classNames = []

def load_images():
    global images, classNames
    images = []
    classNames = []
    myList = os.listdir(IMAGES_PATH)
    for cls in myList:
        curImg = cv2.imread(f"{IMAGES_PATH}/{cls}")
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

load_images()

def findEncodings(images):
    """Encode the list of images."""
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Encode known faces
print("Encoding images...")
encodeListKnown = findEncodings(images)
print("Encoding complete.")

attendance_tracker = {}

def markAttendance(name):
    """Mark attendance in a CSV file."""
    now = datetime.now()
    if name in attendance_tracker:
        last_attendance_time = attendance_tracker[name]
        if (now - last_attendance_time) < timedelta(minutes=30):
            return
    attendance_tracker[name] = now
    with open(ATTENDANCE_FILE, 'a') as f:
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{dtString}\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/mark_attendance')
def mark_attendance():
    return render_template('mark_attendance.html')

@app.route('/attendance_prompt/<name>')
def attendance_prompt(name):
    return render_template('attendance_prompt.html', name=name)

# Global variable for camera
camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    cap = get_camera()
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed_register')
def video_feed_register():
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed (register): {e}")
        return "Camera feed error.", 500

@app.route('/capture_register', methods=['POST'])
def capture_register():
    global camera
    success, frame = get_camera().read()
    if not success:
        return "Error capturing image.", 500

    name = request.form.get('name').strip()
    if not name:
        return "Name is required for registration", 400

    # Save the snapshot with the provided name
    image_path = os.path.join(IMAGES_PATH, f"{name}.jpg")
    cv2.imwrite(image_path, frame)

    # Reload images and encodings
    load_images()
    global encodeListKnown
    encodeListKnown = findEncodings(images)

    return render_template('success.html', message=f"Registration successful for {name}!")

def generate_frames_attendance():
    cap = get_camera()
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    start_time = datetime.now()
    attendee_name = None
    while True:
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                markAttendance(name)
                attendee_name = name

                # Draw a bounding box with a welcome message
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Welcome {name}!", (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if (datetime.now() - start_time).seconds > 10:
            break

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

    if attendee_name:
        # Redirect to the attendance prompt page with attendee name
        return redirect(url_for('attendance_prompt', name=attendee_name))

@app.route('/video_feed_attendance')
def video_feed_attendance():
    try:
        return Response(generate_frames_attendance(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed (attendance): {e}")
        return "Camera feed error.", 500

if __name__ == '__main__':
    app.run(debug=True)
