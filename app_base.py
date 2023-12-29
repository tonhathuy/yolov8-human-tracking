from threading import Lock
from collections import defaultdict
import numpy as np
import time
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, Response, session

import cv2
from ultralytics import YOLO

MODEL_PATH = '/research/model/yolov8n_FP16.engine'
CAMERA_SOURCE = "/research/people.mp4"  # use 0 for web camera

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None
app = Flask(__name__)
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

model = YOLO(MODEL_PATH)
# mode.track a dummy frame to load the model
model.track(np.zeros((640, 480, 3), dtype=np.uint8))
camera = cv2.VideoCapture(CAMERA_SOURCE)  # use 0 for web camera
fps = 0.0
#  rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'


list_of_ids = [1, 2]
id_selected = 'all'
# Store the track history
track_history = defaultdict(lambda: [])

def predict(frame):
    global list_of_ids
    start_time = time.time()
    results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
    end_time = time.time()
    # print(results[0])
    try:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
    except:
        return results[0].plot(probs=False, labels=False)
    if list_of_ids != track_ids:
        list_of_ids = track_ids
    annotated_frame = results[0].plot(probs=False, labels=False)
    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 90:  # retain 90 tracks for 90 frames
            track.pop(0)
        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        # show fps 
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, str(int(fps)) + ' FPS' , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
        # print('>>>>>>>>>>>>>>', track_id, id_selected, '>>>>>>>>>>>>>>')
        if id_selected == 'all':
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        elif int(track_id) != int(id_selected):
            continue
        else:
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    return annotated_frame

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame 
        success, frame = camera.read()  # read the camera frame
        frame = predict(frame)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# update value when list_of_ids changes
def background_thread_update_list_of_ids():
    global list_of_ids
    while True:
        socketio.sleep(1)
        print(list_of_ids)
        socketio.emit('my_response',
                      {'data': list_of_ids},
                      )
@socketio.on('tracking')
def handle_message(message):
    global id_selected
    id_selected = message['data']
    # print('>>>>>>>>>>> received message: ' + str(message['data']))
    emit('test_response', {'data': 'OK', 'id_selected': id_selected})


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread_update_list_of_ids)
    emit('my_response', {'status': 'Connected', 'data' : [1,2,3,4,5]})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8001)
