from threading import Lock
from collections import defaultdict
import numpy as np
import time
# import keyboard listener
from pynput.keyboard import Key, Listener
import cv2

from ultralytics import YOLO

from flask_socketio import SocketIO, emit
from flask import Flask, render_template, Response


async_mode = None
app = Flask(__name__)
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

MODEL_PATH = '/research/model/yolov8n_FP16_simp.engine'
CAMERA_SOURCE = "/research/IMG_4740.MOV"  # use 0 for web camera

NORMAL = 1 # turn off human detection 
TRACKING_OFF = 2 # turn on tracking with id_selected of the box is yellow and the others  is red , enable keyboard listener to select the id_selected
TRACKING_ON = 3 # turn on tracking with id_selected of the box is green and the others hidden, disable keyboard listener RIGHT and LEFT arrow

model = YOLO(MODEL_PATH)
# mode.track a dummy frame to load the model
model.track(np.zeros((640, 480, 3), dtype=np.uint8), persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
camera = cv2.VideoCapture(CAMERA_SOURCE) 

# Variables for tracking
list_of_ids = []
id_selected = -1
list_id_sorted = []
dict_id = {} # key: id, value: boxes 
is_tracking = False
key_pressed = None
fps = 0.0
mode = NORMAL 
# Store the track history
track_history = defaultdict(lambda: [])


def sort_list_of_ids(list_of_ids, dict_id):
    # a list of IDs sorted by distance from left to right along the X axis
    list_id_sorted = []
    for id in list_of_ids:
        x, y, w, h = dict_id[id]
        list_id_sorted.append((id, x))
    list_id_sorted.sort(key=lambda x: x[1])
    return [id for id, x in list_id_sorted]

def find_ID_nearby_center(list_id_sorted, dict_id, width):
    # find the ID of the box closest to the center of the screen
    x_center = width / 2
    distance = 100000
    for id in list_id_sorted:
        x, y, w, h = dict_id[id]
        if abs(x - x_center) < distance:
            distance = abs(x - x_center)
    return id if distance < 100000 else -1

def draw_box(frame, box, track_ids, color=(0, 255, 255)):
    # convert box to x1, y1, x2, y2 with int type
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Draw the bounding box with green line
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 10)
    # Draw the track ID
    cv2.putText(frame, str(track_ids), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)
    return frame

# keyboard listener left , right , up , down and space key 
# press up arrow to turn on object detection and tracking ( TRACKING_OFF )
# press down arrow to turn off object detection and tracking ( NORMAL )
# press left arrow to select the previous box ( TRACKING_OFF )
# press right arrow to select the next box ( TRACKING_OFF )
# press space to switch between TRACKING_ON and TRACKING_OFF ( TRACKING_ON -> TRACKING_OFF or TRACKING_OFF -> TRACKING_ON )
def on_press(key):
    global id_selected, list_id_sorted, dict_id, is_tracking, list_of_ids, key_pressed, mode
    if list_of_ids == [] and mode != NORMAL:
        id_selected = -1 
        return
    try:
        if key == Key.left:
            print('left arrow key pressed')
            if is_tracking == False or id_selected == -1:
                return
            list_id_sorted = sort_list_of_ids(list_of_ids, dict_id)
            index = list_id_sorted.index(id_selected)
            if index - 1 < 0:
                id_selected = list_id_sorted[0]
            else:
                id_selected = list_id_sorted[index - 1]
        elif key == Key.right:
            print('right arrow key pressed')
            if is_tracking == False or id_selected == -1:
                return
            list_id_sorted = sort_list_of_ids(list_of_ids, dict_id)
            index = list_id_sorted.index(id_selected)
            if index + 1 > len(list_id_sorted) - 1:
                id_selected = list_id_sorted[-1]
            else:
                id_selected = list_id_sorted[index + 1]
        elif key == Key.up:
            mode = TRACKING_OFF
            is_tracking = True
            print('up arrow key pressed')
        elif key == Key.down:
            mode = NORMAL
            is_tracking = False
            id_selected = -1
            print('down arrow key pressed')
        elif key == Key.space:
            if mode == TRACKING_OFF:
                mode = TRACKING_ON
            elif mode == TRACKING_ON:
                mode = TRACKING_OFF
            print('space key pressed')
    except AttributeError:
        print('special key pressed {0}'.format(key))

def on_release(key):
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released on a new thread
listener = Listener(on_press=on_press, on_release=on_release)
listener.start()


def predict(frame):
    global list_of_ids, id_selected, list_id_sorted, dict_id, is_tracking, mode
    if mode == NORMAL:
        return frame
    start_time = time.time()
    results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
    end_time = time.time()
    try:
        boxes = results[0].boxes.xywh.cpu()
        boxes_xyxy = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        dict_id = {id: box for id, box in zip(track_ids, boxes)}
    except:
        return results[0].plot(probs=False, labels=False)
    if list_of_ids != track_ids:
        list_of_ids = track_ids
    if id_selected not in list_of_ids and id_selected != -1:
        id_selected = -1
    is_show_boxes = False
    if mode == TRACKING_OFF:  is_show_boxes=True
    annotated_frame = results[0].plot(probs=False, labels=False, boxes=is_show_boxes)
    if is_tracking and id_selected == -1:
        list_id_sorted = sort_list_of_ids(list_of_ids, dict_id)
        id_selected = find_ID_nearby_center(list_id_sorted, dict_id, frame.shape[1])

    # Plot the tracks
    for box, box_xyxy, track_id in zip(boxes, boxes_xyxy, track_ids):
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
        if id_selected == -1:
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        elif int(track_id) != int(id_selected):
            continue
        else:
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            if mode == TRACKING_ON: 
                annotated_frame = draw_box(annotated_frame, box_xyxy, track_id, color=(0, 255, 0))
            else:
                annotated_frame = draw_box(annotated_frame, box_xyxy, track_id)
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
