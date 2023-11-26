# yolov8-human-tracking Web UI

## 1. Pull and run yolo docker: 
```
docker run -it --name yolov8 --privileged --ipc=host --net=host --gpus all -v $(pwd):/research --gpus all ultralytics/ultralytics:latest
```
## 2. Install necessary packages ( in docker container ) 
```
RUN pip install --no-cache flask==3.0.0 flask_socketio==5.3.6 pynput==1.7.6
```
## 3. Convert pytorch model to tensorRT 
```
yolo export model=/research/model/yolov8n.pt format=engine half=True simplify=True
```
Note: 
-  `half` : FP16 quantization
-  `simplify`: simplify model
-  `format=engine`: TensorRT

## 4. Run app.py 

Config: 
- `MODEL_PATH` = '/research/model/yolov8n.engine'
- `CAMERA_SOURCE` = "/research/IMG_4740.MOV"  # use 0 for web camera

Run: 
```
python app.py 
```
Now access the address `http://127.0.0.1:8001/` to see the results. 
