# yolov8-human-tracking Web UI

## 1. Pull and run yolo docker: 
```
docker run -it --name yolov8 --privileged --ipc=host --net=host --gpus all -v $(pwd):/research --gpus all ultralytics/ultralytics:8.0.229-jetson
```
## 2. Install necessary packages ( in docker container ) 
```
pip install --no-cache flask==3.0.0 flask_socketio==5.3.6 pynput==1.7.6
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

Config ( in app.py code ):
- `MODEL_PATH` = '/research/model/yolov8n_FP16_simp.engine'
- `CAMERA_SOURCE` = "/research/people.mp4"  

Run: 
```
cd /research
python app.py 
```
Now access the address `http://127.0.0.1:8001/` to see the results. 
