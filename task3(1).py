from ultralytics import YOLO
import numpy 
model = YOLO("yolov8n.pt","vs")
detection_output = model.predict(source="test/PAUV/testvid.mp4",conf=0.25,save=True)