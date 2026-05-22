from ultralytics import YOLO
import cv2

# load models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO('./mo')