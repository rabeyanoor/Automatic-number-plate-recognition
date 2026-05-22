from ultralytics import YOLO
import cv2

# load models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('')