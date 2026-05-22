from ultralytics import YOLO
import cv2

# load models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

#read frames
frame_nmr = -1
ret =True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr <10 :
        pass

    # detect vehicles
    detections = coco_model(frame)[0]
    for detection in detections.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id 
    print(detections)

    # loop through detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            # if the detected object is a car (class_id 2 in COCO dataset)
            if class_id == 2:
                # crop the detected car from the frame
                car_image = frame[y1:y2, x1:x2]

                # detect license plate in the cropped car image
                lp_results = license_plate_detector(car_image)

                # loop through detected license plates
                for lp_result in lp_results:
                    for lp_box in lp_result.boxes:
                        lx1, ly1, lx2, ly2 = map(int, lp_box.xyxy[0])
                        lp_confidence = lp_box.conf[0]

                        # draw bounding box around the detected license plate
                        cv2.rectangle(car_image, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

    # display the frame with detections
    cv2.imshow('Frame', frame)

    # exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break