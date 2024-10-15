import cv2
import numpy as np

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2  # Non-maximum suppression threshold

cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Load class names from coco.names file
classNames = []
classFile = 'coco.names'
with open(classFile, 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load configuration and weights for the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Video capture loop
while True:
    success, img = cap.read()

    # Check if frame was read successfully
    if not success:
        print("Error: Failed to read frame from camera.")
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # Convert bbox and confs to list format
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Flatten classIds for correct indexing
    classIds = classIds.flatten()

    # Perform Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Use classIds[i] directly instead of classIds[i][0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classIds[i] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Output', img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
