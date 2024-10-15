import cv2
import numpy as np

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2  # Non-maximum suppression threshold

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)  # Width
# cap.set(4, 720)   # Height
# cap.set(10, 150)  # Brightness

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
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # Perform Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        i = i[0]  # Get index
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Output', img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
