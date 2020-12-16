import cv2
import matplotlib.pyplot as plt
 
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
#need labels
classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as ft:
    classLabels = ft.read().rstrip('\n').split('\n')
    #classLabels.append(ft.read()) could be done

#print(classLabels)
model.setInputSize(320,320) #since it was pre-described in the cofig file
model.setInputScale(1.0/127.5) #255/2
model.setInputMean((127.5, 127.5, 127.5))# mobilenet takes input as [-1,1]
model.setInputSwapRB(True)#automatic swap from BGR to RGB
#print(len(classLabels))
cap = cv2.VideoCapture("<enter name>")

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.56)

    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame, boxes, (255,0,0),2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color = (0, 255, 0), thickness = 1)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()










