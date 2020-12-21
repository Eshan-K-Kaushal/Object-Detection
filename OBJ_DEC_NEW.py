#auth - Eshan
#changes - 1)See confidence vals 2)Format

import cv2
import matplotlib.pyplot as py

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
cap = cv2.VideoCapture(0)
cap.set(3,1200)
cap.set(4,720)
#img = cv2.imread('sample1.jpg')
#cv2.imshow("Output",img)

#cv2.waitKey(0)

#either write classnames yourself or import

classNames = []
classFile = 'labels.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

mod = cv2.dnn_DetectionModel(frozen_model, config_file)
mod.setInputSize(320,320)
mod.setInputScale(1.0/ 127.5)
mod.setInputMean((127.5, 127.5, 127.5))
mod.setInputSwapRB(True)
while True:
    success, img = cap.read()
    classIds, confs, bbox = mod.detect(img, confThreshold=.54)
    print(classIds, bbox)

    if len(classIds)!=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0,0,255),thickness=3)
            cv2.putText(img, classNames[classId-1], (box[0]+10, box[1]+25),
                        cv2.FONT_HERSHEY_SIMPLEX,fontScale=3,color=(255,255,0), thickness=4)
            cv2.putText(img,str(round(confidence*100)), (box[0]+60, box[1]+100),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=.8, color=(255,0,0),
                        thickness=2)

    cv2.imshow("Video", img)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    #cv2.waitKey(0)

    #fig = py.gcf()
    #fig.canvas.set_window_title("Output")
    #py.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #py.imshow(img)
    #py.show(block = False)
    #py.pause(10)
    #py.show()
    #py.close()
