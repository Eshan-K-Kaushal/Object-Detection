import cv2
import matplotlib.pyplot as plt
from os import listdir
from pathlib import Path
from os.path import isfile, join 
 
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
#read an image
path = Path('D:/Object_Detection/sample_pics')
imagepath = 'D:/bject_Detection/sample_pics'
images = []
for imagepath in path.glob("*.jpg"):
    img = cv2.imread(str(imagepath))
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    images.append(img)
    fig = plt.gcf()
    fig.canvas.set_window_title('Image')
    plt.imshow(img)#BGR
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    #plt.imshow(img, cmap = 'gray')
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    #plt.show()
    plt.show(block = False)
    plt.pause(.7)
    plt.close()

    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.55)
    print(ClassIndex)

    
    font_scale  = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img,boxes,(255,0,0), 2)
        cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale = font_scale, color = (0, 255, 0), thickness= 2)


    plt.imshow(img)
    plt.show(block  = False)
    plt.pause(5)
    plt.close()












