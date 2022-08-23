import cv2

img = cv2.VideoCapture(0)

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath  = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean(127.5, 127.5, 127.5)
net.setInputSwapRB(True)
while (True):
    s, frame = img.read()
    cv2.imshow("out",frame)

    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    print(classIds, bbox)



img.release()
cv2.destroyAllWindows()