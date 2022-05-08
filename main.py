import cv2
import cv2.dnn
import numpy as np

classesFile = 'coco.names.txt'
cap = cv2.VideoCapture(0)
with open(classesFile, 'rt') as f:
	classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg.txt' #Config file, please update according to the config file you downloaded
modelWeights = 'yolov3.weights'  #Config file, please update according to the config file you downloaded
widthWidthTrget = 320
widthHeightTrget = 320
confTreshold = 0.5
nmsTreshold = 0.3
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
	height, width, channels = img.shape
	bbox = []  # border box
	classIds = []
	confs = []  # confidences
	for output in outputs:
		for detection in output:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confTreshold:
				w, h = int(detection[2] * width), int(detection[3] * height)  # get pixels values
				x, y = int((detection[0] * width) - w / 2), int((detection[1] * height) - h / 2)
				bbox.append([x, y, w, h])
				classIds.append(classId)
				confs.append(float(confidence))
	indices = cv2.dnn.NMSBoxes(bbox, confs, confTreshold, nmsTreshold)  # tell us which indices to keep
	for i in indices:
		box = bbox[i]
		x, y, w, h = box[0], box[1], box[2], box[3]
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
		cv2.putText(img, f"{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,255,0,255,0)


while True:
	success, img = cap.read()
	blob = cv2.dnn.blobFromImage(img, 1 / 255, (widthWidthTrget, widthHeightTrget), [0, 0, 0], 1, crop=False)
	net.setInput(blob)

	layerNames = net.getLayerNames()
	outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
	outputs = net.forward(outputNames)
	findObjects(outputs, img)

	cv2.imshow('Image', img)
	cv2.waitKey(1)
