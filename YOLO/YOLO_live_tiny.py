# python YOLO_live_tiny.py -y yolov3-tiny

import numpy as np
import argparse
from imutils.video import VideoStream, FPS
import imutils
import time
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.3, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


def get_model():
	net = cv2.dnn.readNetFromDarknet(os.path.sep.join([args["yolo"], "yolov3-tiny.cfg"]), 
												os.path.sep.join([args["yolo"], "yolov3-tiny.weights"]))
	labels = open(os.path.sep.join([args["yolo"], "coco.names"])).read().strip().split("\n")

	getLayer = net.getLayerNames()
	out_layer_names = [getLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	return labels, net, out_layer_names


def get_color(labels):
	np.random.seed(1)
	colors = np.random.randint(0, 255, size=(len(labels),3), dtype="uint8")

	return colors


def init_video():
	video_stream = VideoStream(src=0).start()
	time.sleep(2.0)
	fps_record = FPS().start()

	return video_stream, fps_record

#---------------------------------------------------------------------------------#

def get_input(video_stream):
	frame = video_stream.read()
	(frame_height, frame_width) = frame.shape[:2]
	return frame, frame_width, frame_height


def preprocess_input(net, frame):
	blob = cv2.dnn.blobFromImage(frame, 1.0/255, (416,416), swapRB=True, crop=False)
	net.setInput(blob)


def forward_pass(net, out_layer_names):
	yolo_output = net.forward(out_layer_names)
	return yolo_output


def filter_output(yolo_output, frame_width, frame_height):
	boxes = []
	confidences = []
	classIDs = [] 

	for output in yolo_output:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID] 

			if confidence > args["confidence"]:
				box_data = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height]) 
				(center_X, center_Y, box_width, box_height) = box_data.astype("int")
				x = int(center_X - (box_width/2))
				y = int(center_Y - (box_height/2))
				boxes.append([x, y, int(box_width), int(box_height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	return boxes, confidences, classIDs, indices


def draw_box(frame, boxes, confidences, classIDs, indices, labels, colors):
	if len(indices) > 0:
		for i in indices.flatten():
			x,y,w,h = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
			color = [int(c) for c in colors[classIDs[i]]]
			cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
			object_name = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
			cv2.putText(frame, object_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

	return frame


def show_output(frame, fps_record):
	stop = 0
	cv2.imshow("Frame", frame)
	if cv2.waitKey(25) & 0xFF == ord("q"):
		stop = 1

	fps_record.update()
	return fps_record, stop


def loop_frames(labels, colors, net, out_layer_names, video_stream, fps_record):
	while True:
		frame, frame_width, frame_height = get_input(video_stream)
		preprocess_input(net, frame)
		yolo_output = forward_pass(net, out_layer_names)
		boxes, confidences, classIDs, indices = filter_output(yolo_output, frame_width, frame_height)
		output_frame = draw_box(frame, boxes, confidences, classIDs, indices, labels, colors)
		fps_record, stop = show_output(output_frame, fps_record)
		if stop == 1:
			break

	return fps_record


def clean_up(fps_record, video_stream):
	fps_record.stop()
	print("Video time: {:.2f}".format(fps_record.elapsed()))
	print("Approximate FPS: {:.2f}".format(fps_record.fps()))

	cv2.destroyAllWindows()
	video_stream.stop()


def run():
	labels, net, out_layer_names = get_model()
	colors = get_color(labels)
	video_stream, fps_record = init_video()
	fps_record = loop_frames(labels, colors, net, out_layer_names, video_stream, fps_record)
	clean_up(fps_record, video_stream)


run()
