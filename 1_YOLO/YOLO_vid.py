# ==================================
# Video Object Detection with YOLOv3
# ==================================

# RUN WITH EXAMPLE COMMAND BELOW:

# python YOLO_vid.py -i vid_IO/drive.mp4 -o vid_IO/drive_processed.mp4 -y yolov3

import numpy as np
import argparse
import imutils
import time
import cv2
import os

"""User inputs through command line"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


def get_model():
	"""Load YOLOv3 using cv2 built in DNN module."""
	model = cv2.dnn.readNetFromDarknet(os.path.sep.join([args["yolo"], "yolov3.cfg"]), 
												os.path.sep.join([args["yolo"], "yolov3.weights"]))

	# load COCO class labels (open coco.names file (concatenated) -> extract string -> removed lead and end
	# whitespace, split by \n)
	labels = open(os.path.sep.join([args["yolo"], "coco.names"])).read().strip().split("\n")

	# get output layer names (getLayerNames not subscriptable)
	getLayer = model.getLayerNames()
	out_layer_names = [getLayer[i[0] - 1] for i in model.getUnconnectedOutLayers()]
	# print("YOLOv3 output layer names:", *out_layer_names, sep=" ")

	return labels, model, out_layer_names


def get_color(labels):
	"""Initialize random colors."""
	np.random.seed(1)
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

	return colors


def init_video():
	"""Initialize video stream."""
	video_stream = cv2.VideoCapture(args["input"])
	writer = None # videoWriter object
	(frame_width, frame_height) = (None, None)

	return video_stream, writer, frame_width, frame_height


def get_frame_number(video_stream):
	"""Try to get total frame number for estimating process time."""
	try: 
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
		num_frame = int(video_stream.get(prop))
		print("{} total frames in video".format(num_frame))
	except:
		print("Cannot determine the approximate processing time needed.")
		num_frame = -1

	return num_frame


# -----------------------------------------------------------
# Below functions are all called in the video stream pipeline
# -----------------------------------------------------------


def preprocess_input(model, frame):
	"""Augment input and set up for forward pass."""
	blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (416, 416), swapRB=True, crop=False)
	model.setInput(blob)


def get_input(video_stream, frame_width, frame_height):
	"""Grab frames and return dimensions."""
	(grabbed, frame) = video_stream.read()
	# end of video, break out of loop
	if not grabbed:
		return None, None, None, grabbed
	if frame_width is None or frame_height is None:
		(frame_height, frame_width) = frame.shape[:2]

	return frame, frame_height, frame_width, grabbed


def forward_pass(writer, model, out_layer_names, num_frame):
	"""Forward pass, non-max suppression done by default. Estimate process time."""
	tick = time.time()
	layer_outputs = model.forward(out_layer_names)
	tock = time.time()

	# writer is uninitialized for the first frame only
	if writer == None and num_frame > 0:
		print("YOLOv3 took {:.3f} seconds for one frame".format(tock - tick))
		print("YOLOv3 takes estimated total time of {:.3f} seconds for the video".format((tock - tick) * num_frame))

	return layer_outputs


def filter_output(layer_outputs, frame_width, frame_height):
	"""Get lists for bounding box."""
	boxes = [] 
	confidences = []
	classIDs = [] 

	# process output
	for output in layer_outputs:
		for detection in output:
			scores = detection[5:] # detection starts with locational variables (0 to 1)
			classID = np.argmax(scores)
			confidence = scores[classID] 

			if confidence > args["confidence"]: # filter out low confidence
				box_data = detection[:4] * np.array([frame_width, frame_height, frame_width, frame_height]) 
				(center_X, center_Y, box_width, box_height) = box_data.astype("int")

				x = int(center_X - (box_width / 2))
				y = int(center_Y - (box_height / 2))

				# record box data, confidence, and class ID for the detected (note boxes is 2d)
				boxes.append([x, y, int(box_width), int(box_height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# with box dimension, we can now call non-maxima suppression (filtering out overlapping)
	indices = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	return boxes, confidences, classIDs, indices


def draw_box(frame, boxes, confidences, classIDs, indices, labels, colors):
	"""Draw all bounding boxes."""
	if len(indices) > 0:
		for i in indices.flatten():
			x,y,w,h = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]

			# draw, where OpenCV library certainly shines
			color = [int(c) for c in colors[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
			object_name = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
			cv2.putText(frame, object_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

	return frame


def write_to_video(writer, frame):
	"""Initialize writer for first frame."""
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v") # or change to *"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

	writer.write(frame)

	return writer


def loop_frames(labels, colors, model, out_layer_names, video_stream, writer, frame_width, frame_height, num_frame):
	"""Loop through each frame until no frame is grabbed"""
	while True:
		frame, frame_height, frame_width, grabbed = get_input(video_stream, frame_width, frame_height)
		if not grabbed:
			# return both parameters for clean_up, breaks out of loop
			print("Finished!")
			return writer, video_stream

		preprocess_input(model, frame)
		yolo_output = forward_pass(writer, model, out_layer_names, num_frame)
		boxes, confidences, classIDs, indices = filter_output(yolo_output, frame_width, frame_height)
		output_frame = draw_box(frame, boxes, confidences, classIDs, indices, labels, colors)
		writer = write_to_video(writer, output_frame)


def show_output():
	"""Show output as a frame in video stream, press q to exit and update the fps."""
	output_stream = cv2.VideoCapture(args["output"])
	if (output_stream.isOpened() == False):  
		print("Error opening video file, try opening it from the output path") 

	while output_stream.isOpened():
		grabbed, frame = output_stream.read()
		if grabbed == True:
			cv2.imshow('Frame', frame)
			# press Q to exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			break

	# clean up output
	output_stream.release() 
	cv2.destroyAllWindows() 


def clean_up(writer, video_stream):
	"""Stop recording fps and display performance data."""
	writer.release()
	video_stream.release()


def run():
	"""Organize and call the useful functions."""
	labels, model, out_layer_names = get_model()
	colors = get_color(labels)
	video_stream, writer, frame_width, frame_height = init_video()
	num_frame = get_frame_number(video_stream)
	writer, video_stream = loop_frames(labels, colors, model, out_layer_names, video_stream, writer, frame_width, frame_height, num_frame)
	clean_up(writer, video_stream)
	show_output()


run()



