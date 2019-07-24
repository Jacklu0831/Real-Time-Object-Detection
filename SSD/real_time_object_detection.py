# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# functional programming is purely for better readability

from imutils.video import VideoStream, FPS
import numpy as np 
import argparse
from imutils import resize
import time
import cv2

# user inputs through command line (no input output since its webcam real time detection)
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe file")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.35, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# get class labels
def get_model():
	labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	return labels, net


def get_color(labels):
	# random colors
	np.random.seed(1)
	colors = np.random.uniform(0, 255, size=(len(labels),3))

	return colors


def init_video():
	# start camera (2 seconds warm up), imutils is a series of convenient packages
	video_stream = VideoStream(src=0).start()
	time.sleep(2.0)
	# keeps track of the frames per second, note it includes all other processes not just forward pass
	fps_record = FPS().start()

	return video_stream, fps_record


#
# below functions are all called in the loop_frame function
#

def get_input(video_stream):
	# grab frames and return dimensions
	frame = video_stream.read()
	frame = resize(frame, width = 400)
	(frame_height, frame_width) = frame.shape[:2]

	return frame, frame_width, frame_height


def preprocess_input(net, frame):
	# augment input and set up for forward pass
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
	net.setInput(blob)


def forward_pass(net):
	# forward pass, note that non-max suppression is done by default
	SSD_output = net.forward()
	return SSD_output


def draw_box(labels, SSD_output, frame, frame_width, frame_height, colors):
	# for every bounding box
	for i in np.arange(0, SSD_output.shape[2]):
		# extract confidence
		confidence = SSD_output[0, 0, i, 2]
		if confidence > args["confidence"]:
			index = int(SSD_output[0,0,i,1])
			# 3:7 for two diagonal dots on 2D plane
			box = SSD_output[0,0,i,3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
			# diagonal coordinates
			x1, y1, x2, y2 = box.astype("int")
			cv2.rectangle(frame, (x1,y1), (x2,y2), colors[index], 1)

			# get text vertical position
			y_text = y1-15 if y1>30 else y1+15
			text = "{}: {:.2f}%".format(labels[index], confidence * 100)
			cv2.putText(frame, text, (x1,y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 1) 

	return frame


def show_output(frame, fps_record):
	# show output as a frame in video stream, press q to exit and update the fps
	stop = 0
	cv2.imshow("Frame", frame)
	if cv2.waitKey(25) & 0xFF == ord("q"):
		stop = 1

	fps_record.update()
	return fps_record, stop


def clean_up(fps_record, video_stream):
	# stop recording fps and display performance data
	fps_record.stop()
	print("Video time: {:.2f}".format(fps_record.elapsed()))
	print("Approximate FPS: {:.2f}".format(fps_record.fps()))

	cv2.destroyAllWindows()
	video_stream.stop()


def loop_frames(labels, colors, net, video_stream, fps_record):
	while True:
		frame, frame_width, frame_height = get_input(video_stream)
		preprocess_input(net, frame)
		SSD_output = forward_pass(net)

		output_frame = draw_box(labels, SSD_output, frame, frame_width, frame_height, colors)
		fps_record, stop = show_output(frame, fps_record)
		if stop == 1:
			break

	return fps_record


def run():
	labels, net = get_model()
	colors = get_color(labels)
	video_stream, fps_record = init_video()
	fps_record = loop_frames(labels, colors, net, video_stream, fps_record)
	clean_up(fps_record, video_stream)


run()
