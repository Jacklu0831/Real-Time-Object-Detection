# Object Detection with YOLOv3 and YOLOv3-tiny

YOLO object detection with pretrained model from [Darknet](https://pjreddie.com/darknet/yolo/) on the COCO 2017 dataset. My implementations of YOLO include object detection for images, pre-recorded videos, and live-webcam videostreams. I also tried out YOLOv3-tiny, measured its FPS, and compared its performance to YOLOv3. 

Note that I did not build and train the models in this repo since my primary focus was to test out performances of different models, become robust with OpenCV (images, videos, draw boxes, preprocess), and also to find out how good is my laptop at handling live object detection (spoilers the result is real disappointing for this one). For my own models, just refer to most of my other repos such as [SRGAN](https://github.com/Jacklu0831/Super-Resolution-GAN). 

---

## Performance (real-time)

Below are screen-captured videos that let me and you really see the difference in FPS between the two outputs.

<h3 align="center"><b>YOLOv3 - avg 2.42 FPS</b></h3>

<p align="center"><image src="vid_live/yolo.gif"></image></p>

<h3 align="center"><b>YOLOv3 (tiny) - avg 5.73 FPS</b></h3>

<p align="center"><image src="vid_live/yolotiny.gif"></image></p>

**Frames per Second**

Note that the FPS is only how fast my laptop is able to handle the job and they are only here to compare the performance between models. For example, [here](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006) YOLOv2 ran at 45 FPS and tiny ran at 155 FPS on good GPUs. However, it could be calculated that YOLOv3-tiny is almost 2.5 as fast than YOLOv3, implying much less computational expense due to having a shallower neural architecture and less parameters to train (more details in Background section).

A effective way to increase the FPS is to simply modify the config file of the models to decrease the frame size being processed by the network. However, low resolution images contain less information, the feature extraction process will be much more prone to errors and small objects that are further away will not be contained in the pixelated images at all (this is actually a big problem I am facing in the [SRGAN project](https://github.com/Jacklu0831/Super-Resolution-GAN)). By decreasing input frame resolution, I was able to increase the FPS of YOLOv3 to as fast as YOLOv3-tiny at the cost of worse accuracy than YOLOv3-tiny, which is expected. 

**Accuracy**

With the same confidence threshold of 0.5 and a threshold overlap ratio (non-maximum suppression coefficient) of 0.3 for both YOLOv3 and YOLOv3-tiny. YOLOv3 is able to detect objects with more precise bounding box positions and dimensions with more stable confidence scores. Also, YOLOv3 detected the traffic lights from afar, which is very important for [self-driving cars](https://github.com/Jacklu0831/Self-Driving-Car).

---

## Background

### Pipeline

<p align="center"><image src="assets/confidence.png"></image></p>

YOLO uses a single CNN network for both classification and object localization with bounding boxes. It could be broken down into several tasks.

1. Divide the input image into grids to find bounding boxes separately in each grid.

2. Forward pass through Conv layers to output a vector containing box position, box dimension, a box confidence score (how likely an object is there), and a number of conditional class probabilities depending on how many objects you want YOLO to recognize.

3. With so many bounding boxes, we first filter out most of them by specifying a confidence threshold. The tuned value for my implementation is `0.5`. You can override this default value through command line to play with it.

4. Compute the class confidence scores, which equals `box confidence x conditional class probabilities`.

5. To avoid making multiple detections for the same object, non-maximal suppression is performed where images with intersection over union values of more than `0.3` (my selection) are grouped together and only the one with the top score is kept.

6. Repeat above steps for each frame.

### Model Architecture

<p align="center"><image src="assets/yolo_model.png"></image></p>

Shown in the image above, YOLOv3 has 75 convolutional layers and not a single fully-connected layer, it also employs ResNet-alike structure as a way to improve accuracy from its predecessors. On the other hand, YOLOv3 only uses 16 convolutional layers and 2 fully-connected layers. Therefore, as shown in Performance section,YOLOv3-tiny has significantly less accuracy than YOLOv3. 

---

## Files

<pre>
README.md            - `self`
YOLO_img.py          - Implementation: image
YOLO_vid.py          - Implementation: video
YOLO_live.py         - Implementation: live video stream
img_IO               - Directory for input/output images                    
vid_IO               - Directory for input videos 
vid_outputs          - Directory for output videos 
assets               - Some pictures for README
</pre>

Note that the YOLOv3 and YOLOv3-tiny model config and weight files are not included due to their size. Go to [here](https://pjreddie.com/darknet/yolo/), first **definitely checkout his unique resume**, then download the config and weights files of your favorite model and pass in their paths when using `YOLO_live.py`. 

---

## How to Use

Run python scripts in command line.

YOLO_img.py (specify input and output paths)\
`
python YOLO_img.py -i img_IO/work_table.jpg -o img_IO/work_table.jpg -y yolov3 -d 10
`

YOLO_vid.py (specify input and output paths)\
`
python YOLO_vid.py -i vid_inputs/car_crash.mp4 -o vid_outputs/car_crasmp4 -y yolov3
`

YOLO_live.py (specify folder containing config and weights)\
`python YOLO_live.py -y yolov3`\
or\
`python YOLO_live.py -y yolov3-tiny`

---

## Resources

- [YOLO9000: Better, Faster, Stronger (this paper screams: worship this model](https://arxiv.org/pdf/1612.08242.pdf)
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Darknet Trained Models](https://pjreddie.com/darknet/yolo/)
- [Another person with the low FPS issue](https://github.com/pjreddie/darknet/issues/80)
- [OpenCV Tutorials](https://www.pyimagesearch.com/start-here-learn-computer-vision-opencv/)
- [YOLO vs. SSD Blog](https://technostacks.com/blog/yolo-vs-ssd/)
