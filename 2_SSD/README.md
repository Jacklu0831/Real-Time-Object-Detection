# Object Detection with MobileNet SSD

720 400
2560 1440

SSD object detection with pretrained MobileNet for feature extraction from [this repo](https://github.com/chuanqi305/MobileNet-SSD). The aim for this implementation is for real-time webcam object detection. A driving video is used to evaluate its performance.

Note that I did not build and train the models in this repo since my primary focus was to test out performances of different models (YOLO), become robust with OpenCV (images, videos, draw boxes, preprocess), and also to find out how good is my laptop at handling live object detection (spoilers the result is real disappointing for this one). For my own models, just refer to most of my other repos such as [SRGAN](https://github.com/Jacklu0831/Super-Resolution-GAN). 

---

## Performance (real-time)

Below are screen-captured videos that let me and you really see how fast the frame rate produced is. The reasons and the math behind why MobileNet SSD is fast will be explained in the Background section below. 

<h3 align="center"><b>SSD - avg 14.56 FPS</b></h3>

<p align="center"><image src="vid_live/ssd.gif" width="50%" height="50%"></image></p>

**Frames per Second**

Note that the FPS is only how fast my laptop is able to handle the job. With the right GPU or even better CPU, I'm sure that the FPS would rise even higher. MobileNet is designed for being able to process video frames without demanding a high computational expense, this will be talked about in detail later. 

**Accuracy**

As seen in the video, the accuracy of the model is accurate only within a very limited range. Further objects are barely detected at all. This is simply due to the model not being trained enough on this specific dataset around driving. However, the processing speed (how real-time) is the main concern fo this repo. 

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
README.md                               - `self`
YOLO_img.py                             - Implementation: image
MobileNetSSD_deploy.caffemodel
MobileNetSSD_deploy.prototxt.txt
vid_live                                - Directory for output saved live outputs
</pre>

Note that the YOLOv3 and YOLOv3-tiny model config and weight files are not included due to their size. Go to [here](https://pjreddie.com/darknet/yolo/), first **definitely checkout his unique resume**, then download the config and weights files of your favorite model and pass in their paths when using `YOLO_live.py`. 

---

## How to Use

Run python script in command line.

SSD_live.py\
`python real_time_object_detection.py`

---

## Resources

- [YOLO9000: Better, Faster, Stronger (this paper screams: worship this model](https://arxiv.org/pdf/1612.08242.pdf)
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Darknet Trained Models](https://pjreddie.com/darknet/yolo/)
- [Another person with the low FPS issue](https://github.com/pjreddie/darknet/issues/80)
- [OpenCV Tutorials](https://www.pyimagesearch.com/start-here-learn-computer-vision-opencv/)
- [YOLO vs. SSD Blog](https://technostacks.com/blog/yolo-vs-ssd/)


As seem from the output of YOLOv3 detection, each image takes 350-400ms just for a forward-pass on my CPU through the pre-trained model, 2-3 fps simply isn't cutting it when it comes to real-time object detection. Therefore, the SSD object detection model is my second attempt at the task. 



# What is SSD
SSD stands for single shot detector, instead of just reducing the images' spatial dimentions until the second last layer is ready for classification, SSD uses multi-scale feature maps to detect objects independently. It adds 6 extra layers to a base model, predicts bounding boxes from each of the 5 layers (last one for classification) and keep track of all of them. This way the total number of prediction boxes will be almost two magnitudes higher than YOLO! With more bounding boxes from different feature map sizes, SSD attains better coverage of location and is less likely to miss objects. 


[insert why is ssd faster]
does not deal well with low resolution is biggest con for detecting small features, yolo takes long time for high resolution, so ssd is perfect for webcam


https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06
https://technostacks.com/blog/yolo-vs-ssd/
http://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/
https://github.com/parultaneja/Real-Time-Object-Detection-With-Sound

imutils

https://github.com/chuanqi305/MobileNet-SSD
https://www.d2l.ai/chapter_computer-vision/ssd.html