As seem from the output of YOLOv3 detection, each image takes 350-400ms just for a forward-pass on my CPU through the pre-trained model, 2-3 fps simply isn't cutting it when it comes to real-time object detection. Therefore, the SSD object detection model is my second attempt at the task. 



# What is SSD
SSD stands for single shot detector, instead of just reducing the images' spatial dimentions until the second last layer is ready for classification, SSD uses multi-scale feature maps to detect objects independently. It adds 6 extra layers to a base model, predicts bounding boxes from each of the 5 layers (last one for classification) and keep track of all of them. This way the total number of prediction boxes will be almost two magnitudes higher than YOLO! With more bounding boxes from different feature map sizes, SSD attains better coverage of location and is less likely to miss objects. 


[insert why is ssd faster]
does not deal well with low resolution is biggest con for detecting small features, yolo takes long time for high resolution, so ssd is perfect for webcam

# Comparing SSD to YOLO

SSD | YOLO
--- | ----




https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06
https://technostacks.com/blog/yolo-vs-ssd/
http://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/
https://github.com/parultaneja/Real-Time-Object-Detection-With-Sound

imutils

https://github.com/chuanqi305/MobileNet-SSD
https://www.d2l.ai/chapter_computer-vision/ssd.html