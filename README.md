# Real-Time-Object-Detection

<p align="center">
	<image src="1_YOLO/vid_IO/driving_out.gif"></image>
	<br>
	<i>Sample Output of YOLOv3</i>
</p>

This repo is for documenting my implementations of some real-time object detection models. The scripts for running them are all included and I wrote quick and simple instructions for anyone who wants to play with them. I also documented their frame rate when running on CPU. Please navigate to __individual project folders__ for much more technical details.

Side note, since I am currently training a [Super Resolution Generative Adversarial Network model (SRGAN)](https://github.com/Jacklu0831/Super-Resolution-GAN), I will likely use these implementions to test out how much super resolution improves object detection performance, which could be a useful application in situations where high resolution camera are unavailable. 

**Models/Techniques I have Tried so Far**
- YOLOv3 (You Only Look Once) [-> details](1_YOLO)
- YOLOv3-tiny [-> details](1_YOLO)
- MobileNet SSD (Single Shot Detection with MobileNet base model) [-> details](2_SSD)

Some other ones I plan to try include: semantic segmentation, fast R-CNN.
