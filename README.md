# Person Following Robot

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/IvLabs/person_following_bot/blob/master/LICENSE.md) 
[![GitHub stars](https://img.shields.io/github/stars/IvLabs/person_following_bot?style=social)](https://github.com/IvLabs/person_following_bot/stargazers)

This repository aims to make an algoirthm to follow a person autonomously. This algorithm has been successfully tested on a TurtleBot2. Hand gesture for stopping/starting has also added for easier target acquisition. Attempts have been made to reduce the effect of occlusion.

If you are using this repository please [cite](#citations) our [paper](https://link.springer.com/chapter/10.1007/978-981-15-3639-7_98) and star our repo. You can also request the paper on [Research Gate](https://www.researchgate.net/publication/342567284_Person_Following_Mobile_Robot_Using_Multiplexed_Detection_and_Tracking) or contact [authors](#contributors) (typically reply within a day)

Incase you find any problem with the code/repo do let us know by raising an issue. 

## Table of contents
- [Person Following Robot](#person-following-robot)
	- [Table of contents](#table-of-contents)
	- [Working Demo](#working-demo)
	- [Results](#results)
	- [Abstract](#abstract)
	- [Project Pipeline](#project-pipeline)
	- [Instructions](#instructions)
		- [Host Machine (machine running roscore)](#host-machine-machine-running-roscore)
		- [Remote Machine](#remote-machine)
		- [Run the tracking code](#run-the-tracking-code)
	- [Dependencies](#dependencies)
		- [Hardware](#hardware)
		- [Software](#software)
	- [Citations](#citations)
	- [Contributors](#contributors)

## Working Demo
[YouTube Link](https://youtu.be/XnrbU1050ls)

## Results
![](img/result.gif)

## Abstract
Helper robots are widely used in various situations, for ex-ample at airports and railway stations. This paper presents a pipeline to multiplex the tracking and detection of a person in dynamic environments using a stereo camera in real-time. Recent developments in object detection using ConvNets have led to robust person detection. These deep convolutional neural networks generally fail to run with high frames rates on devices with less computing power. Trackers are also used to retain the identity of the target person as well as impose fewer constraints on hardware. A concept of multiplexed detection and tracking is used which makes the pipeline faster by many folds. TurtleBot2 is used for prototyping the robot and tuning of the motion controller. Robot Operating System (ROS) is used to set up communication be-tween various nodes of the pipeline. The results found were comparable to current state-of-the-art person followers and can be readily used in day to day life.

## Project Pipeline 

The complete pipeline for the person following robot can be understood using the given flow chart.

![](img/pipeline.jpg)

## Instructions
<!-- Describe the hardware section properly -->

Setting up remote server for faster processing:

### Host Machine (machine running roscore)
<!-- Please how to use these instructions @khush -->

- ```export ROS_MASTER_URI=http://192.168.0.113:11311``` (replace with host machine ip)
- ```export ROS_IP=192.168.0.123 ```(replace with host machine ip)
- ```export ROS_HOSTNAME=192.168.0.123 ``` (replace with host machine ip)

### Remote Machine
- ```export ROS_MASTER_URI=http://192.168.0.113:11311 ```(replace with host machine ip)
- ```export ROS_IP=192.168.0.123``` (replace with local machine ip)
- ```export ROS_HOSTNAME=192.168.0.123``` (replace with host machine ip)

### Run the tracking code
- First install all requirements using `pip3 install requirements.txt`. Hardware requirements can be found [here](#dependencies)
- Download Yolov3 weight file from [here](https://pjreddie.com/media/files/yolov3.weights) and place it inside the cloned repo. 
- Run the [follow.py](follow.py) code to see the output. (Bash command: `python3 follow.py`)

## Dependencies

### Hardware
Its recommended to have a Nvidia GPU for faster processing
- TurtleBot 2 platform
- Intel Realsense Depth Camera
- PC/Onboard Computer(Jetson or )

### Software 

- Python 3.6+
- OpenCV-python: [link](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)
- PyTorch: [link](https://pytorch.org/)
- pyrealsense2: [link](https://pypi.org/project/pyrealsense2/)
- Robot Operating System (ROS): [Link](https://www.ros.org/)

## Citations

If you find our research useful and want to use it, please cite our paper at

```
@InProceedings{10.1007/978-981-15-3639-7_98,
author="Agrawal, Khush
and Lal, Rohit",
editor="Kalamkar, Vilas R.
and Monkova, Katarina",
title="Person Following Mobile Robot Using Multiplexed Detection and Tracking",
booktitle="Advances in Mechanical Engineering",
year="2021",
publisher="Springer Singapore",
address="Singapore",
pages="815--822",
isbn="978-981-15-3639-7"
}
```

## Contributors
- **Khush Agrawal** - [Website](https://khush3.github.io/)
- **Rohit Lal** - [Website](http://take2rohit.github.io/)
