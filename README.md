# Person Following Robot
This project has been presented in the 'International Conference on Advances in Mechanical Engineering, 2020'. [Link to paper](https://www.springer.com/in/book/9789811536380)

## Results:
![](https://github.com/khush3/person_following_robot/blob/master/result.gif)
## Abstract:
Helper robots are widely used in various situations, for ex-ample at airports and railway stations. This paper presents a pipeline to multiplex the tracking and detection of a person in dynamic environments using a stereo camera in real-time. Recent developments in object detection using ConvNets have led to robust person detection. These deep convolutional neural networks generally fail to run with high frames rates on devices with less computing power. Trackers are also used to retain the identity of the target person as well as impose fewer constraints on hardware. A concept of multiplexed detection and tracking is used which makes the pipeline faster by many folds. TurtleBot2 is used for prototyping the robot and tuning of the motion controller. Robot Operating System (ROS) is used to set up communication be-tween various nodes of the pipeline. The results found were comparable to current state-of-the-art person followers and can be readily used in day to day life.

# Instructions:
### Setting up remote server for faster processing:
##### Host Machine (machine running roscore):
```export ROS_MASTER_URI=http://192.168.0.113:11311``` (replace with host machine ip)
```export ROS_IP=192.168.0.123 ```(replace with host machine ip)
```export ROS_HOSTNAME=192.168.0.123 ``` (replace with host machine ip)

##### Remote Machine:
```export ROS_MASTER_URI=http://192.168.0.113:11311 ```(replace with host machine ip)
```export ROS_IP=192.168.0.123``` (replace with local machine ip)
```export ROS_HOSTNAME=192.168.0.123``` (replace with host machine ip)

## Dependencies:
##### Anaconda: [link](https://docs.anaconda.com/anaconda/install/linux/)
##### OpenCV: [link](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)
##### PyTorch: [link](https://pytorch.org/)
##### pyrealsense2: [link](https://pypi.org/project/pyrealsense2/)



