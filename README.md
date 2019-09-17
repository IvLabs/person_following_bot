# Person Following Robot
* *This Repository will be updated once I receive updates on my submission for a research conference*
# Overview:
This is a deep learning based robot which follows a person in dynamic environment. Currrently I am usign a Turtle-bot which is an easy to use robotic development platform.

# Challenges:
Setting live realsense stream in python code:
Found this python module:https://pypi.org/project/pyrealsense2/

# Instructions:

# Dependencies:
# Anaconda:
https://docs.anaconda.com/anaconda/install/linux/
# OpenCV:
https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html
# Pytorch:
https://pytorch.org/

# Function:
# 1.change_res.py
Execute ```python change_res.py -h [help]```

Prints help options to terminal

Execute ```python change_res.py -d [output_directory] -i [input_directory] -hd [height_output_img] -wd [width_output_img]```

Change resolution of input images (from input_directory) to images of resolution (hd, wd) (to output_directory)

# 2.extract_images.py
Execute ```python extract_images.py -h [help]```

Prints help options to terminal

Execute ```python extract_images -d [output_directory] -i [input_bag_file]```

Extract images from input bag file.

# 3.xmlparser.py
Execute ```python xmlparser.py -h [help]```

Prints help options to terminal

Execute ```python xmlparser.py -d [output_directory] -i [input_directory]```

Parse xml files to csv files and write (to output_directory). Change child address to parse specific end.

# Modules:
# imagehelpers:
1. show(image, bounding_box='None'):displays image using matplotlib with bounding box(optional).

coordinates of bounding box are optional it should it in the form:[Xmin,Ymin,Xmax,Ymax]

# Running across multiple machines:
# for host machine ( machine running roscore):
```export ROS_MASTER_URI=http://192.168.0.113:11311``` (replace with host machine ip)

```export ROS_IP=192.168.0.123 ```(replace with host machine ip)

```export ROS_HOSTNAME=192.168.0.123 ``` (replace with host machine ip)

# for remote machine:
```export ROS_MASTER_URI=http://192.168.0.113:11311 ```(replace with host machine ip)
```export ROS_IP=192.168.0.123``` (replace with local machine ip)
```export ROS_HOSTNAME=192.168.0.123``` (replace with host machine ip)
