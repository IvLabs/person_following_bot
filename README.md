# Person Following Robot

# Overview:
This is a deep learning based robot which follows a person in dynamic environment. Currrently I am usign a Turtle-bot which is an easy to use robotic development platform. 

# Challenges:
Setting live realsense stream in python code:
Found this python module:https://pypi.org/project/pyrealsense2/

# Instructions: 
# Function: 
# 1.change_res.py
Execute $python change_res.py -h [help]

Prints help options to terminal

Execute $python change_res.py -d [output_directory] -i [input_directory] -hd [height_output_img] -wd [width_output_img]

Change resolution of input images (from input_directory) to images of resolution (hd, wd) (to output_directory) 

Execute $python extract_images.py -h [help]

Prints help options to terminal

Execute $python extract_images.py -d [output_directory] -i[input_directory]

Extract images from input directory to output directory.

# 2.extract_images.py
Execute $python extract_images.py -h [help]

Prints help options to terminal

Execute $python extract_images -d [output_directory] -i [input_bag_file]

Extract images from input bag file.

# Modules:
# imagehelpers:
1. show(image, bounding_box='None'):displays image using matplotlib with bounding box(optional).

coordinates of bounding box are optional it should it in the form:[Xmin,Ymin,Xmax,Ymax]
