 ##### Follower Python3.6
#   Basic Follower using only detection - Yes
#   Tested Tracking Algorithms -
#   GAN Trained to obtain depth -
#
#
from __future__ import division
print('#####################| INITILIZATION SEQUENCE STARTED |##################')
import rospy
import message_filters
import sys, select, tty, time, argparse, torch
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
try:
    sys.path.append('yolo_dep/')
    sys.path.append('other_dep/')
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import torch.nn as nn
import numpy as np
import pickle as pkl
import rshelper as rsh
import pyrealsense2 as rs
from torch.autograd import Variable
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image


def write(x, img):

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label == "person":
        color = (255, 0, 0)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    else:
        print('No person detected')
    return img


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v2 Video Detection Module')

    parser.add_argument("--video", dest = 'video', help =
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--height", dest = "height", type = int, help = "height of images", default = 360)
    parser.add_argument("--width", dest = "width", type = int, help = "width of images", default = 640)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "yolo_dep/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolo_dep/data/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()



def get_bounding_box(msg, _):
    """
    Callback function to get bounding box coordinates
    """
    global output
    global frames

    frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), 1)
    cv2.waitKey(1)
    rgb = frame
    img, orig_im, dim = prep_image(rgb, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1,2)

    if CUDA:
        img = img.cuda().half()
        im_dim = im_dim.half().cuda()
        # write_results = write_results_half
        predict_transform = predict_transform_half

    output = model(Variable(img, volatile = True), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

    frames += 1
    # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

    list(map(lambda x: write(x, orig_im), output))
    cv2.imshow("frame", orig_im)
    key = cv2.waitKey(1)



def get_control(_, msg):
    """
    Callback function to publish turtlebot controlling messages
    """

    frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), 1)
    cv2.waitKey(1)
    linear, turn, distance = rsh.turtle_controller(frame, output, VERBOSE=False)

    twist = Twist()
    twist.linear.x = 3*linear; twist.linear.y = 0; twist.linear.z = 0
    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 2*turn
    try:
        pub.publish(twist)
    except:
        print('Could not publish cmd_vel')


if __name__ == '__main__':
    classes = load_classes('yolo_dep/data/coco.names')
    colors = pkl.load(open("yolo_dep/pallete", "rb"))

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda().half()

    model.eval()

    frames = 0
    width = 640; height = 480;
    start = time.time()
    global output;

    print('Press Ctrl+C for exiting')
    rospy.init_node('detector', anonymous=True)
    pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)
    sub_rgb = message_filters.Subscriber("/output/image_raw/compressed_rgb",
                                         CompressedImage, queue_size=1, buff_size=2**25)
    sub_depth = message_filters.Subscriber("/output/image_raw/compressed_depth",
                                           CompressedImage, queue_size=1, buff_size=2**25)
    ts = message_filters.TimeSynchronizer([sub_rgb, sub_depth], 10)
    ts.registerCallback(get_bounding_box)
    ts.registerCallback(get_control)

    rospy.spin()
