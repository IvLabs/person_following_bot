from __future__ import division
print('################| INITILIZATION SEQUENCE STARTED |#############')
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
#from util import *
import util
from darknet import Darknet
import pandas as pd
import random
import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyrealsense2 as rs



def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Person Follower - khush3')

    parser.add_argument("--video", dest = 'video', help =
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    # parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--height", dest = "height", type = int, help = "height of images", default = 360)
    parser.add_argument("--width", dest = "width", type = int, help = "width of images", default = 640)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()

def main():


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
    # Use external camera for detection
    # rsh.initialize_camera(args.width, args.height)
    # Use the webcam for detection
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot capture source'
    frames = 0
    # width = 640; height = 480;
    start = time.time()
    # pipe = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
    # profile = pipe.start(config)
    # align_to = rs.stream.color
    # align = rs.align(align_to)
    print('################| INITILIZATION SEQUENCE COMPLETE |#############')

    while(1):

        # rgb[1,:,:,:,], depth = rsh.get_rgbd()
        # temp = pipe.wait_for_frames()
        # aligned_frames = align.process(temp)
        # aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        # color_frame = aligned_frames.get_color_frame()
        #
        # if not aligned_depth_frame or not color_frame:
        #     pass
        #
        # rgb = np.asanyarray(color_frame.get_data(),dtype=np.uint8)
        # depth = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.uint8)
        # # rgb = rgb#.transpose(2,0,1)#, depth.tranpose(1,0)
        ret, rgb = cap.read()
        img, orig_im, dim = util.prep_image(rgb, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1,2)

        if CUDA:
            img = img.cuda().half()
            im_dim = im_dim.half().cuda()
            # write_results = write_results_half
            predict_transform = predict_transform_half

        output = model(Variable(img, volatile = True), CUDA)
        output = util.write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue


        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])


        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: util.write(x, orig_im), output))

        cv2.imshow("frame", orig_im)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print('################| QUIT |#############')
            break
        frames += 1

        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))


if __name__ == '__main__':
    main()
