from __future__ import division
import pyrealsense2 as rs

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *

from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, your_class):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label in your_class:
        color = (0,255,0)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img



def yolo_output(frame, model, your_class, confidence, nms_thesh, CUDA, inp_dim):
    """
    Get the labeled image and the bounding box coordinates.

    """
    num_classes = 80
    bbox_attrs = 5 + num_classes
    img, orig_im, dim = prep_image(frame, inp_dim)

    im_dim = torch.FloatTensor(dim).repeat(1,2)

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
#            im_dim = im_dim.repeat(output.size(0), 1)
    output[:,[1,3]] *= frame.shape[1]
    output[:,[2,4]] *= frame.shape[0]

    classes = load_classes('data/coco.names')
    box = list([])
    list(map(lambda x: write(x, orig_im, classes, your_class), output))
    for i in range(output.shape[0]):
        if int(output[i, -1]) == 0:
            c1 = tuple(output[i,1:3].int())
            c2 = tuple(output[i,3:5].int())
            box.append([c1[0].item(),c1[1].item(), c2[0].item(),c2[1].item()])

    return orig_im, box



if __name__ == '__main__':

    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    confidence = 0.5
    nms_thesh = 0.4
    CUDA = torch.cuda.is_available()
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = 160
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    # Setup Realsense pipeline
    pipe = rs.pipeline()
    configure = rs.config()
    width = 640; height = 480;
    configure.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    configure.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
    dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
    spat_filter = rs.spatial_filter()      # Spatial    - edge-preserving spatial smoothing
    temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
    pipe.start(configure)
    align_to = rs.stream.color
    align = rs.align(align_to)

    while(1):

        # temp = pipe.wait_for_frames()
        # aligned_frames = align.process(temp)
        # depth_frame = aligned_frames.get_depth_frame()
        # filtered = dec_filter.process(depth_frame)
        # filtered = spat_filter.process(fisltered)
        # filtered = temp_filter.process(filtered)

        # aligned_depth_frame = np.asanyarray(filtered.get_data(),dtype=np.uint8) # aligned_depth_frame is a 640x480 depth image
        # color_frame = np.asanyarray(aligned_frames.get_color_frame().get_data(),dtype=np.uint8)

        img, box = yolo_output(color_frame,model,['cell phone', 'person'], confidence, nms_thesh, CUDA, inp_dim)
        print('BOX:', box)
        cv2.imshow("frame",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(np.max(aligned_depth_frame), qnp.min(aligned_depth_frame))
        cv2.imshow("depth",aligned_depth_frame)
        key = cv2.waitKey(1)
        # print(box)
        if key & 0xFF == ord('q'):
            break
