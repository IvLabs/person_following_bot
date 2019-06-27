import realsese2 as rs
import numpy as np
import cv2 as cv


def initialize_camera(width, height):
    """
    Set up camera pipeline for given resolution
    """

    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
    profile = pipe.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)


def get_test_input(input_dim, CUDA):

    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_
