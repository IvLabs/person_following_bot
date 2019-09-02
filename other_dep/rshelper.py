import pyrealsense2 as rs
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


def get_rgbd():
    """
    Return rgb and depth image from setup camera pipeline
    """

    frames = pipe.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        pass

    rgb = np.asanyarray(color_frame.get_data(),dtype=np.uint8)
    depth = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.uint8)
    return rgb.transpose(2,0,1), depth.tranpose(1,0)


def turtle_controller(frame, output):
    alpha = 1;

    c1 = tuple(output[0,1:3].int())
    c2 = tuple(output[0,3:5].int())
    box = [c1[0].item(),c1[1].item(), c2[0].item(),c2[1].item()]
    print(box)
    try:
        distance = mean(frame[int((box[0]+box[2])/2),int((box[1]+box[3])/2)]*alpha)
    except:
        distance = 0

    linear_speed = (320 - int((box[0]+box[2])/2))/320
    turn_speed = (240 - int((box[1]+box[3])/2))/240

    return linear_speed, turn_speed, distance
