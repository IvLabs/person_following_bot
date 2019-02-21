import argparse
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
#from skimage import io
import os


def main():
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
        print('Created out_dir')
    try:
        config = rs.config()
        rs.config.enable_device_from_file(config, args.input)
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        pipeline.start(config)
        i = 0
        while True:
            print("Saving frame:", i)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            cv.imwrite(args.directory + "/" + str(i).zfill(6) + "depth.png", depth_image)
            rgb_frame = frames.get_color_frame()
            rgb_image = np.asanyarray(rgb_frame.get_data())
            r = rgb_image[:,:,0].copy()
            b = rgb_image[:,:,2].copy()
            rgb_image[:,:,0] = b
            rgb_image[:,:,2] = r
            cv.imwrite(args.directory + "/" + str(i).zfill(6) + "rgb.png", rgb_image)
            i += 1
    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    parser.add_argument("-i", "--input", type=str, help="Bag file to read")
    args = parser.parse_args()

    main()
