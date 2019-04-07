import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import imghelpers as im
import argparse
import os
from skimage import io


def main():

    if not os.path.exists(args.out_rdir):
        os.mkdir(args.out_rdir)
        print('Created out_rgb_dir')
    if not os.path.exists(args.out_ddir):
        os.mkdir(args.out_ddir)
        print('Created out_depth_dir')
    if not os.path.exists(args.annotations_dir):
        os.mkdir(args.annotations_dir)
        print('Created args_out_dir')

    #Create pipeline for realsense2
    pipe = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
    profile = pipe.start(config)

    # Declare filters
    dc_ftr = rs.decimation_filter ()   # Decimation - reduces depth frame density
    st_ftr = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
    tp_ftr = rs.temporal_filter()    # Temporal   - reduces temporal noise
    th_ftr = rs.threshold_filter()

    align_to = rs.stream.color
    align = rs.align(align_to)
    rgbd = np.zeros((1,4,360,640))
    i=0
    print('################|Initialization seguence completed.|################')


    try:
        while(1):

            # Get frameset of color and depth
            frames = pipe.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() #  640x480
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # Post processing
            filtered = tp_ftr.process(st_ftr.process(dc_ftr.process(aligned_depth_frame)))
            filtered = th_ftr.process(filtered)#can be removed if person is outside frame

            rgb = np.asanyarray(color_frame.get_data())
            rgb = np.transpose(rgb,(2,0,1))
            depth = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.uint16)
            depth = depth
            im.show(rgb,args.bb)
            im.rgbsave(i, rgb, str('./' + args.out_rdir))
            im.dsave(i, depth, str('./' + args.out_ddir))
            np.savetxt('./' + args.annotations_dir + '/' + str(i) + '.csv', args.bb, delimiter=',')
            i = i+ 1
            print('saved'+str(i)+'image')
            # time.sleep(1)
    finally:
        pipe.stop()
        print('################| Error in pipeline |################')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-or", "--out_rdir", type=str, help="Path to save the rgb images")
    parser.add_argument("-od", "--out_ddir", type=str, help="Path to save the depth images")
    parser.add_argument("-oa", "--annotations_dir", type=str, help="Path to save the annotaions")
    parser.add_argument('--bb', nargs='+', type=int, help="bounding box coordinates = xmin ymin xmax ymax")
    args = parser.parse_args()
    main()
