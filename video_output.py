"""Read Images from camera and publish
"""

import sys
import sys, time
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
import pyrealsense2 as rs
VERBOSE=False
try:
   sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
   pass
# __author__ =  'Khush Agrawal <agrawalkhush2000@gmail.com>'
import cv2 as cv
global stamp
stamp = 0



def pub_img():
    '''Callback function of subscribed topic.
    Here images get converted and features detected'''
    global stamp

    # cap = cv2.VideoCapture(1)
    config = rs.config()
    rs.config.enable_device_from_file(config, '20190722_223724.bag')
    pipeline = rs.pipeline()
    #config.enable_stream(rs.stream.depth, 640, 320, rs.format.z16, 30)
    #config.enable_stream(rs.stream.color, 640, 320, rs.format.rgb8, 30)
    pipeline.start(config)
    i = 0

    pub_rgb = rospy.Publisher("/output/image_raw/compressed_rgb", CompressedImage, queue_size=1)
    pub_depth = rospy.Publisher("/output/image_raw/compressed_depth", CompressedImage, queue_size=1)


    print('Hey! I started publishing images.')

    while(True):

        print("Saving frame:", i)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
    #cv.imwrite(args.directory + "/" + str(i).zfill(6) + "depth.png", depth_image)
        rgb_frame = frames.get_color_frame()
        rgb_image = np.asanyarray(rgb_frame.get_data())
        cv.imshow('this', rgb_image)
        cv.waitKey(1)

        # buffer = np.zeros((480,640,4))

        # buffer[:,:,0:3] = rgb_image#(np.asanyarray(color_frame.get_data(),dtype=np.uint8))
        # buffer[:,:,3] = depth_image#np.asanyarray(aligned_depth_frame.get_data(),dtype=np.uint8)

        #### Create CompressedImage ####
        msg_rgb = CompressedImage()
        msg_depth = CompressedImage()
        msg_rgb.header.stamp = rospy.Time.now()
        msg_depth.header.stamp = msg_rgb.header.stamp#rospy.Time.now()
        msg_rgb.format = "rgb"#f'{stamp}'
        msg_depth.format = "depth"
        msg_rgb.data = np.array(cv.imencode('.jpg', rgb_image)[1]).tostring()
        msg_depth.data = np.array(cv.imencode('.jpg', depth_image)[1]).tostring()
        # Publish depth frame
        # pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=1)
        # r = rospy.Rate(10)
        # try:
        pub_rgb.publish(msg_rgb)
        pub_depth.publish(msg_depth)
        # r.sleep()
        # except:
        #     print('Could not publish')

        stamp += 1
        if stamp >= 100000:
            stamp = 0

def main(args):
    '''Initializes and cleanup ros node'''
    # ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    pub_img()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
