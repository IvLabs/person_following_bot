"""Read Images from camera and publish
"""

import sys
#try:
#    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#except:
#    pass
# __author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
# __version__=  '0.1'
# __license__ = 'BSD'
import sys, time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage

VERBOSE=False



def pub_img():
    '''Callback function of subscribed topic.
    Here images get converted and features detected'''

    cap = cv2.VideoCapture(0)
    print('Initilized')

    while(True):

        ret, frame = cap.read()
        #### Create Compressed Image ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "rgb"

        msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        # Publish new image
        pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=1)
        # r = rospy.Rate(1)
        try:
            pub.publish(msg)
            # r.sleep()
        except:
            print('Could not publish')


def main(args):
    '''Initializes and cleanup ros node'''
    # ic = image_feature()
    rospy.init_node('image_publisher', anonymous=True)
    pub_img()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
