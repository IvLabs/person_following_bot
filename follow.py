import sys
import rospy
from std_msgs.msg import String
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
from imutils.video import FPS
from realsense import *
from collections import namedtuple
import numpy as np
import imutils
import time
from scipy.ndimage import gaussian_filter
from stop_gesture import stop_robot, get_hand, play_music


cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
# cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)


DETECT_AFTER = 100
algo = 'kcf'
output_size = (1280,720)
REALSENSE = True
TINY = 0
PLAY_FROM_FILE = 0
occlusion = False

linear, angular = 0, 0

x,y,w,h = 0,0,0,0

prev_linear, prev_angular = 0, 0


OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}


def get_iou(boxA, boxB):
	""" Find iou of detection and tracking boxes
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


def get_controls(x, z, Kp_l, Ki_l, Kd_l, Kp_a, Ki_a, Kd_a):
	global i_error_l 
	global i_error_a
	global d_error_l
	global d_error_a

	twist = Twist()

	p_error_l = z - 1.5
	p_error_a = x - 640
	i_error_l += p_error_l
	i_error_a += p_error_a
	curr_d_error_l = p_error_l - d_error_l
	curr_d_error_a = p_error_a - d_error_a

	linear = Kp_l*p_error_l + Ki_l*i_error_l + Kd_l*curr_d_error_l
	angular = Kp_a*p_error_a + Ki_a*i_error_a + Kd_a*curr_d_error_a
	# print('linear: {} ,angular: {}  \n'.format(linear,angular))

	if linear > 0.3:
		linear = 0.3

	if angular > 0.3:
		angular = 0.3

	if linear < -0.3:
		linear = -0.3

	if angular < -0.3:
		angular = -0.3

	twist.linear.x = linear; twist.linear.y = 0; twist.linear.z = 0
	twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = angular
	
	return twist

def time_wait(duration,linear,angular):
	twist = Twist()
	twist.linear.x = linear; twist.linear.y = 0; twist.linear.z = 0
	twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = angular
	t = time.time()

	while time.time()-t < duration:
		pub.publish(twist)


def get_coordinates(box, x, y, x1, y1):
	""" Get co-ordinates of flaged person
	"""
	if len(box) == 0:
#		print('!!!!!!!!No person detected!!!!')
		return
	iou_scores = []
	for i in range(len(box)):
		iou_scores.append(get_iou(box[i],[x,y,x1,y1]))

	index = np.argmax(iou_scores)
#	print(iou_scores, ' ',box, ' ', x, y, x1, y1)

	if np.sum(iou_scores) == 0:
		# print('#'*20, 'No Match found', '#'*20)
		box = np.array(box)
		distance = np.power(((x+x1)/2 - np.array(box[:,0] + box[:,2])/2),2) + np.power(((y+y1)/2 - (box[:,1]+box[:,3])/2), 2)
		index = np.argmin(distance)

	x, y, w, h = box[index][0], box[index][1], (box[index][2]-box[index][0]), (box[index][3]-box[index][1])
	initBB = (x+w//2-50,y+h//3-50,100,100)

	return initBB, (x,y,x+w,y+h)


def get_smoother(depth_frame, box):
	if np.sum(box) == 0:
		return
	sum_ = np.zeros((100,100))
	for i in range(100):
		for j in range(100):
			sum_[i,j] = depth_frame.get_distance(box[0]+i, box[1]+j)
	smooth = gaussian_filter(sum_, sigma=7)
	const = np.sum(sum_)
	return const/(10000)



if __name__ == '__main__':
	confidence = 0.25
	nms_thesh = 0.4
	success = True
	i_error_l = 0
	i_error_a = 0
	d_error_l = 0
	d_error_a = 0
	j=0
	CUDA = torch.cuda.is_available()
	
	rospy.init_node('detector', anonymous=True)
	pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)

	cfgfile = "cfg/yolov3.cfg"
	weightsfile = "yolov3.weights"

	if TINY:
		cfgfile = "cfg/yolov3-tiny.cfg"
		weightsfile = "yolov3-tiny.weights"

	videofile = '/home/rohit/Downloads/final_5d701c13620efe001423f13a_572762.mp4'
	# videofile = '/home/rohit/Downloads/final_5d701c13620efe001423f13a_572762.mp4'
	# videofile = '/home/rex/projects_test/yolov3_simplified/test_video.mp4'
	model = Darknet(cfgfile)
	model.load_weights(weightsfile)
	model.net_info["height"] = 160
	inp_dim = int(model.net_info["height"])

	if CUDA:
		model.cuda()
	model.eval()

	pipe = rs.pipeline()
	configure = rs.config()
	if PLAY_FROM_FILE:
		rs.config.enable_device_from_file(configure, '/home/rex/Documents/zade_occluded.bag')
	width = 1280; height = 720;
	configure.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
	configure.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
	dec_filter = rs.decimation_filter()   # Decimation - reduces depth frame density
	spat_filter = rs.spatial_filter()     # Spatial    - edge-preserving spatial smoothing
	temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise
	pipe.start(configure)
	align_to = rs.stream.color
	align = rs.align(align_to)

	temp = pipe.wait_for_frames()
	aligned_frames = align.process(temp)
	depth_frame = aligned_frames.get_depth_frame()

	frame = np.asanyarray(aligned_frames.get_color_frame().get_data(),dtype=np.uint8)


	run = True
	time_start = -1
	frame_number = -1
	fps = 0
	STOP_THRESHOLD = .2

	frame_number = -1
	fps = 0
	frame_ = 0
	a = time.time()
	times = []
	frame = cv2.resize(frame, output_size, interpolation = cv2.INTER_AREA)
	initBB = cv2.selectROI('Frame', cv2.cvtColor(frame,cv2.COLOR_RGB2BGR), fromCenter=False)
	(H, W) = frame.shape[:2]
	tracker = OPENCV_OBJECT_TRACKERS[algo]()
	tracker.init(frame, initBB)
	calc_z_prev = 0
	while True:
		try:

			start = time.time()
			frame_number+=1
			frame_ += 1

			# calc_x,calc_y = initBB[0]+initBB[2]//2, initBB[1]+initBB[3]//2
		
			temp = pipe.wait_for_frames()
			aligned_frames = align.process(temp)
			
			depth_frame = aligned_frames.get_depth_frame()
			frame = np.asanyarray(aligned_frames.get_color_frame().get_data(), dtype=np.uint8)

			frame = cv2.resize(frame, output_size, interpolation = cv2.INTER_AREA)


			# img, box = yolo_output(frame,model,['cell phone', 'person'], confidence, nms_thesh, CUDA, inp_dim)
			if frame_number % DETECT_AFTER == (DETECT_AFTER-1) or not success :

				img, yolo_box = yolo_output(frame.copy(),model,['person'], confidence, nms_thesh, CUDA, inp_dim)
				cv2.imshow('yolo', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
				# print('initial: ', yolo_box)
				distance_list = []
				person_in_range = []
				for i in yolo_box:
					curr_dist = get_smoother(depth_frame, i)
					distance_list.append(curr_dist)	
					if 1 < curr_dist < 8:
						person_in_range.append(i)
						# yolo_box.remove(i)
						# print('distance : {:4.2f}'.format(curr_dist))


				# print('final: ', person_in_range)

				initBB, trueBB = get_coordinates(yolo_box, x, y, x+w, y+h)
				tracker = OPENCV_OBJECT_TRACKERS[algo]()
				tracker.init(frame, initBB)
				# print('new tracker')
				fps = (frame_)//(time.time()-a)
				frame_ = 0
				a = time.time()

			# cv2.rectangle(frame, (b1,b2), (b3,b4),(0, 255, 255), 2)
			(success, box) = tracker.update(frame)

			if success:
				(x, y, w, h) = [int(v) for v in box]
				cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

				calc_x, calc_z = (x+w/2), depth_frame.get_distance(x+w//2, y+h//2)
				# print('distance: {:2.2f}'.format(calc_z))
				twist = get_controls(calc_x, calc_z, 1/5, 0, 0.1,-1/500, 0, 0)

				if twist.linear.x < STOP_THRESHOLD:
					hand_x_relative, hand_y_relative = get_hand(cv2.cvtColor(frame[trueBB[1]:trueBB[3], trueBB[0]:trueBB[2],:],cv2.COLOR_BGR2RGB))
					hand_x, hand_y = (hand_x_relative + trueBB[0]), (hand_y_relative + trueBB[1])
					hand_dist = depth_frame.get_distance(hand_x, hand_y)
					run, time_start = stop_robot(hand_dist, calc_z, time_start, run, dist_threshold=.3, time_threshold=1.0)

				
		
			# print(calc_z_prev, calc_z, 'flag:', occlusion)
			
			if  calc_z_prev - calc_z > 0.50:
				occlusion = not occlusion
				# pub.publish(prev_linear, prev_anular)
				# play_music('/home/rohit/Downloads/nikal_crop.mp3')
				time_wait(1, prev_linear, prev_angular)
				# time.sleep(1)
				print(j , ' :occlusion')
				j+=1
		
			twist.linear.x *= run
			twist.angular.z *= run
			if not occlusion:
				pub.publish(twist)		

			occlusion = False
				# time.sleep(1)
				# occlusion = False
			calc_z_prev = calc_z
			prev_linear, prev_angular = twist.linear.x, twist.angular.z

			# if frame__%200 == 0:
			# 	prev_linear, prev_angular = twist.linear.x, twist.angular.z

			info = [("Tracker", algo),("Success", "Yes" if success else "No"),("FPS", "{:.2f}".format(fps)),]

			for (i, (k, v) ) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			
			# cv2.imshow("Depth", np.asanyarray(depth_frame.get_data()))
			cv2.imshow("Frame", cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
			# times.append(10//((end - start)))
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
		except NameError:
			pass
		
		# except ValueError:
		# 	raise e

		except Exception as e:
			# raise e
			# pass
			print(e)

	cv2.destroyAllWindows()