from realsense import *

if __name__ == '__main__':

	cfgfile = "cfg/yolov3.cfg"
	weightsfile = "yolov3.weights"
	confidence = 0.25
	nms_thesh = 0.4
	CUDA = torch.cuda.is_available()
	model = Darknet(cfgfile)
	model.load_weights(weightsfile)
	model.net_info["height"] = 160
	inp_dim = int(model.net_info["height"])
	if CUDA:
		model.cuda()
	model.eval()
	videofile = 0
	cap = cv2.VideoCapture(videofile)
	while True:
		ret, frame = cap.read()
		img, box = yolo_output(frame,model,['cell phone'], confidence, nms_thesh, CUDA, inp_dim)
		box = (box[0][0], box[0][1], box[0][2]-box[0][0], box[0][3]-box[0][0])
		print(box)
		cv2.imshow("frame", img)
		cv2.waitKey(1)
			