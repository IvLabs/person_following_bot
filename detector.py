print('#################################|Initializing|#################################')
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import model as m
import imghelpers as im


alpha = 1 # Define factor(pixel to mt.) for x displacement
beta = 1 # Define factor(pixel to mt.) for y displacement
# #Check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)
#
# data_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#Create instace of detector model
detector = m.DETECTOR()

#Load saved model
try:
    checkpoint = torch.load('./parameters/model5.tar')
    detector.load_state_dict(checkpoint['detector_dict'])
    print("###################|Loaded saved model|#######################")
except:
    print(">>>>>>>>>>>>Error: 1<<<<<<<<<<<<<<<")


# Convert to cuda
# detector = detector.cuda()
detector.eval()

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
profile = pipe.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
rgbd = torch.zeros(1,4,360,640)
i=0

print('################|Initialization seguence completed.|################')
try:
    while(1):
        # Get frameset of color and depth
        frames = pipe.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        rgb = np.asanyarray(color_frame.get_data())
        rgb = np.transpose(rgb,(2,0,1))
        d = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.uint8)
        # d = np.transpose(d,(1,0))
        rgbd[0,0:3,:,:] = torch.tensor(rgb);
        rgbd[0,3,:,:] = torch.tensor(d)

        # Pass obtained image to model
        bounding_box = detector(rgbd)
        bb = list(map(int, bounding_box[0,0,0,:]))
        if i%20 ==9:
            im.show(rgb, bb)
            print(bounding_box)


        # Find Coordinates for target person
        region = [int((bounding_box[0,0,0,0] + bounding_box[0,0,0,2])/2), int((bounding_box[0,0,0,1] + bounding_box[0,0,0,3])/2)]
        y_dist = d[region[0]:region[0]+5, region[1]:region[1]+5]
        y_dist = beta*y_dist/5;
        x_dist = abs(region[1]-240)*alpha*y_dist
        i = i+ 1
finally:
    pipe.stop()
