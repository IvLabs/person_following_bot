import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


alpha = 1 # Define factor(pixel to mt.) for x displacement
beta = 1 # Define factor(pixel to mt.) for y displacement
# #Check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)
print('#################################|Initializing|#################################')

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

#Create instace of detector model
detector = m.DETECTOR()

#Load saved model
try:
    checkpoint = torch.load('./parameters/detector.tar')
    detector.load_state_dict(checkpoint['detector_dict'])
    print("###################|Loaded saved model|#######################")
except:
    print(">>>>>>>>>>>>Error: 1<<<<<<<<<<<<<<<")


# Convert to cuda
detector = detector.cuda()
detector.test()

print('################|Completed Initializations. Training started|################')

pipe = rs.pipeline()
profile = pipe.start()
align_to = rs.stream.color
align = rs.align(align_to)

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
        d = np.asanyarray(aligned_depth_frame.get_data())
        rgbd[0:3,:,:] = rgb;rgbd[3,:,:] = d

        # Pass obtained image to model
        bounding_box = detector(rgbd)
        im.show(np.asanyarray(color_frame.get_data()),bounding_box)

        # Find Coordinates for target person
        region = [(bounding_box[0]+bounding_box[2])/2, (bounding_box[1]+bounding_box[3])/2]
        y_dist = d[region[0]:region[0]+5, region[1]:region[1]+5]
        y_dist = beta*y_dist/5;
        x_dist = abs(region[1]-240)*aplha*y_dist
finally:
    pipe.stop()
