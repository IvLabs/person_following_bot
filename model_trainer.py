import dataclass, torch
import imghelpers as im, model as m
from torchvision import transforms, datasets
import numpy as np

#Check for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)
print('#################################|Initializing|#################################')

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# Create datasets
trainset = dataclass.RGBD_dataset('./../FINAL_dataset/RGB/', './../FINAL_dataset/DEPTH/', './../FINAL_dataset/ANNOTATIONS/', data_transform)

#Create instace of detector model
detector = m.DETECTOR()


import torch.optim as optim
import torch.nn as nn
loss_squared = nn.MSELoss()
loss_linear = nn.L1Loss()
detector_optimizer = optim.Adam(detector.parameters(), lr=0.0002, betas = (0.5,0.999))

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

detector.apply(weights_init)

#Convert to cuda
#detector = detector.cuda()


#Load previously saved model
try:
    checkpoint = torch.load('./parameters/detector.tar')
    detector.load_state_dict(checkpoint['detector_dict'])
    detector_optimizer.load_state_dict(checkpoint['detector_optimizer_dict'])
    print("##############################|Loaded saved model|##############################")

except:
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Training new model<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

#Convert to cuda
#detector = detector.cuda()
detector.train()

#Load Losses buffer
try:
    detector_loss_buffer = np.load('./losses/detector_loss_buffer.npy')
    print('###########################|Loaded losses buffer file|##########################')

except:
    print('>>>>>>>>>>>>>>>>Could not load buffers. Created new buffers<<<<<<<<<<<<<<<<<<<<<')

    detector_loss_buffer = []

rgb_b = torch.zeros(1,4,360,640)

print('#################|Completed Initializations. Training started|##################')


#Train the detector
for epoch in range(100):

    detector_loss = 0
    running_detector_loss = 0
    i = 0

    for rgb, depth, annotation in trainset:

        i += 1;
        annotation = torch.FloatTensor(annotation)
        rgb_b[0,0:3,:,:] = rgb
        rgb_b[0,3,:,:] = depth[0,:,:]
        detector_optimizer.zero_grad()
        bounding_box = detector(rgb_b)
        detector_loss = nn.functional.l1_loss(bounding_box, annotation)
        detector_loss.backward()
        detector_optimizer.step()

        running_detector_loss += detector_loss.item()

        if i % 50 == 49:    # print every 2000 mini-batches
            print('Epoch: %d | No of images: %5d | Total Loss: %.3f' %
                  (epoch + 1, i + 1,running_detector_loss / 50))
            detector_loss_buffer = np.append(detector_loss_buffer, running_detector_loss/50)
            running_detector_loss = 0
            #save loss buffers
            np.save('./losses/detector_loss_buffer',detector_loss_buffer)

    torch.save({
        'detector_dict': detector.state_dict(),
        'detector_optimizer_dict':detector_optimizer.state_dict(),
    }, './parameters/model' + str(epoch)+ '.tar')
