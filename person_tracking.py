import dataclass, torch
import imghelpers as im, model as m
# create dataloader instances
from torchvision import transforms, datasets

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
trainset = dataclass.RGBD_dataset('./../setA/','./../setA_depth/','./../setA_annotations/', data_transform)

#Create instace of detector model
detector = m.DETECTOR()

import torch.optim as optim
import torch.nn as nn
loss_squared = nn.MSELoss()
loss_linear = nn.L1Loss()
F_gen_optimizer = optim.Adam(detector.parameters(), lr=0.0002, betas = (0.5,0.999))

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
    print("###################|Loaded saved model|#######################")
except:
    print(">>>>>>>>>>>>Training new model<<<<<<<<<<<<<<<")


#Convert to cuda
#detector = detector.cuda()
detector.train()

#Load Losses buffer
try:
    detector_loss_buffer = np.load('./losses/detector_loss_buffer.npy')
    print('################|Loaded losses buffer file|###############')
except:
    print('>>>>>>>>>>>>>>>Could not load buffers. Created new buffers<<<<<<<<<<<<<<<<<<<<<')


print('################|Completed Initializations. Training started|################')

#Train the detector
for epoch in range(100):

    detector_loss = 0
    bounding_box = detector(rgbd)
    loss = loss_linear()
