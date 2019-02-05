#Detector model
import torch.nn as nn
import torch.nn.functional as F

class DETECTOR(nn.Module):

    def __init__(self):
        super(DETECTOR, self).__init__()
        self.to_bounding_box = nn.Linear(19,4)
        self.block = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size = 4, stride= 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size = 4, stride= 2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size = 4, stride= 2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size = 4, stride= 2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size = 20, stride= 1),  #Apply conv to get one dimensional output
        )

    def forward(self, x):
        x = self.block(x);
        x = self.to_bounding_box(x)
        return x
