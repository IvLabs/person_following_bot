# Functions to save and display image
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def show(img):
    img = img.cpu()
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def save(a, img, path):

    img = img[0].cpu()
    npimg = img.detach().numpy()
    io.imsave(str(path) + str(a) + '.jpg', np.transpose(npimg, (1, 2, 0)))
