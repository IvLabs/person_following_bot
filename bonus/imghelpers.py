# Functions to save and display image
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import matplotlib.patches as patches
from skimage import io


def show(img,bounding_box='None'):

    plt.close('all')
    try:
        # print('--Display-- ')
        img = img.cpu()
        npimg = img.detach().numpy()
    except:
        pass

    _,obj = plt.subplots(1)
    obj.imshow(np.transpose(img, (1, 2, 0)))
    if bounding_box != 'None':
        rect = patches.Rectangle(bounding_box[0:2],bounding_box[2]-bounding_box[0],bounding_box[3]-bounding_box[1],linewidth=2,edgecolor='r',facecolor='none')
        obj.add_patch(rect)
    plt.show(block=False)
    plt.pause(0.09)


def rgbsave(a, img, path):

    try:
        # print('--Saving-- ')
        img = img.cpu()
        npimg = img.detach().numpy()
    except:
        npimg = img
        pass

    io.imsave(str(path) + '/' + str(a) + '.jpg', np.transpose(npimg, (1, 2, 0)))


def dsave(a, img, path):

    try:
        # print('--Saving-- ')
        img = img.cpu()
        npimg = img.detach().numpy()
    except:
        npimg = img
        pass

    io.imsave(str(path) + '/' + str(a) + '.jpg', npimg)
