# Functions to save and display image
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import matplotlib.patches as patches

def show(img,bounding_box='None'):
    try:
        img = img.cpu()
        npimg = img.detach().numpy()
    except:
        print('CPU ')
        pass

    _,obj = plt.subplots(1)
    obj.imshow(np.transpose(img, (1, 2, 0)))
    if bounding_box != 'None':
        rect = patches.Rectangle(bounding_box[0:2],bounding_box[2]-bounding_box[0],bounding_box[3]-bounding_box[1],linewidth=1,edgecolor='w',facecolor='none')
        obj.add_patch(rect)

    plt.show()
    plt.close('all')


def save(a, img, path):
    img = img[0].cpu()
    npimg = img.detach().numpy()
    io.imsave(str(path) + str(a) + '.jpg', np.transpose(npimg, (1, 2, 0)))
