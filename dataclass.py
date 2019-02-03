from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import xml.etree.ElementTree as ET

class RGBD_dataset(Dataset):
    """RGBD dataset."""

    def __init__(self, path_rgb, path_depth, path_annotations, transform=None):
        """
        Args:
            path_rgb (string): Path to the rgb images directory.
            path_depth (string): Path to the depth images directory.
            path_annotations (string): Path to the annotations directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rgb_path = path_rgb
        self.depth_path = path_depth
        self.annotations_path = path_annotations
        self.transform = transform
        self.list_rgb =  os.listdir(path_rgb)
        self.list_depth =  os.listdir(path_depth)
        self.list_annotations = os.listdir(path_annotations)

    def __len__(self):
        return len(self.list_annotations)

    def __getitem__(self, idx):
        annotations_name = os.path.join(self.annotations_path, self.list_annotations[idx])
        tree = ET.parse(annotations_name)
        root = tree.getroot()
        ANNOTATIONS = [ root[6][4][0].text, root[6][4][1].text, root[6][4][2].text, root[6][4][3].text]
        ANNOTATIONS = list(map(int, ANNOTATIONS))
        rgb_name = os.path.join(self.rgb_path, self.list_annotations[idx].strip('.xml') + '.jpg')
        RGB = io.imread(rgb_name)
        depth_name = os.path.join(self.depth_path, self.list_annotations[idx].strip('_rgb.xml') + '_depth.jpg')
        DEPTH = io.imread(depth_name)
        if self.transform:
            RGB = self.transform(RGB)
            DEPTH = self.transform(DEPTH)
        return RGB,DEPTH, ANNOTATIONS
