import os
import cv2 as cv
list_annotations = os.listdir('./setA_annotations')
# Create class for dataset

    def __init__(self, path_rgb, path_depth, path_annotations, transform=None):
        """
        Args:
            path_X (string): Path to the X images directory.
            path_Y (string): Path to the Y images directory.
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
        annotations_name = os.path.join(self.annotations_path,
                                self.list_annotations[idx%len(self.list_annotations)])
        ANNOTATIONS = io.imread(annotations_name)
        rgb_name = os.path.join(self.rgb_path,
                                self.list_rgb[idx%len(self.list_rgb)])
        RGB = io.imread(rgb_name)
        depth_name = os.path.join(self.depth_path,
                                self.list_depth[idx%len(self.list_depth)])
        DEPTH = io.imread(depth_name)

        if self.transform:
            RGB = self.transform(RGB)
            DEPTH = self.transform(DEPTH)

        return RGB,DEPTH, ANNOTATIONS


# create dataloader instances
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
 #       transforms.RandomSizedCrop(224),
 #       transforms.RandomHorizontalFlip(),
 #       transforms.ToTensor(),
 #       transforms.Normalize(mean=[0.485, 0.456, 0.406],
 #                         std=[0.229, 0.224, 0.225]),
    ])

# Create datasets
trainset = person_dataset('./../setA/','./../setA_depth/','./../setA_annotations/', data_transform)
# test_set = person_dataset('./../person/TestA/testA','./../person/TestB/testB', data_transform)
