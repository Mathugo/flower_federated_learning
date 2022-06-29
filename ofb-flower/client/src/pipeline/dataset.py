from typing import List
from torch.utils.data import Dataset
import pandas as pd
import torch, os, sys, glob
from skimage import io
from PIL import Image
import numpy as np
from torchvision import transforms
sys.path.append("../..")
from models import *
""" dataset """

class ClassifyDataset(Dataset):
    """ Defects classification Dataset """
    def __init__(self, root_dir: str, extension: str="*.jpg", transform: transforms=None, input_size=(224, 224)):
        """
        Args: 
            root_dir (string): Directory where the image are located (dir look like : root_dir/label1/img.jpg)
        """
        self._root_dir = root_dir
        self._extension=extension
        self._labels = self._get_labels()
        self._transform = transform
        self._input_size = input_size
        self._n_classes = 0
        print("[DATASET] Len of the dataset {}".format(len(self._labels[0])))

    def _get_labels(self) -> List:
        """ Get labels from a root directory, in dataframe format {'images': [], 'labels': []} """
        images = []
        labels = [] 
        print("[DATASET] Loading labels ..")
        # encode labels to int
        label = 0
        self._n_classes=0
        if os.path.isdir(self._root_dir):
            for label_dir in os.listdir(self._root_dir):
                if os.path.isdir(os.path.join(self._root_dir, label_dir)):
                    print("[LABEL] Getting label {} -> {}".format(label, label_dir))
                    self._n_classes+=1
                    label_dir = os.path.join(label_dir, self._extension)
                    for img in glob.glob(os.path.join(self._root_dir, label_dir)):
                        images.append(img)
                        labels.append(label)
                    label+=1
            one_hot = (torch.nn.functional.one_hot(torch.tensor((labels)), self._n_classes)).type(torch.float)
            print("[DATASET] Done")
            return [images, one_hot]
        else:
            raise Exception('root_dir', '{} is not a valid directory'.format(self._root_dir))

    def __len__(self) -> int:
        return len(self._labels[0])
    
    def __getitem__(self, idx: int):
        """ get single item/image from the dataset"""
        if torch.is_tensor(idx):
            print("Tensor in getitem !")
            idx = idx.tolist()
        img_name = self._labels[0][idx]
        one_hot_label = self._labels[1][idx]
        image = Image.open(img_name)
        if self._transform != None:
            image = self._transform(image)
            #self._label_tf(one_hot_label)

        return image, one_hot_label

    @property
    def labels(self):
        return self._labels

class GearDataset(Dataset):
    """ Gear Dataset """
    
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._gear_labels = pd.read_csv(csv_file, delimiter=',')
        self._root_dir = root_dir
        self._transform = transform

    def __len__(self) -> int:
        return len(self._gear_labels)
    
    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self._root_dir,
                                self._gear_labels.iloc[idx, 0])
        image = io.imread(img_name)
        labels = self._gear_labels.iloc[idx, 1:]
        labels = np.array([labels])
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
        return sample


