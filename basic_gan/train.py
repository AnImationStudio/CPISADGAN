from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

PATH="../../CPISADGAN/deepfashion_dataset/list_eval_partition.txt"
IMG_BASEPATH = "/nitthilan/data/DeepFashion/inshop_cloth_retrival_benchmark/img/"


with open(PATH, "r") as img_list_file:
    img_path_list = img_list_file.readlines()


class DeepFashion(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.deepfashion_path_list = []
        self.root_dir = root_dir

        for img_path in img_path_list[2:]:
		    img_path = img_path.split(" ")[0]
		    split_img_path = img_path.split("/")
		    self.deepfashion_path_list.append(img_path)

        self.transform = transform

    def __len__(self):
        return len(self.deepfashion_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.deepfashion_path_list[idx])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image