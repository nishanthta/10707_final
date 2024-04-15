from collections import defaultdict
import time
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from statistics import mean

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from itertools import combinations

import random
from tqdm import tqdm # Progress Bar

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
from statistics import mean

import cv2

import itertools
import random
import os
from utils import *
from config import *

from sklearn.metrics import classification_report, confusion_matrix


class TripletDataset(Dataset):
    
    def __init__(self, training_df=None,transform=None):
        self.training_df = training_df
        self.training_df.columns = ["Anchor_Path", "Positive_Path", "Negative_Path"]  
        self.transform = transform

    def __getitem__(self, index):
        # Getting the image paths
        anchor_path = os.path.join(self.training_df.iat[int(index), 0])
        positive_path = os.path.join(self.training_df.iat[int(index), 1])
        negative_path = os.path.join(self.training_df.iat[int(index), 2])

        # Loading the images
        anchor_img = Image.open(anchor_path)
        positive_img = Image.open(positive_path)
        negative_img = Image.open(negative_path)
        
        # preprocess the image
        anchor_img = preprocess_image(anchor_img)
        positive_img = preprocess_image(positive_img)
        negative_img = preprocess_image(negative_img)
        
        # Apply image transformations
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.training_df)


class DupletDataset(Dataset):
    
    def __init__(self,dataframe=None,transform=None):
        # used to prepare the labels and images path
        self.training_df=dataframe
        self.training_df.columns =["image1","image2","label"]   
        self.transform = transform

    def __getitem__(self,index):
        
        # getting the image path
        image1_path=os.path.join(self.training_df.iat[int(index),0])
        image2_path=os.path.join(self.training_df.iat[int(index),1])
        
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        
        img0 = preprocess_image(img0)
        img1 = preprocess_image(img1)
        
        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        label = torch.from_numpy(np.array([int(self.training_df.iat[int(index), 2])], dtype=np.float32))
        
        return img0, img1 , label
    
    def __len__(self):
        return len(self.training_df)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights
        attention_scores = self.conv(x)
        attention_weights = self.sigmoid(attention_scores)

        # Apply attention to the input feature map
        attended_features = x * attention_weights

        return attended_features


class SiameseResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super(SiameseResNet, self).__init__()
        self.baseModel = models.resnet18(pretrained=pretrained)

        # Experiment with different spatial sizes based on the image resolution and signature complexity
        self.attention1 = SpatialAttention(in_channels=64)  # Spatial attention for layer 1
        self.attention2 = SpatialAttention(in_channels=128)  # Spatial attention for layer 2

        self.baseModel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        out = self.baseModel.conv1(x)
        out = self.baseModel.bn1(out)
        out = self.baseModel.relu(out)
        out = self.baseModel.maxpool(out)

        out = self.attention1(self.baseModel.layer1(out))  # Applying spatial attention to layer 1
        out = self.attention2(self.baseModel.layer2(out))  # Applying spatial attention to layer 2
        out = self.baseModel.layer3(out)  # No attention for layer 3
        out = self.baseModel.layer4(out)  # No attention for layer 4

        out = self.baseModel.avgpool(out)
        out = torch.flatten(out, 1)
        return out

class BaselineSiameseResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super(BaselineSiameseResNet, self).__init__()
        self.baseModel = models.resnet18(pretrained=pretrained)
        self.baseModel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        out = self.baseModel.conv1(x)
        out = self.baseModel.bn1(out)
        out = self.baseModel.relu(out)
        out = self.baseModel.maxpool(out)
        out = self.baseModel.layer1(out)
        out = self.baseModel.layer2(out)
        out = self.baseModel.layer3(out)
        out = self.baseModel.layer4(out)
        out = self.baseModel.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_anchor_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_anchor_negative = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(distance_anchor_positive - distance_anchor_negative + self.margin, min=0.0)
        return loss.mean()