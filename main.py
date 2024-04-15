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
from model_data_defs import *

from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_model(model, loaders, n_epochs, optimizer, loss_fn):
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    counter = 0
    
    print("------------------------Training--------------------------")
    for epoch in range(1, n_epochs + 1):
        t0 = datetime.now()
        print(f"Beginning Epoch {epoch}/{n_epochs}...")
        train_loss, val_loss = [], []
        
        model.train()
        for i, data in tqdm(enumerate(loaders['train'], 0)):
            anchor, positive, negative = data
            anchor = anchor.to(device=device)
            positive = positive.to(device=device)
            negative = negative.to(device=device)
            
            optimizer.zero_grad()
            anchor_embeddings = model(anchor)
            positive_embeddings = model(positive)
            negative_embeddings = model(negative)
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        model.eval()
        for i, data in tqdm(enumerate(loaders['val'], 0)):
            anchor, positive, negative = data
            anchor = anchor.to(device=device)
            positive = positive.to(device=device)
            negative = negative.to(device=device)
            
            anchor_embeddings = model(anchor)
            positive_embeddings = model(positive)
            negative_embeddings = model(negative)
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            val_loss.append(loss.item())
        
        dt = datetime.now() - t0
        print('\nEpoch: {}\tTrain Loss: {:.4f}\tVal Loss: {:.4f}\tDuration: {}'.format(epoch, np.mean(train_loss), np.mean(val_loss), dt))
        
        history['train_loss'].append(np.mean(train_loss))
        history['val_loss'].append(np.mean(val_loss))
        
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            torch.save(model.state_dict(), f'/home/nthumbav/Downloads/10707_final/checkpoints/{DATASET}_{SIGNATURE}_{MODEL_TYPE}.pth')
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.4f}")
                return history
    
    return history




root_dir = '/home/nthumbav/Downloads/OSV_2D/'

train_path, val_path, test_path = os.path.join(root_dir, DATASET, 'train'), os.path.join(root_dir, DATASET, 'val'), os.path.join(root_dir, DATASET, 'test')
train_df, val_df, test_df = triplet_dataset_preparation(train_path), triplet_dataset_preparation(val_path), triplet_dataset_preparation(test_path)

transformation = transforms.Compose([
    transforms.Resize((200,300)),
    transforms.RandomRotation((-5,10)),
    transforms.ToTensor(),
])

from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms


train_dataset, val_dataset, test_dataset =  TripletDataset(train_df, transform = transformation), TripletDataset(val_df, transform = transformation), TripletDataset(test_df, transform = transformation)

loaders = defaultdict(list)
loaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE)
loaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size= BATCH_SIZE)
loaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size= BATCH_SIZE)

# from torch.utils.data import DataLoader, SubsetRandomSampler
# num_samples_to_show = 5

# # Get random indices to select samples from the DataLoader
# num_samples = len(loaders['train'].dataset)
# random_indices = np.random.choice(num_samples, num_samples_to_show, replace=False)

# # Create a SubsetRandomSampler using the random indices
# sampler = SubsetRandomSampler(random_indices)

# # Create a new DataLoader using the SubsetRandomSampler
# random_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler)

# # Show the random samples
# show_triplets_from_dataloader(random_loader, num_samples_to_show)
# Create an instance of SiameseResnet with the ResNet model and embedding size

# from torchsummary import summary

if MODEL_TYPE == 'SiameseResNet':
    model = SiameseResNet()

elif MODEL_TYPE == 'BaselineSiameseResNet':
    model = BaselineSiameseResNet()

model = nn.DataParallel(model).to(device)
# # summary(model, (1,200,300))
triplet_loss = TripletLoss(TRIPLET_LOSS_MARGIN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
history = train_model(model, loaders, NUM_EPOCHS, optimizer, triplet_loss)

