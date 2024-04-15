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
from scipy.spatial.distance import cdist

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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict

def calculate_metrics(model, loader, loss_fn):
    model.eval()
    test_loss = []
    
    # Lists to store embedding differences
    positive_diff_list = []
    negative_diff_list = []
    
    # Lists to store ground truth and predicted labels
    ground_truth = []
    predictions = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader, 0)):
            anchor, positive, negative = data
            anchor = anchor.to(device=device)
            positive = positive.to(device=device)
            negative = negative.to(device=device)
            
            anchor_embeddings = model(anchor)
            positive_embeddings = model(positive)
            negative_embeddings = model(negative)
            
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            test_loss.append(loss.item())
            
            # Calculate embedding differences
            positive_diff = torch.norm(positive_embeddings - anchor_embeddings, dim=1).cpu().numpy()
            negative_diff = torch.norm(negative_embeddings - anchor_embeddings, dim=1).cpu().numpy()
            positive_diff_list.extend(positive_diff)
            negative_diff_list.extend(negative_diff)
            
            # Predict labels based on embedding differences
            labels = (positive_diff < negative_diff).astype(int)
            ground_truth.extend([1] * len(labels))
            predictions.extend(labels.tolist())
    
    test_loss = np.mean(test_loss)
    
    # Calculate average embedding differences
    positive_diff_mean = np.mean(positive_diff_list)
    positive_diff_std = np.std(positive_diff_list)
    negative_diff_mean = np.mean(negative_diff_list)
    negative_diff_std = np.std(negative_diff_list)
    
    # Calculate classification accuracy
    accuracy = np.mean(np.array(ground_truth) == np.array(predictions))
    cm = confusion_matrix(ground_truth, predictions)

    # Create a defaultdict to store the results
    metrics = defaultdict(list)
    metrics['test_loss'].append(test_loss)
    metrics['positive_diff_mean'].append(positive_diff_mean)
    metrics['positive_diff_std'].append(positive_diff_std)
    metrics['negative_diff_mean'].append(negative_diff_mean)
    metrics['negative_diff_std'].append(negative_diff_std)
    metrics['confusion_matrix'].append(cm)
    metrics['accuracy'].append(accuracy)
    
    return metrics

def test_model(model, loader, loss_fn):
    model.eval()
    test_loss = []
    
    # Lists to store embedding differences
    positive_diff_list = []
    negative_diff_list = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader, 0)):
            anchor, positive, negative = data
            anchor = anchor.to(device=device)
            positive = positive.to(device=device)
            negative = negative.to(device=device)
            
            anchor_embeddings = model(anchor)
            positive_embeddings = model(positive)
            negative_embeddings = model(negative)
            
            loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
            test_loss.append(loss.item())
            
            # Calculate embedding differences and store them in lists
            positive_diff = positive_embeddings - anchor_embeddings
            negative_diff = negative_embeddings - anchor_embeddings
            positive_diff_list.append(positive_diff.cpu().numpy())
            negative_diff_list.append(negative_diff.cpu().numpy())
    
    test_loss = np.mean(test_loss)
    print(f'Test Loss: {test_loss:.4f}')
    
    # Concatenate embedding differences from all batches
    positive_diff = np.concatenate(positive_diff_list, axis=0)
    negative_diff = np.concatenate(negative_diff_list, axis=0)
    
    # Perform t-SNE dimensionality reduction on embedding differences
    tsne_embeddings = TSNE(n_components=2, random_state=42).fit_transform(np.concatenate((positive_diff, negative_diff), axis=0))
    
    # Create a scatter plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], hue=['Anchor-Positive']*len(positive_diff) + ['Anchor-Negative']*len(negative_diff), palette='deep', alpha=0.7)
    # plt.title('t-SNE Visualization of Embedding Differences')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Embedding Difference')
    plt.tight_layout()
    plt.savefig('/home/nthumbav/Downloads/10707_final/scratch/embeddings_tsne.png', dpi=300)
    
    return test_loss


root_dir = '/home/nthumbav/Downloads/OSV_2D/'
test_path = os.path.join(root_dir, DATASET, 'test')
test_df = triplet_dataset_preparation(test_path)

transformation = transforms.Compose([
    transforms.Resize((200,300)),
    # transforms.RandomRotation((-5,10)),
    transforms.ToTensor(),
])



test_dataset =  TripletDataset(test_df, transform = transformation)

loaders = defaultdict(list)
loaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size= BATCH_SIZE)

triplet_loss = TripletLoss(TRIPLET_LOSS_MARGIN).to(device)

if MODEL_TYPE == 'SiameseResNet':
    model = SiameseResNet()

elif MODEL_TYPE == 'BaselineSiameseResNet':
    model = BaselineSiameseResNet()

model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('/home/nthumbav/Downloads/10707_final/checkpoints/CEDAR_first_run_BaselineSiameseResNet.pth'))


# metrics = test_model(model, loaders['test'], triplet_loss)
metrics = calculate_metrics(model, loaders['test'], triplet_loss)

for key in list(metrics.keys()):
    print(key, ' ', metrics[key])

pass
