from collections import defaultdict
import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from models.backbone.vit import ViT
from models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
                for _ in range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            logits = torch.zeros(*priors.size(), device=x.device)
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                if i < self.num_iterations - 1:
                    logits = logits + (priors * outputs).sum(dim=-1, keepdim=True)
            return outputs.squeeze(3)
        else:
            outputs = [capsule(x) for capsule in self.capsules]
            outputs = torch.stack(outputs, dim=1)
            outputs = outputs.view(x.size(0), self.num_capsules, -1)
            return self.squash(outputs)

class CapsNet(nn.Module):
    def __init__(self, image_channels, primary_capsules, primary_dim, num_classes, out_dim, num_routing):
        super(CapsNet, self).__init__()
        self.conv_layer = nn.Conv2d(image_channels, 256, kernel_size=9, stride=1)
        # Adjust primary capsules to properly handle outputs
        self.primary_capsules = CapsuleLayer(num_capsules=primary_capsules, num_route_nodes=-1, in_channels=256,
                                             out_channels=primary_dim, kernel_size=9, stride=2)
        # Here, consider adjusting `num_route_nodes` for digit_capsules based on primary_capsules output
        self.digit_capsules = CapsuleLayer(num_capsules=num_classes, num_route_nodes=32*1152, # Example, calculate the actual number based on the output from primary_capsules
                                           in_channels=primary_dim, out_channels=out_dim, num_iterations=num_routing)

    def forward(self, x):
        x = F.relu(self.conv_layer(x), inplace=True)
        x = self.primary_capsules(x)
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.digit_capsules(x)
        lengths = x.norm(dim=-1)
        return x, lengths

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class ViTPose(nn.Module):
    def __init__(self):
        super(ViTPose, self).__init__()
        # Vision Transformer Backbone
        self.backbone = ViT(
            img_size=256, 
            patch_size=16, 
            in_chans=1, 
            embed_dim=768, 
            depth=12, 
            num_heads=12, 
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1
        )
        # Initialize the keypoint head with 'extra' options
        self.keypoint_head = TopdownHeatmapSimpleHead(
            in_channels=768,
            out_channels=17,  # Assuming 17 keypoints
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra={'final_conv_kernel': 1}  # Setting the final conv kernel size
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.keypoint_head(x)
        return x
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance between output1 and output2
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # Contrastive loss
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        if torch.isnan(loss):
            print('Nan')
        return loss

class PointwiseContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(PointwiseContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Ensure label is broadcastable to the output size
        label = label.view(-1, 1, 1, 1).expand_as(output1)
        
        # Calculate pairwise cosine similarity
        cosine_sim = F.cosine_similarity(output1, output2, dim=1, eps=1e-6)

        # Calculate pairwise distance
        euclidean_dist = F.pairwise_distance(output1.view(output1.size(0), -1),
                                             output2.view(output2.size(0), -1),
                                             keepdim=True).view_as(output1)

        # Calculate loss for positive and negative pairs
        positive_loss = 1 - cosine_sim  # Maximize cosine similarity for positive pairs
        negative_loss = F.relu(euclidean_dist - self.margin)  # Ensure distance is beyond a margin for negative pairs

        # Apply labels to determine which loss to use for each pixel
        loss = torch.where(label == 1, positive_loss, negative_loss)

        return loss.mean()

class SiameseCapsNet(nn.Module):
    def __init__(self, image_channels, primary_capsules, primary_dim, num_classes, out_dim, num_routing):
        super(SiameseCapsNet, self).__init__()
        self.capsnet = CapsNet(image_channels, primary_capsules, primary_dim, num_classes, out_dim, num_routing)

    def forward(self, x1, x2):
        output1, _ = self.capsnet(x1)
        output2, _ = self.capsnet(x2)
        return output1, output2

def contrastive_loss(output1, output2, label, margin=1.0):
    # Euclidean distance between output1 and output2
    euclidean_distance = F.pairwise_distance(output1, output2)
    # Contrastive loss
    loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

