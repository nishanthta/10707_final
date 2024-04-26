from collections import defaultdict
import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm
from models.backbone.vit import ViT
from models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from models.losses import JointsMSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from itertools import combinations, product
from torch.utils.data import Dataset
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50
from sklearn.metrics import roc_auc_score

def seed_everything(seed_value = 42):
    random.seed(seed_value)

    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True  # ensure deterministic behavior for cuDNN
        torch.backends.cudnn.benchmark = False  # it can be beneficial to turn this off as it can introduce randomness

    print(f'Set all random seeds to {seed_value}.')


class DupletDatasetCEDAR(Dataset):
    def __init__(self, base_dir, signers, transform=None, limit=None):
        """
        Initialize the dataset with the directory of images and transforms.
        base_dir: The directory that contains subdirectories of original and forgery images.
        transform: Transformations to apply to each image.
        """
        self.transform = transform
        self.signers = signers
        self.pairs, self.labels = self._create_pairs(base_dir, limit)
        self.base_dir = base_dir

    def _create_pairs(self, base_dir, limit):
        pairs = []
        labels = []
        for signer in self.signers:
            subdir = os.path.join(base_dir, signer)
            originals = [os.path.join(subdir, f) for f in os.listdir(subdir) if 'original' in f]
            forgeries = [os.path.join(subdir, f) for f in os.listdir(subdir) if 'forgeries' in f]

            original_pairs = list(combinations(originals, 2))
            forgery_pairs = list(product(originals, forgeries))
            # print('original pairs length', len(original_pairs))
            # print('forgery pairs length', len(forgery_pairs))

            if limit:
                original_pairs = original_pairs[:limit]
                forgery_pairs = forgery_pairs[:limit]

            for pair in original_pairs:
                pairs.append(pair)
                labels.append(0)
            for pair in forgery_pairs:
                pairs.append(pair)
                labels.append(1)

        return pairs, labels
    
    def __getitem__(self, index):
        img1_path, img2_path = self.pairs[index]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor([self.labels[index]], dtype=torch.float32)
        return [img1, img2], label

    def __len__(self):
        return len(self.pairs)

# Example transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization for EfficientNet
])

class SimplePoseResNet(nn.Module):
    def __init__(self, num_joints):
        super(SimplePoseResNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Removing the final fully connected layer
        self.resnet.avgpool = nn.Identity()

        # Additional layers for pose estimation
        self.pose_layers = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_joints, kernel_size=1)  # Output channels = num_joints
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # This is where you usually have spatial dimensions

        # Apply pose estimation layers
        x = self.pose_layers(x)  # This expects a spatial dimension
        x = x.view(x.size(0), -1)  # Flatten the output properly before passing to MLP
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print(f"Input shape to MLPClassifier: {x.shape}")  # Debugging line to check input shape
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance between output1 and output2
        
        output1_flat = output1.view(output1.size(0), -1)
        output2_flat = output2.view(output2.size(0), -1)
        euclidean_distance = F.pairwise_distance(output1_flat, output2_flat, keepdim=True)
        # Contrastive loss
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss


def train_model(model, mlp_model, train_loader, val_loader, device, patience, lr, num_epochs, initial_epochs=5):
    train_losses = []
    val_losses = []
    # criterion = JointsMSELoss().to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_contrastive = ContrastiveLoss()
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    scaler = GradScaler()
    epochs_without_improvement, min_loss = 0, 1e8

     # Initial settings for loss weighting
    contrastive_weight = 1.0
    bce_weight = 0.0 # Start with a smaller weight for BCE


    model.train()
    mlp_model.train()
    for epoch in range(200):  # Number of epochs
        total_loss, val_loss = 0, 0
        for images, targets in tqdm(train_loader):

            images = [image.to(device) for image in images]
            targets = targets.to(device)
            batch_size = targets.size(0)            

            optimizer.zero_grad()

            with autocast():
                output1 = model(images[0])
                output2 = model(images[1])
                output = torch.reshape(torch.cat((output1, output2), dim=1), (targets.size(0), -1))
                final_output = mlp_model(output)
                classification_loss = nn.BCEWithLogitsLoss()(final_output, targets)
                contrastive_loss = ContrastiveLoss()(output1, output2, targets)
                loss = contrastive_loss + 1e-4 * classification_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # # Print and analyze gradients
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Parameter: {name}, Gradient Norm: {param.grad.norm().item()}')


            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        model.eval()
        mlp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = [image.to(device) for image in images]
                targets = targets.to(device)
                output1 = model(images[0])
                output2 = model(images[1])
                output = torch.reshape(torch.cat((output1, output2), dim=1), (targets.size(0), -1))
                
                final_output = mlp_model(output)
                output1 = output1.view(output1.size(0), -1)
                output2 = output2.view(output2.size(0), -1)
                contrastive_loss = criterion_contrastive(output1, output2, targets)
                classification_loss = criterion_bce(final_output, targets)
                loss = contrastive_weight * contrastive_loss + bce_weight * classification_loss
                # loss = criterion_bce(final_output, targets) + criterion_contrastive(output1, output2, targets)
                val_loss += loss.item()
                
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}, Val Loss : {avg_val_loss}')
        scheduler.step(avg_val_loss)  # Pass the validation loss to the scheduler

        # Dynamic checkpoint saving
        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_filename = f'model_epoch{epoch+1}_lr{lr}_valloss{avg_val_loss:.4f}_{timestamp}.pth'
            torch.save(model.state_dict(), checkpoint_filename)
            torch.save(mlp_model.state_dict(), checkpoint_filename.replace("model", "mlp_model"))
        
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping activated')
                break

        # Compute average losses
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Store losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
    return train_losses, val_losses


def save_model(model, epoch, loss, descriptor="ViTPose"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{descriptor}_Epoch{epoch}_Loss{loss:.4f}_{timestamp}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Saved model as {filename}")

# def validate_model(model, mlp_model, dataloader, device, criterion):
#     model.eval()
#     mlp_model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, targets in dataloader:
#             images = [image.to(device) for image in images]
#             targets = targets.to(device)
#             output1 = model(images[0])
#             output2 = model(images[1])
#             output = torch.reshape(torch.cat((output1, output2), dim=1), (targets.size(0), -1))
#             final_output = mlp_model(output)
#             loss = criterion(final_output, targets)
#             val_loss += loss.item()
            
#         avg_val_loss = val_loss / len(dataloader)
        
#         return avg_val_loss
def test_model(model, mlp_model, test_loader, device):
    model.eval()
    mlp_model.eval()
    pos_distances, neg_distances = [], []  # Lists to store distances for positive and negative pairs
    distances = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = [image.to(device) for image in images]
            targets = targets.to(device)
            
            output1 = model(images[0])
            output2 = model(images[1])
            output = torch.cat((output1, output2), dim=1)
            final_output = mlp_model(output)
            
            euclidean_distance = F.pairwise_distance(output1, output2).item()
            distances.append(euclidean_distance)
            labels.append(targets.item())

            if targets.item() == 1:
                pos_distances.append(euclidean_distance)
            else:
                neg_distances.append(euclidean_distance)

    # Calculate AUC score assuming that a smaller distance indicates a match
    auc_score = roc_auc_score(labels, 1 - np.array(distances))  # Convert distances to similarities
    print(f"AUC Score: {auc_score}")

    # Create the violin plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=[pos_distances, neg_distances])
    plt.xticks([0, 1], ['Positive Pairs', 'Negative Pairs'])
    plt.title('Distribution of Distances Between Pairs')
    plt.ylabel('Euclidean Distance')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_filename = f'violinplot_{timestamp}.png'
    plt.savefig(plot_filename)

    plt.show()

    return pos_distances, neg_distances, auc_score

def plot_losses(train_losses, val_losses, lr, num_epochs):
    plt.figure(figsize=(10, 5))
    sns.lineplot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    sns.lineplot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss: LR={lr}, Epochs={num_epochs}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_filename = f'training_validation_loss_lr{lr}_epochs{num_epochs}_{timestamp}.png'
    plt.savefig(plot_filename)
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePoseResNet(num_joints=40).to(device)
    mlp_model = MLPClassifier(input_dim=5120, hidden_dim=512, output_dim=1).to(device) #update this
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    
    
    print("Checkpoint file not found. Training from scratch.")
 
    
    # Load dataset
    # df = pd.read_csv('path_to_cedar_dataset.csv')  # Update path as necessary
    # dataset = DupletDataset(dataframe=df)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize dataset
    # dataset = DupletDatasetCEDAR(base_dir='OSV_2D/CEDAR', transform=transform)
    dirs = defaultdict(list)
    dirs['train'] = '/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/datasets/CEDAR/train'
    dirs['val'] = '/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/datasets/CEDAR/val'
    dirs['test'] = '/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/datasets/CEDAR/test'
    
    train_signers = [name for name in os.listdir(dirs['train']) if os.path.isdir(os.path.join(dirs['train'], name))]
    val_signers = [name for name in os.listdir(dirs['val']) if os.path.isdir(os.path.join(dirs['val'], name))]
    test_signers = [name for name in os.listdir(dirs['test']) if os.path.isdir(os.path.join(dirs['test'], name))]
    
    #COMMENT OUT THE FOLLOWING TEST
    # train_signers, val_signers = train_signers[:5], val_signers[:2]

    # limit = 50
    # train_dataset = DupletDatasetCEDAR(dirs['train'], transform=transform, signers=train_signers, limit=limit)
    # val_dataset = DupletDatasetCEDAR(dirs['val'], transform=transform, signers=val_signers, limit=limit)
    # test_dataset = DupletDatasetCEDAR(dirs['test'], transform=transform, signers=test_signers, limit=limit)
    train_dataset = DupletDatasetCEDAR(dirs['train'], transform=transform, signers=train_signers)
    val_dataset = DupletDatasetCEDAR(dirs['val'], transform=transform, signers=val_signers)
    test_dataset = DupletDatasetCEDAR(dirs['test'], transform=transform, signers=test_signers)
    
    # Load datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # train_losses, val_losses = train_model(model, mlp_model, train_loader, val_loader, device, patience=10)
    
    # Parameters for training
    lr = 0.001
    num_epochs = 50
    patience = 10

    # # Train and validate
    # train_losses, val_losses = train_model(model, mlp_model, train_loader, val_loader, device, patience, lr, num_epochs)
    # plot_losses(train_losses, val_losses, lr, num_epochs)


    #uncomment below for testing
    
    # Path to the finetuned model checkpoint
    checkpoint_path = '/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/model_epoch1_lr0.001_valloss0.1695_20240425-195051.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # print("Available keys in checkpoint:", checkpoint.keys())
    # state_dict = checkpoint['model_state_dict']
    final_model = SimplePoseResNet(num_joints=40).to(device)
    # final_model.load_state_dict(checkpoint.get('state_dict', checkpoint))  # Fallback to checkpoint if 'state_dict' key is not found
    final_model.load_state_dict(checkpoint)
    print("Final model loaded from given checkpoint file:", checkpoint_path)
    test_model(final_model, mlp_model, test_loader, device)
    # test_model_auc(final_model, mlp_model, test_loader, device)


if __name__ == '__main__':
    main()

    
