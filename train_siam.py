from collections import defaultdict
import os
import random
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
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
from torch.nn.utils import clip_grad_norm_
import datetime
import numpy as np
import seaborn as sns
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
    def __init__(self, base_dir, signers, transform=None):
        """
        Initialize the dataset with the directory of images and transforms.
        base_dir: The directory that contains subdirectories of original and forgery images.
        transform: Transformations to apply to each image.
        """
        self.transform = transform
        self.signers = signers
        self.pairs, self.labels = self._create_pairs(base_dir)
        self.base_dir = base_dir

    def _create_pairs(self, base_dir):
        pairs = []
        labels = []
        # Walk through the directory
        for signer in self.signers:
            subdir = os.path.join(base_dir, signer)
            originals = [os.path.join(subdir, f) for f in os.listdir(subdir) if 'original' in f]
            forgeries = [os.path.join(subdir, f) for f in os.listdir(subdir) if 'forgeries' in f]

            original_pairs = list(combinations(originals, 2))
            for pair in original_pairs:
                pairs.append(pair)
                labels.append(0)

            forgery_pairs = list(product(originals, forgeries))
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
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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


def plot_losses(train_losses, val_losses):
    sns.set(style="whitegrid")  # Set the style of the plot using seaborn
    plt.figure(figsize=(10, 5))  # Set the size of the plot
    plt.title('Training and Validation Loss')  # Set the title of the plot

    # Plot training and validation loss
    sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses, label='Training Loss', linewidth=2.5)
    sns.lineplot(x=range(1, len(val_losses)+1), y=val_losses, label='Validation Loss', linewidth=2.5)

    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.legend()  # Add a legend
    plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
    
    plt.savefig('training_validation_loss.png')  # Save the plot to a file
    plt.show()  # Display the plot

def train_model(model, mlp_model, train_loader, val_loader, device, patience, lr, num_epochs):
    criterion_contrastive = ContrastiveLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    scaler = GradScaler()

    epochs_without_improvement, min_loss = 0, float('inf')
    train_losses, val_losses = [], []

    model.train()
    mlp_model.train()
    for epoch in range(num_epochs):  # Adjust the number of epochs if needed
        total_loss, val_loss = 0, 0
        for images, targets in tqdm(train_loader):
            images = [image.to(device) for image in images]
            targets = targets.to(device)

            optimizer.zero_grad()
            with autocast():
                output1 = model(images[0])
                output2 = model(images[1])
                loss = criterion_contrastive(output1.view(output1.shape[0], -1), output2.view(output2.shape[0], -1), targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss = 0
        model.eval()
        mlp_model.eval()
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = [image.to(device) for image in images]
                targets = targets.to(device)
                output1 = model(images[0])
                output2 = model(images[1])
                loss = criterion_contrastive(output1.view(output1.shape[0], -1), output2.view(output2.shape[0], -1), targets)
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss : {avg_val_loss}')

        if avg_val_loss < min_loss:
            min_loss = avg_val_loss
            epochs_without_improvement = 0
            # torch.save(model.state_dict(), f'vitpose_checkpoint_epoch_{epoch+1}.pth')
            # torch.save(mlp_model.state_dict(), f'mlp_checkpoint_epoch_{epoch+1}.pth')\
            checkpoint_filename = f'vitpose_lr{lr}_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pth'
            torch.save(model.state_dict(), checkpoint_filename)
            torch.save(mlp_model.state_dict(), checkpoint_filename.replace("vitpose", "mlp"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping activated')
                break

    return train_losses, val_losses

def plot_losses(train_losses, val_losses, lr, num_epochs):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses, label='Training Loss', linewidth=2.5)
    sns.lineplot(x=range(1, len(val_losses)+1), y=val_losses, label='Validation Loss', linewidth=2.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (LR={lr}, Epochs={num_epochs})')
    plt.legend()
    plot_filename = f'training_validation_loss_lr{lr}_epochs{num_epochs}.png'
    plt.savefig(plot_filename)
    plt.show()


def test_model(model, mlp_model, test_loader, device):
    model.eval()
    mlp_model.eval()
    pos_distances, neg_distances = [], [] 
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = [image.to(device) for image in images]
            targets = targets.to(device)
            output1 = model(images[0])
            output2 = model(images[1])
            # output = torch.reshape(torch.cat((output1, output2), dim=1), (targets.size(0), -1))
            # final_output = mlp_model(output)
            distance = F.pairwise_distance(output1.view(output1.shape[0], -1), output2.view(output1.shape[0], -1), keepdim=True)
            if targets.item() == 1:
                pos_distances.append(distance.item())
            else:
                neg_distances.append(distance.item())

    data = [pos_distances, neg_distances]

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=data)
    plt.xticks([0, 1], ['Forgery', 'Original'])
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/figs/violin_plot_{timestamp}.pth"
    plt.savefig(filename, dpi=300)
    return pos_distances, neg_distances

def test_model_auc(model, mlp_model, test_loader, device):
    model.eval()
    mlp_model.eval()
    distances = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = [image.to(device) for image in images]
            targets = targets.to(device)
            output1 = model(images[0])
            output2 = model(images[1])
            distance = F.pairwise_distance(output1.view(output1.shape[0], -1), output2.view(output2.shape[0], -1)).squeeze().cpu().numpy()
            distances.extend(distance)
            labels.extend(targets.cpu().numpy())

    # Convert distances to a similarity score or keep as is depending on your approach
    # Here assuming that lower distance means higher similarity (common in contrastive learning)
    similarities = 1 / (1 + np.array(distances))  # Example conversion, adjust based on your specific needs
    auc_score = roc_auc_score(labels, similarities)  # Calculate AUC based on the similarity scores

    print(f"AUC Score: {auc_score}")
    return auc_score

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

    
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = SiameseCapsNet(image_channels=1, primary_capsules=32, primary_dim=8, num_classes=10, out_dim=16, num_routing=3).to(device)
    model = ViTPose().to(device)
    mlp_model = MLPClassifier(input_dim=34*64*64, hidden_dim=512, output_dim=1).to(device) #update this
    # mlp_model = MLPClassifier(input_dim=34*64*64, hidden_dim=512, output_dim=1).to(device) #update this
    # model = torch.load('/home/nthumbav/Downloads/ViTPose_pytorch/vitpose_ckpt.pth')

    # Path to the pretrained model checkpoint
    checkpoint_path = '/data-fast/james/adl/chkpts/vitpose-b-multi-coco.pth'


    #fake path
    # checkpoint_path = '/ViTPose_pytorch/chkpts/xyz.pth'
    # checkpoint_path = None

    num_epochs = 100
    learning_rate = 1e-3
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['state_dict']

        # Adjust the first convolutional layer weights
        # Average the RGB channels weights to fit grayscale input
        first_conv_weight = state_dict['backbone.patch_embed.proj.weight']
        first_conv_weight_mean = first_conv_weight.mean(dim=1, keepdim=True)
        state_dict['backbone.patch_embed.proj.weight'] = first_conv_weight_mean

        # Resize positional embeddings if necessary
        pos_embed = state_dict['backbone.pos_embed']
        current_pos_embed = model.backbone.pos_embed
        if pos_embed.shape != current_pos_embed.shape:
            # Interpolate positional embeddings
            new_pos_embed = F.interpolate(pos_embed.permute(0, 2, 1), size=current_pos_embed.shape[-2], mode='linear', align_corners=True)
            state_dict['backbone.pos_embed'] = new_pos_embed.permute(0, 2, 1)

        # Load the adjusted state dict
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained model loaded successfully with adjustments.")
    else:
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
    #train_signers, val_signers = train_signers[:2], val_signers[:1]

    train_dataset = DupletDatasetCEDAR(dirs['train'], transform=transform, signers=train_signers)
    val_dataset = DupletDatasetCEDAR(dirs['val'], transform=transform, signers=val_signers)
    test_dataset = DupletDatasetCEDAR(dirs['test'], transform=transform, signers=test_signers)
    
    # Load datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    #uncomment below for testing
    
    # Path to the finetuned model checkpoint
    checkpoint_path = '/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/vitpose_lr0.001_epoch3_valloss0.0167.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # print("Available keys in checkpoint:", checkpoint.keys())
    # state_dict = checkpoint['model_state_dict']
    final_model = ViTPose().to(device)
    final_model.load_state_dict(checkpoint)
    print("Final model loaded from given checkpoint file:", checkpoint_path)
    test_model(final_model, mlp_model, test_loader, device)
    test_model_auc(final_model, mlp_model, test_loader, device)

    #uncomment below for training run
    # train_losses, val_losses = train_model(model, mlp_model, train_loader, val_loader, device, patience=5, lr=learning_rate, num_epochs=num_epochs)
    # plot_losses(train_losses, val_losses, learning_rate, num_epochs)


    
if __name__ == '__main__':
    main()
