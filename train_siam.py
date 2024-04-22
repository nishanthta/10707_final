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
        
        output1_flat = output1.view(output1.size(0), -1)
        output2_flat = output2.view(output2.size(0), -1)
        euclidean_distance = F.pairwise_distance(output1_flat, output2_flat, keepdim=True)
        # Contrastive loss
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss


def train_model(model, mlp_model, train_loader, val_loader, device, patience):
    # criterion = JointsMSELoss().to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_contrastive = ContrastiveLoss()
    optimizer = AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()
    epochs_without_improvement, min_loss = 0, 1e8

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
                
                # # Flatten or pool the outputs to reduce them to 2D
                # output1 = F.adaptive_avg_pool2d(output1, (1, 1)).view(output1.size(0), -1)
                # output2 = F.adaptive_avg_pool2d(output2, (1, 1)).view(output2.size(0), -1)
                output = torch.reshape(torch.cat((output1, output2), dim=1), (batch_size, -1))
                
                # Verify outputs are correct for pairwise distance calculation
                # print("Output1 shape:", output1.shape)  # Should be [batch_size, feature_length]
                # print("Output2 shape:", output2.shape)
                final_output = mlp_model(output)
                loss = criterion_bce(final_output, targets) + 1e-4*criterion_contrastive(output1, output2, targets)

            scaler.scale(loss).backward()

            # Print and analyze gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f'Parameter: {name}, Gradient Norm: {param.grad.norm().item()}')


            clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
         
        # scheduler.step(val_loss)

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
                loss = criterion_bce(final_output, targets) + criterion_contrastive(output1, output2, targets)
                val_loss += loss.item()
                
            avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}, Val Loss : {avg_val_loss}')
        scheduler.step(avg_val_loss)  # Pass the validation loss to the scheduler

        if avg_val_loss < min_loss:
            epochs_without_improvement = 0
            torch.save(model, '/data-fast/james/adl/chkpts/finetuned_coco_b_cedar_vitpose_ckpt.pth')
            torch.save(mlp_model, '/data-fast/james/adl/chkpts/finetuned_coco_b_cedar_mlp_ckpt.pth')
            # torch.save(model, '/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/chkpts/scratch_cedar_vitpose_ckpt.pth')
            # torch.save(mlp_model, '/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/chkpts/scratch_cedar_mlp_ckpt.pth')
        
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping activated')
                break

        


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
    model = ViTPose().to(device)
    mlp_model = MLPClassifier(input_dim=34*64*64, hidden_dim=512, output_dim=1).to(device) #update this
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    # Path to the pretrained model checkpoint
    checkpoint_path = '/data-fast/james/adl/chkpts/vitpose-b-multi-coco.pth'
    # checkpoint_path = '/ViTPose_pytorch/chkpts/xyz.pth'
    # checkpoint_path = None
    
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
    # train_signers, val_signers = train_signers[:5], val_signers[:2]

    train_dataset = DupletDatasetCEDAR(dirs['train'], transform=transform, signers=train_signers)
    val_dataset = DupletDatasetCEDAR(dirs['val'], transform=transform, signers=val_signers)
    test_dataset = DupletDatasetCEDAR(dirs['test'], transform=transform, signers=test_signers)
    
    # Load datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    train_model(model, mlp_model, train_loader, val_loader, device, patience=10)
    

if __name__ == '__main__':
    main()
