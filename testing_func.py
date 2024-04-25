import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train_poseresnet import SimplePoseResNet, MLPClassifier, DupletDatasetCEDAR

def test_training_step():
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # Initialize the dataset and dataloader for testing
    test_dataset = DupletDatasetCEDAR(base_dir='/home/jamesemi/Desktop/james/adl/ViTPose_pytorch/datasets/CEDAR/train', signers=['1'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)  # Small batch size for testing

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model components
    model = SimplePoseResNet(num_joints=40).to(device)
    mlp_model = MLPClassifier(input_dim=5120, hidden_dim=512, output_dim=1).to(device)

    # Define the criterion and optimizer
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(mlp_model.parameters()), lr=0.01)

    # Test forward and backward pass
    model.train()
    mlp_model.train()
    for images, targets in test_loader:
        images = [image.to(device) for image in images]
        targets = targets.to(device)

        # Forward pass
        output1 = model(images[0])
        output2 = model(images[1])
        output = torch.cat((output1, output2), dim=1)
        final_output = mlp_model(output)

        # Loss computation
        loss = criterion(final_output, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Test loss: {loss.item()}')
        break  # Run the test for only one batch

if __name__ == "__main__":
    test_training_step()
