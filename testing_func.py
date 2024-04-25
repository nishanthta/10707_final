import torch
from train_poseresnet import SimplePoseResNet  # Import your model class here
from train_poseresnet import DupletDatasetCEDAR  # Import your dataset class here
from torchvision import transforms

def test_training_step():
    # Load a small portion of the dataset for testing
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization for EfficientNet
    ])
    test_loader = torch.utils.data.DataLoader(
        DupletDatasetCEDAR(train=False, transform=transform),  # Assuming you have transforms
        batch_size=10,  # Small batch size for quick testing
        shuffle=True
    )

    # Initialize the model
    model = SimplePoseResNet()
    model.train()

    # Optionally, move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define a loss function
    criterion = torch.nn.CrossEntropyLoss()  # Example loss function, change as needed

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run a single mini-batch to test
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check for any non-finite numbers in outputs
        if not torch.isfinite(loss):
            print("Non-finite loss encountered.")
        else:
            print("Loss computed successfully: ", loss.item())
        break  # Only one batch for testing

if __name__ == "__main__":
    test_training_step()
