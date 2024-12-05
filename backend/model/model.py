import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import os

# Automatically select the best available device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for the training and testing sets (with data augmentation for training)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Larger input size for pre-trained models
    transforms.RandomHorizontalFlip(),  # Random horizontal flipping
    transforms.RandomRotation(15),  # Random rotation up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),  # Larger input size for pre-trained models
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the entire dataset (replace 'path_to_data' with the actual dataset path)
dataset = ImageFolder(root='Z:/food-101/images', transform=transform_train)

# Split dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load a pre-trained model (e.g., ResNet18) and modify the final layer for the Food-101 dataset
net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 101)  # 101 classes for Food-101 dataset
net = net.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-4)  # AdamW optimizer with weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR after every 5 epochs

# Define the model path
model_path = 'enhanced_trained_model.pth'  # Path to save the trained model

# Save and Load checkpoint functions
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"Saving checkpoint at {filename}...")
    torch.save(state, filename)

def load_checkpoint(checkpoint_file):
    print(f"Loading checkpoint from {checkpoint_file}...")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    return checkpoint

# Training function
def train_model(net, train_loader, criterion, optimizer, scheduler, num_epochs=10, start_epoch=0):
    print("Starting the training process... Let's get this bread! 💪")
    net.train()

    try:
        while True:  # Continuous training loop until interrupted
            print(f"Starting a new training cycle of {num_epochs} epochs... 🚀")
            for epoch in range(num_epochs):
                print(f"Starting Epoch {epoch + 1}/{num_epochs}... 🚀")
                running_loss = 0.0
                progress_bar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='Training')

                for i, data in progress_bar:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                # Step the scheduler
                scheduler.step()

                # Calculate the average loss for the epoch
                average_loss = running_loss / len(train_loader)
                print(f"🚩 Finished Epoch {epoch + 1}/{num_epochs}. Average Loss: {average_loss:.4f}")

                # Save a checkpoint after every epoch
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                save_checkpoint(checkpoint)

            # Run validation after each cycle of training
            validate_model(net, val_loader, criterion)
            print("Training cycle completed. Starting again... Press CTRL+C to stop.")

    except KeyboardInterrupt:
        print("Training interrupted. Stopping training loop.")
        torch.save(net.state_dict(), model_path)
        print(f"🎉 Model saved successfully as '{model_path}'")

# Validation function with detailed comments
def validate_model(net, val_loader, criterion):
    print("Starting validation... Let's see what we've learned! 🧐")
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # Add progress bar for validation
    progress_bar = tqdm(enumerate(val_loader, 0), total=len(val_loader), desc='Validating')

    with torch.no_grad():
        for i, data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")

    # Comment on performance
    if accuracy > 85:
        print("🚀 Excellent performance! The model is achieving high accuracy.")
    elif accuracy > 70:
        print("😊 Good performance! The model is doing well, but there is room for improvement.")
    else:
        print("🔧 The model needs more training or adjustments. Consider tuning hyperparameters or data augmentation.")

    print("🎉 Validation complete! Time to celebrate (or adjust)! 🎉")

# Main function with prompt for retraining
if __name__ == '__main__':
    # Check if a pre-trained model exists, prompt for retraining
    if os.path.exists(model_path):
        retrain = input("A trained model already exists. Do you want to retrain the model? (yes/no): ").strip().lower()
        if retrain == 'yes':
            print("🚀 Retraining the model from scratch... Let’s go!")
            train_model(net, train_loader, criterion, optimizer, scheduler, num_epochs=10)  # Retrain for 10 epochs
            # After retraining, validate
            validate_model(net, val_loader, criterion)
        else:
            print(f"🛠 Loading saved model from '{model_path}' for validation...")
            net.load_state_dict(torch.load(model_path, map_location=device))  # Load model safely
            net.eval()  # Set to evaluation mode
            validate_model(net, val_loader, criterion)  # Run validation after loading model
    else:
        print("No saved model found. Let's train from scratch! 🚀")
        train_model(net, train_loader, criterion, optimizer, scheduler, num_epochs=10)  # Train for 10 epochs
