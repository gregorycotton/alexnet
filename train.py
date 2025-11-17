import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from model import AlexNet
from dataset import Cifar10Dataset

# Hyperparams are based on the paper but adapted for CIFAR-10 (10 classes, 10 epochs instead of 90)
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 10
NUM_CLASSES = 10
DATA_DIR = "./data"
MODEL_SAVE_PATH = "alexnet_cifar10.pth"

# I'm using my MacBook Pro M1 Max 32GB but I'm gonna make it hardware agnostic
def get_default_device():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple M1/M2/M3 GPU)")
        return torch.device('mps')
    else:
        print("Using CPU")
        return torch.device('cpu')

def train():
    device = get_default_device()

    # Load dataset
    print("Loading datasets...")
    train_dataset = Cifar10Dataset(data_dir=DATA_DIR, train=True)
    test_dataset = Cifar10Dataset(data_dir=DATA_DIR, train=False)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=2)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=False, 
                             num_workers=2)
    
    print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} test images.")

    # Initialize mode and all that good stuff
    print("Initializing model...")
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    
    # Includes softmax
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), 
                          lr=LEARNING_RATE, 
                          momentum=MOMENTUM, 
                          weight_decay=WEIGHT_DECAY)
    
    # Paper manually monitors validation error but I'm gonna use a scheduler that automatically cuts the learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass + optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        end_time = time.time()
        
        print("-" * 50)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Time: {end_time - start_time:.2f}s, Avg Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.2f}%")
        print("-" * 50)
        
        scheduler.step()

    # Save the model
    print("Training finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()