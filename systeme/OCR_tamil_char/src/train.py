import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import TamilOCRModel
from dataset import TamilCharacterDataset
import os
from tqdm import tqdm
import pandas as pd

class EarlyStopping:
    """
    Early stopping handler class to prevent overfitting.
    Stops training when validation loss hasn't improved for a specified number of epochs.
    
    Args:
        patience (int): Number of epochs to wait before stopping training if no improvement
        min_delta (float): Minimum change in loss to be considered as an improvement
        verbose (bool): If True, prints message for each validation loss improvement
    """
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience  # Number of epochs to wait before stopping
        self.min_delta = min_delta  # Minimum change in loss to qualify as an improvement
        self.verbose = verbose  # Whether to print messages
        self.counter = 0  # Counter for patience
        self.best_loss = None  # Best validation loss seen so far
        self.early_stop = False  # Whether to stop training
        self.val_loss_min = float('inf')  # Initialize minimum validation loss as infinity

    def __call__(self, val_loss):
        """
        Check if training should be stopped.
        
        Args:
            val_loss (float): Current validation loss
        """
        if self.best_loss is None:
            # First epoch
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            # Loss has increased beyond minimum delta
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss has improved
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda', patience=7):
    """
    Train the OCR model with early stopping.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Maximum number of training epochs
        device: Device to train on (cuda/cpu)
        patience: Number of epochs to wait before early stopping
    """
    best_acc = 0.0  # Track best accuracy
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # ====== Training Phase ======
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over training data
        for inputs, labels in tqdm(train_loader, desc='Training'):
            # Move data to device (GPU/CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predictions
            loss = criterion(outputs, labels)  # Calculate loss
            
            # Backward pass and optimize
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)  # Accumulate loss
            running_corrects += torch.sum(preds == labels.data)  # Count correct predictions
        
        # Calculate epoch-level training metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # ====== Validation Phase ======
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over validation data
        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        # Calculate epoch-level validation metrics
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        # Check early stopping conditions
        early_stopping(val_loss)
        
        # Save model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        print()
        
        # Stop training if early stopping is triggered
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

def main():
    """
    Main function to set up and run the training process.
    """
    # Set up device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
    ])
    
    # Create full training dataset
    full_train_dataset = TamilCharacterDataset(
        data_dir='OCR_tamil_char/data/70-30-split/Train',
        labels_file='OCR_tamil_char/data/70-30-split/train_labels.csv',
        transform=transform
    )
    
    # Split training data into train and validation sets (80-20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f'Training set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')
    
    # Create data loaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Get the number of unique Tamil characters (classes)
    num_classes = len(pd.read_csv('OCR_tamil_char/data/70-30-split/train_labels.csv')['Ground Truth'].unique())
    print(f'Number of classes: {num_classes}')
    
    # Initialize the OCR model
    model = TamilOCRModel(num_classes=num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Standard loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
    
    # Start training with early stopping
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device, patience=7)

if __name__ == '__main__':
    main() 