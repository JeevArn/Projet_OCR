import torch
import torch.nn as nn

class TamilOCRModel(nn.Module):
    """
    Convolutional Neural Network model for Tamil character recognition.
    The architecture consists of:
    - 4 convolutional blocks (each with Conv2D, ReLU, and MaxPool)
    - Adaptive pooling to handle variable input sizes
    - Fully connected layers with dropout for classification
    
    Args:
        num_classes (int): Number of Tamil characters to classify
    """
    def __init__(self, num_classes):
        super(TamilOCRModel, self).__init__()
        
        # CNN Feature Extraction Layers
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: grayscale image (1 channel) -> 32 feature maps
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial dimensions by half
            
            # Second Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 -> 64 feature maps
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 -> 128 feature maps
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth Convolutional Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 128 -> 256 feature maps
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to ensure fixed output size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Prevent overfitting
            nn.Linear(256, 512),  # First dense layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Additional dropout
            nn.Linear(512, num_classes)  # Final classification layer
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
        
        Returns:
            torch.Tensor: Logits for each class, shape (batch_size, num_classes)
        """
        # Extract features using CNN
        x = self.features(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        
        # Flatten the features
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        return x 