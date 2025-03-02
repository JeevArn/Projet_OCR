import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class TamilCharacterDataset(Dataset):
    """
    Custom Dataset class for Tamil character images.
    Handles loading and preprocessing of Tamil character images and their labels.
    Implements the PyTorch Dataset interface for use with DataLoader.
    
    The dataset expects:
    - A directory containing Tamil character images
    - A CSV file with columns 'FileNames' and 'Ground Truth'
    - Optional transforms to be applied to the images
    
    Args:
        data_dir (str): Path to directory containing the image files
        labels_file (str): Path to CSV file containing image filenames and labels
        transform (callable, optional): Optional transforms to be applied to images
    """
    def __init__(self, data_dir, labels_file, transform=None):
        """
        Initialize the dataset by loading the labels file and setting up transforms.
        The labels file should be a CSV with columns:
        - FileNames: image filenames
        - Ground Truth: corresponding labels (character classes)
        """
        self.data_dir = data_dir  # Directory containing the images
        self.transform = transform  # Transforms to be applied to images
        self.labels_df = pd.read_csv(labels_file)  # Load the labels CSV file
        
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        Required by PyTorch Dataset interface.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        """
        Fetch and return a single sample from the dataset.
        Required by PyTorch Dataset interface.
        
        Args:
            idx (int or torch.Tensor): Index of the sample to fetch
        
        Returns:
            tuple: (image, label) where image is the processed image tensor
                  and label is the corresponding class index
        """
        if torch.is_tensor(idx):
            # Convert tensor index to list
            idx = idx.tolist()
            
        # Construct full path to image file
        img_name = os.path.join(self.data_dir, self.labels_df.iloc[idx, 1])  # FileNames is in column 1
        
        # Load image and convert to grayscale
        image = Image.open(img_name).convert('L')  # 'L' mode = grayscale
        
        # Get corresponding label
        label = self.labels_df.iloc[idx, 2]  # Ground Truth is in column 2
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
            
        return image, label 