"""
Script to predict Tamil text from images using a trained PyTorch model.
Takes an image path and type (word, line, or document) as input and returns the predicted text.
"""
import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import click
import pandas as pd
from typing import List, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from systeme.OCR_latin_char.scripts.segmentation import word_to_chars, line_to_words, doc_to_lines
from systeme.OCR_tamil_char.src.model import TamilOCRModel


def load_tamil_mapping(mapping_file: str) -> dict:
    """
    Load the Tamil character mapping from file.
    
    Args:
        mapping_file (str): Path to the mapping file
        
    Returns:
        dict: Mapping from class indices to Tamil Unicode characters
    """
    mapping_df = pd.read_csv(mapping_file, sep=';')
    return {idx: unicode_char for idx, unicode_char in zip(mapping_df['class'], mapping_df['unicode'])}


def preprocess_image(image: Image.Image, image_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """
    Preprocess an image for the model.
    
    Args:
        image (PIL.Image): Input image
        image_size (tuple): Target size for the image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Apply transformations
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def predict_word(image_path: str, model: torch.nn.Module, char_mapping: dict, device: torch.device) -> str:
    """
    Predict text from a word image.
    
    Args:
        image_path (str): Path to the word image
        model (torch.nn.Module): Trained PyTorch model
        char_mapping (dict): Mapping from class indices to Tamil Unicode characters
        device (torch.device): Device to run predictions on
        
    Returns:
        str: Predicted word
    """
    # Get individual characters
    character_images = word_to_chars(image_path)
    
    predicted_word = ""
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for char_img in character_images:
            # Preprocess the character image
            input_tensor = preprocess_image(char_img).to(device)
            
            # Get prediction
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Convert to Unicode character
            unicode_char = char_mapping.get(predicted_class, '')
            if ' ' in unicode_char:  # Handle composite characters
                unicode_chars = [chr(int(code, 16)) for code in unicode_char.split()]
                predicted_word += ''.join(unicode_chars)
            else:
                predicted_word += chr(int(unicode_char, 16))
    
    return predicted_word


def predict_text(doc_path: str, doc_type: str, model: torch.nn.Module, char_mapping: dict, device: torch.device) -> str:
    """
    Predict text from a document image.
    
    Args:
        doc_path (str): Path to the document image
        doc_type (str): Type of document ('word', 'line', or 'doc')
        model (torch.nn.Module): Trained PyTorch model
        char_mapping (dict): Mapping from class indices to Tamil Unicode characters
        device (torch.device): Device to run predictions on
        
    Returns:
        str: Predicted text
    """
    if doc_type == "word":
        return predict_word(doc_path, model, char_mapping, device)
    
    elif doc_type == "line":
        words = line_to_words(doc_path)
        return ' '.join(predict_word(word, model, char_mapping, device) for word in words)
    
    elif doc_type == "doc":
        lines = doc_to_lines(doc_path)
        result = []
        for line in lines:
            words = line_to_words(line)
            line_text = ' '.join(predict_word(word, model, char_mapping, device) for word in words)
            result.append(line_text)
        return '\n'.join(result)
    
    return None


@click.command()
@click.argument('doc_path')
@click.argument('doc_type')
def main(doc_path: str, doc_type: str):
    """Main function to run Tamil OCR prediction."""
    
    if not os.path.exists(doc_path):
        print(f"The file '{doc_path}' does not exist.")
        sys.exit(1)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load character mapping first to get number of classes
    mapping_file = 'systeme/OCR_tamil_char/tamil-unicode-mapping.txt'
    char_mapping = load_tamil_mapping(mapping_file)
    num_classes = len(char_mapping)
    
    # Initialize model architecture
    model = TamilOCRModel(num_classes=num_classes)
    
    # Load the trained weights
    model_path = 'systeme/OCR_tamil_char/models/best_model.pth'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Perform OCR
    predicted_text = predict_text(doc_path, doc_type, model, char_mapping, device)
    print(predicted_text)


if __name__ == '__main__':
    main() 