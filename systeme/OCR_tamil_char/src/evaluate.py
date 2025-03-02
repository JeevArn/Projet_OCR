import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import TamilOCRModel
from dataset import TamilCharacterDataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_unicode_mapping():
    """
    Load the Tamil Unicode mapping from the mapping file.
    
    Returns:
        dict: Mapping from class index to Tamil Unicode character
    """
    mapping = {}
    with open('systeme/OCR_tamil_char/tamil-unicode-mapping.txt', 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        for line in f:
            if line.strip():  # Skip empty lines
                class_idx, unicode_hex = line.strip().split(';')
                # Convert hex codes to actual characters
                char = ''.join(chr(int(code, 16)) for code in unicode_hex.split())
                mapping[int(class_idx)] = char
    return mapping

def plot_confusion_matrix(cm, classes, save_dir='systeme/OCR_tamil_char/plots'):
    """
    Plot confusion matrix using seaborn's heatmap with Tamil characters.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        classes (list): List of class labels
        save_path (str): Path to save the confusion matrix plot
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load Tamil character mapping
    tamil_chars = load_unicode_mapping()
    
    # Convert numerical classes to Tamil characters with indices
    tamil_labels = [f"{tamil_chars[int(c)]} ({c})" for c in classes]
    
    # Normalize the confusion matrix to percentages
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_sum = np.where(cm_sum == 0, 1, cm_sum)
    cm_normalized = (cm / cm_sum) * 100
    
    # Create figure and axes with larger size
    plt.figure(figsize=(40, 40))
    
    # Try to use a Unicode-compatible font, with fallbacks
    font_options = ['Arial Unicode MS', 'DejaVu Sans', 'FreeSans', 'Arial']
    for font in font_options:
        try:
            plt.rcParams['font.family'] = font
            # Test if the font works with Tamil characters
            plt.text(0, 0, 'அ', fontfamily=font)
            break
        except:
            continue
    
    # Plot the confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,  # Show numbers in cells
        fmt='.0f',   # Format as integer percentage
        cmap='Blues',  # Use Blues colormap
        xticklabels=tamil_labels,
        yticklabels=tamil_labels,
        square=True,  # Make cells square
        cbar_kws={
            'label': 'Percentage (%)',
            'orientation': 'vertical'
        },
        annot_kws={'size': 8},  # Slightly larger text size for better fit
        vmin=0,      # Minimum value for colormap
        vmax=100     # Maximum value for colormap
    )
    
    # Customize the plot
    plt.title('Confusion Matrix (% of True Labels)', pad=20, fontsize=24)
    plt.ylabel('True Label', fontsize=20, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=20, labelpad=10)
    
    # Rotate x-axis labels for better readability and increase font size
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high DPI
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300, facecolor='white')

    plt.close()

def plot_class_accuracies(predictions, true_labels, confidences, classes, save_dir='systeme/OCR_tamil_char/plots'):
    """
    Plot additional evaluation metrics.
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        confidences: Prediction confidences
        classes: List of class labels
        save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load Tamil character mapping
    tamil_chars = load_unicode_mapping()
    
    # Per-class Accuracy Plot with Tamil Characters
    class_accuracies = {}
    class_counts = {}
    for c in classes:
        mask = np.array(true_labels) == int(c)
        count = np.sum(mask)
        if count > 0:  # Only calculate if we have samples for this class
            class_accuracies[c] = np.mean(np.array(predictions)[mask] == np.array(true_labels)[mask]) * 100
            class_counts[c] = count
    
    # Create labels with Tamil characters and indices
    tamil_labels = [f"{tamil_chars[int(c)]} ({c})" for c in class_accuracies.keys()]
    
    # Sort by accuracy for better visualization
    sorted_items = sorted(zip(tamil_labels, class_accuracies.values(), class_counts.values()), 
                        key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_accuracies, sorted_counts = zip(*sorted_items)
    
    # Calculate figure size based on number of classes
    num_classes = len(sorted_labels)
    fig_width = max(20, num_classes * 0.3)  # Adjust width based on number of classes
    plt.figure(figsize=(fig_width, 12))
    
    # Try to use a Unicode-compatible font, with fallbacks
    font_options = ['Arial Unicode MS', 'DejaVu Sans', 'FreeSans', 'Arial']
    for font in font_options:
        try:
            plt.rcParams['font.family'] = font
            # Test if the font works with Tamil characters
            plt.text(0, 0, 'அ', fontfamily=font)
            break
        except:
            continue
    
    # Create bar plot with adjusted width
    x = np.arange(len(sorted_labels))
    bar_width = 0.6  # Make bars thinner
    bars = plt.bar(x, sorted_accuracies, width=bar_width)
    
    # Customize the plot
    plt.title('Accuracy by Character Class', fontsize=16, pad=20)
    plt.xlabel('Tamil Characters', fontsize=14, labelpad=10)
    plt.ylabel('Accuracy (%)', fontsize=14, labelpad=10)
    
    # Set x-axis ticks and labels with more space
    plt.xticks(x, sorted_labels, rotation=45, ha='right', fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to start from 0 and end at 100
    plt.ylim(0, 100)  # Changed back to 100 since we don't need extra space for labels
    
    # Add more space at the bottom for labels
    plt.subplots_adjust(bottom=0.2)
    
    # Save the plot with high quality
    plt.savefig(os.path.join(save_dir, 'class_accuracies.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def evaluate_model(model, test_loader, criterion, device, num_classes):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The trained OCR model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        num_classes: Number of character classes
    
    Returns:
        tuple: (test_loss, test_accuracy, predictions, true_labels, confidence_scores)
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    # Evaluate model
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions, labels, and confidence scores
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate metrics
    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = running_corrects.double() / len(test_loader.dataset)
    
    return test_loss, test_accuracy, all_preds, all_labels, all_confidences

def main():
    """
    Main evaluation function.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create test dataset
    test_dataset = TamilCharacterDataset(
        data_dir='systeme/OCR_tamil_char/data/70-30-split/Test',
        labels_file='systeme/OCR_tamil_char/data/70-30-split/test_labels.csv',
        transform=transform
    )
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Get number of classes
    num_classes = len(pd.read_csv('systeme/OCR_tamil_char/data/70-30-split/train_labels.csv')['Ground Truth'].unique())
    print(f'Number of classes: {num_classes}')
    
    # Initialize model
    model = TamilOCRModel(num_classes=num_classes).to(device)
    
    # Load trained model weights
    if os.path.exists('systeme/OCR_tamil_char/models/best_model.pth'):
        model.load_state_dict(torch.load('systeme/OCR_tamil_char/models/best_model.pth', map_location=device))
        print('Loaded trained model weights')
    else:
        print('Error: No trained model weights found!')
        return
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    test_loss, test_accuracy, predictions, true_labels, confidences = evaluate_model(
        model, test_loader, criterion, device, num_classes
    )
    
    # Print overall metrics
    print('\nTest Results:')
    print(f'Loss: {test_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.4f}')
    print(f'Average Confidence: {np.mean(confidences):.4f}')
    
    # Get unique classes
    classes = sorted(pd.read_csv('systeme/OCR_tamil_char/data/70-30-split/train_labels.csv')['Ground Truth'].unique())
    
    # Generate and print detailed classification report
    report = classification_report(true_labels, predictions, target_names=[str(c) for c in classes])
    print('\nClassification Report:')
    print(report)
    
    # Save classification report to file
    with open('systeme/OCR_tamil_char/classification_report.txt', 'w') as f:
        f.write('Test Results:\n')
        f.write(f'Loss: {test_loss:.4f}\n')
        f.write(f'Accuracy: {test_accuracy:.4f}\n')
        f.write(f'Average Confidence: {np.mean(confidences):.4f}\n\n')
        f.write('Classification Report:\n')
        f.write(report)
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, classes)
    
    # Generate class accuracies plot
    plot_class_accuracies(predictions, true_labels, confidences, classes)
    
    # Save high-confidence misclassifications for analysis
    misclassified_indices = np.where(np.array(predictions) != np.array(true_labels))[0]
    high_conf_errors = [(i, true_labels[i], predictions[i], confidences[i]) 
                       for i in misclassified_indices 
                       if confidences[i] > 0.8]  # Filter for high confidence mistakes
    
    if high_conf_errors:
        # Load Tamil character mapping
        tamil_chars = load_unicode_mapping()
        
        # Create data for the table
        table_data = []
        for idx, true_label, pred_label, conf in sorted(high_conf_errors, key=lambda x: x[3], reverse=True):
            table_data.append({
                'True Character': tamil_chars[true_label],
                'True Label': true_label,
                'Predicted Character': tamil_chars[pred_label],
                'Predicted Label': pred_label,
                'Confidence': f'{conf:.3f}'
            })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        df.to_csv('systeme/OCR_tamil_char/plots/high_confidence_mistakes.csv', index=False)

if __name__ == '__main__':
    main() 
