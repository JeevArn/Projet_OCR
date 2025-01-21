import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFont, ImageDraw
from PIL import Image
import cv2
import pytesseract
from pytesseract import Output
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt



def word_to_chars(image_path):
    """
    Découpe une image de mot en images de caractères individuels en utilisant PyTesseract.
    
    Parameters:
        image_path (str): Chemin vers l'image du mot.
    
    Returns:
        list: Liste des images des caractères en format PIL.
    """
    # Charger l'image
    image = cv2.imread(image_path)

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarisation des pixels (0 si prche de noir ; 255 si proche de blanc)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Utiliser PyTesseract pour détecter les caractères
    boxes = pytesseract.image_to_boxes(binary_image, config="--psm 7")

    # Initialiser la liste des images des caractères
    characters = []

    # Vérifier si des boxes ont été détectées
    if boxes:

        # Parcourir les boxes et extraire les images des caractères
        for box in boxes.splitlines():

            # Spliter les données de la box
            box_data = box.split()

            # Vérifier que la box est bien au format attendu (char, x1, y1, x2, y2, page)
            if len(box_data) == 6:

                # Extraire les caractères et les coordonnées de la box
                char, x, y, w, h, _ = box_data

                # Convertir les coordonnées en entiers
                x, y, w, h = map(int, [x, y, w, h])

                # Ajustement des coordonnées entre pytesseract et cv2 (origine différente)           
                y = image.shape[0] - y
                h = image.shape[0] - h

                # Découper l'image du caractère
                char_image = binary_image[h:y, x:w]

                # Convertir en format PIL
                char_pil = Image.fromarray(char_image)
                characters.append(char_pil)

    return characters



def line_to_words(image_path):
    """
    Découpe une image de ligne en images de mots en utilisant PyTesseract.
    
    Parameters:
        image_path (str): Chemin vers l'image de la ligne.
    
    Returns:
        list: Liste des images des mots en format PIL.
    """
    pass



def doc_to_lines(image_path):
    """
    Découpe une image de document en images de lignes en utilisant PyTesseract.
    
    Parameters:
        image_path (str): Chemin vers l'image du document.
    
    Returns:
        list: Liste des images des lignes en format PIL.
    """
    pass