"""
Ce script prend en arguments un chemin vers une image de texte et son type (mot, ligne ou paragraphe) et retourne le contenu textuel de l'image.
"""



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
import argparse
import sys
from segmentation import word_to_chars, line_to_words, doc_to_lines



def predict_word(image_path, model, label_encoder, image_size=(28, 28)):
    """
    Prédit un mot à partir d'une image de mot.
    
    Parameters:
        image_path (str): Chemin vers l'image du mot.
        model: Modèle entraîné pour la reconnaissance de caractères.
        label_encoder: Encodeur pour convertir les prédictions en caractères.
        image_size (tuple): Dimensions des images pour le modèle.
    
    Returns:
        str: Mot prédit.
    """
    # Découper l'image en caractères
    character_images = word_to_chars(image_path)

    predicted_word = ""
    for img in character_images:
        # Prétraiter l'image
        img_resized = img.resize(image_size)
        img_array = np.array(img_resized) / 255.0
        img_array = img_array.reshape(1, image_size[0], image_size[1], 1)

        # Prédire le caractère
        prediction = model.predict(img_array)
        char = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        predicted_word += char

    return predicted_word



def predict_text(docPath, docType, model, label_encoder):
    """
    Prédit le texte d'une image de document.
    
    Parameters:
        docPath (str): Chemin vers l'image du document.
        docType (str): Type de document ("word", "line" ou "doc").
        model: Modèle entraîné pour la reconnaissance de caractères.
        label_encoder: Encodeur pour convertir les prédictions en caractères.
    
    Returns:
        str: Texte prédit.
    """
    # Si l'image est un mot
    if docType == "word":

        # Prédire le mot
        predicted_word = predict_word(docPath, model, label_encoder)

        # Afficher le résultat
        return predicted_word


    # Si l'image est une ligne
    elif docType == "line":

        # Segmenter la ligne en mots
        words = line_to_words(docPath)

        # Initialiser le résultat
        result = ""

        # Parcourir chacun des mots de la ligne
        for word in words:

            # Prédire le mot
            predicted_word = predict_word(word, model, label_encoder)

            # Ajouter le mot au résultat
            result += predicted_word + " "
        
        # Afficher le résultat
        return result


    # Si l'image est un document
    elif docType == "doc":
        
        # Segmenter le document en lignes
        lines = doc_to_lines(docPath)

        # Initialiser le résultat
        result = ""

        # Parcourir chacune des lignes du document
        for line in lines:

            # Segmenter la ligne en mots
            words = line_to_words(line)

            # Parcourir chacun des mots de la ligne
            for word in words:

                # Prédire le mot
                predicted_word = predict_word(word, model, label_encoder)

                # Ajouter le mot au résultat
                result += predicted_word + " "

        # Afficher le résultat
        return result



def main():

    # Vérification du nombre d'arguments
    if len(sys.argv) != 3:
        print("Erreur : Nombre incorrect d'arguments.")
        print("Usage : python3 ocr_script.py <docPath> <docType>")
        print('Exemple : python3 ocr_script.py chemin/vers/image.png word')
        sys.exit(1)

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Renvoie le contenu textuel d'une image.")
    parser.add_argument('docPath', help="Chemin vers l'image à océriser")
    parser.add_argument('docType', help='Type de document (doc/line/word)')
    args = parser.parse_args()

    # Charger le modèle d'OCR pré-entraîné
    model = load_model('../models/OCR_50000w_20e.h5')

    # Formater le LabelEncoder
    classes = np.load('../data/classes.npy')
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    # Effectuer l'OCR de l'image
    ocr_text = predict_text(args.docPath, args.docType, model, label_encoder)
    print(ocr_text)



if __name__ == '__main__':
    main()