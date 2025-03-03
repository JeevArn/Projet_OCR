"""
Ce script prend en arguments chemin vers une image de texte et son type
(mot, ligne ou paragraphe) et retourne le contenu textuel de l'image.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from systeme.OCR_latin_char.scripts.segmentation import word_to_chars, line_to_words, doc_to_lines
import click
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from keras.utils import disable_interactive_logging
disable_interactive_logging()



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



def predict_text(doc_path, doc_type, model, label_encoder):
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
    if doc_type == "word":

        # Prédire le mot
        predicted_word = predict_word(doc_path, model, label_encoder)

        # Afficher le résultat
        return predicted_word


    # Si l'image est une ligne
    if doc_type == "line":

        # Segmenter la ligne en mots
        words = line_to_words(doc_path)

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
    if doc_type == "doc":

        # Segmenter le document en lignes
        lines = doc_to_lines(doc_path)

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

    return None


@click.command()
@click.argument('doc_path')
@click.argument('doc_type')
def main(doc_path: str, doc_type: str):
    """Fonction principale"""

    if not os.path.exists(doc_path):
        print(f"Le fichier '{doc_path}' n'existe pas.")
        sys.exit(1)

    # Charger le modèle d'OCR pré-entraîné
    model = load_model('systeme/OCR_latin_char/models/OCR_50000w_10e.h5')

    # Formater le LabelEncoder
    classes = np.load('systeme/OCR_latin_char/data/classes_1.npy')
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    # Effectuer l'OCR de l'image
    ocr_text = predict_text(doc_path, doc_type, model, label_encoder)
    print(ocr_text)



if __name__ == '__main__':
    main()
