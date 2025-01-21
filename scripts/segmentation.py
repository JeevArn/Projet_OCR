import os
import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output



def process_image(image_path):
    """
    Charge et prétraite l'image (conversion en niveaux de gris et binarisation).
    
    Parameters:
        image_path (str ou PIL.Image): Chemin de l'image ou objet PIL.
    
    Returns:
        tuple:
            - binary_image : image binaire traitée.
            - original_image : image originale pour des manipulations supplémentaires.
    """
    # Si l'image est un chemin
    if isinstance(image_path, str):
        
        # Charger l'image
        image = cv2.imread(image_path)

        # Vérifier si l'image est valide
        if image is None:
            raise ValueError(f"Impossible de charger l'image à partir de {image_path}. Vérifiez le chemin ou le format.")
    
    # Si l'image est au format PIL
    elif isinstance(image_path, Image.Image):

        # Convertir en numpy array et changer de format (RGB -> BGR)
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    
    # Vérifier si l'entrée est valide
    else:
        raise TypeError("L'entrée doit être soit un chemin vers une image (str), soit une image PIL.Image.Image.")
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarisation des pixels
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary_image, image



def extract_boxes(binary_image, boxes):
    """
    Extrait les zones d'intérêt (caractères, mots, lignes) à partir des coordonnées des boxes.
    
    Parameters:
        binary_image (numpy array): Image binaire traitée.
        boxes (str): Données des boxes détectées par PyTesseract.
    
    Returns:
        list: Liste d'images découpées correspondant aux boxes.
    """
    extracted_images = []

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
            y = binary_image.shape[0] - y
            h = binary_image.shape[0] - h

            # Créer une image de caractère à partir des coordonnées
            extracted_images.append(binary_image[h:y, x:w])
    
    return extracted_images



def word_to_chars(image_path):
    """
    Découpe une image de mot en images de caractères individuels en utilisant PyTesseract.
    
    Parameters:
        image_path (str): Chemin vers l'image du mot.
    
    Returns:
        list: Liste des images des caractères en format PIL.
    """

    # Charger et prétraiter l'image (conversion en niveaux de gris et binarisation)
    binary_image, image = process_image(image_path)

    # Utiliser PyTesseract pour détecter les boxes des caractères
    boxes = pytesseract.image_to_boxes(binary_image, config="--psm 7")

    # Initialiser la liste des caractères
    characters = []

    # Parcourir les boxes détectées
    for char_image in extract_boxes(binary_image, boxes):

        # Créer une image de caractère à partir des coordonnées
        char_pil = Image.fromarray(char_image)

        # Ajouter la caractère à la liste
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
    binary_image, image = process_image(image_path)

    # Utiliser PyTesseract pour détecter les boxes des mots
    boxes = pytesseract.image_to_data(binary_image, config="--psm 6", output_type=Output.DICT)

    # Initialiser la liste des mots
    words = []

    # Nombre de boxes détectées
    n_boxes = len(boxes['text'])

    # Parcourir les boxes détectées et extraire les images des mots
    for i in range(n_boxes):

        # Si les boxes ne sont pas vides
        if boxes['text'][i].strip():
            
            # Extraire les caractères et les coordonnées de la box
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]

            # Créer une image de mot à partir des coordonnées
            word_image = binary_image[y:y+h, x:x+w]

            # Conversion en PIL
            word_pil = Image.fromarray(word_image)

            # Ajouter le mot à la liste
            words.append(word_pil)

    return words



def doc_to_lines(image_path):
    """
    Découpe une image de document en images de lignes en utilisant PyTesseract.
    
    Parameters:
        image_path (str): Chemin vers l'image du document.
    
    Returns:
        list: Liste des images des lignes en format PIL.
    """
    binary_image, image = process_image(image_path)

    # Utiliser PyTesseract pour détecter les boxes des lignes
    boxes = pytesseract.image_to_data(binary_image, config="--psm 6", output_type=Output.DICT)

    # Initialiser la liste des lignes
    lines = []

    # Nombre de boxes détectées
    n_boxes = len(boxes['text'])

    # Parcourir les boxes détectées et extraire les images des mots
    for i in range(n_boxes):

        # Si les boxes ne sont pas vides
        if boxes['text'][i].strip():

            # Extraire les caractères et les coordonnées de la box
            y1 = boxes['top'][i]
            y2 = y1 + boxes['height'][i]

            # Vérifier si c'est une nouvelle ligne ou si la box est sur la même ligne
            if not lines or y1 > lines[-1]['y2']:

                # Ajouter une nouvelle ligne avec les informations de la box
                lines.append({'y1': y1, 'y2': y2, 'image': binary_image[y1:y2, :]})
            else:
                # Mettre à jour la coordonnée du bas de la dernière ligne et ajouter la box à l'image de la ligne existante
                lines[-1]['y2'] = y2
                lines[-1]['image'] = binary_image[lines[-1]['y1']:y2, :]

    # Créer une liste d'images PIL pour chaque ligne
    line_images = [Image.fromarray(line['image']) for line in lines]
    
    return line_images