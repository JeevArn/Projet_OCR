# Optical Character Recognition

## Consignes

Le rendu devra comporter :

**1. Une documentation du projet traitant les points suivants :**

- Les objectifs du projet
- Les données utilisées (origine, format, statut juridique) et les traitements opérés sur celles-ci
- La méthodologie (comment vous vous êtes répartis le travail, comment vous avez identifié les problèmes et les avez résolus, différentes étapes du projet…)
- L’implémentation ou les implémentations (modélisation le cas échéant, modules et/ou API utilisés, différents langages le cas échéant)
- Les résultats (fichiers output, visualisations…) et une discussion sur ces résultats (ce que vous auriez aimé faire et ce que vous avez pu faire par exemple)

On attend de la documentation technique, pas une dissertation. Elle pourra prendre le format d’un ou plusieurs fichiers, d’un site web, d’un notebook de démonstration, à votre convenance

**La documentation ne doit pas, jamais, sous aucun prétexte, comporter de capture d’écran de code.**

**2. Le code Python et les codes annexes (JS par ex.) que vous avez produit. Le code doit être commenté. Des tests, ce serait bien. Évitez les notebooks, préférez les interfaces en ligne de commande ou web (ou graphiques si vous êtes très motivé⋅es)**

**3. Les éventuelles données en input et en output (ou un échantillon si le volume est important)**

Votre travail sera de réaliser une interface web en Python pour un système de TAL, de traitement ou d’accès à des données. Elle devra au moins comprendre une interface programmatique sous la forme d’une API REST utilisable par un serveur ASGI (c’est par exemple le cas de celles réalisée en FastAPI que vous avez vues en cours) et une interface utilisateur qui pourra prendre la forme d’un script ou très, très préférablement d’une interface web HTML + CSS + js


## Système

#### Objectif

Ecrire un script capable de reconnaître les caractères d'une image/document grâce à un modèle neuronal pré-entraîné. (`ocr_script.py`)

**Input :**

- image (fichier jpg ou png)
- type de document (mot unique, ligne unique, plusieurs lignes)

**Output :**

- texte correspondant au contenu de l'image

#### Fait :

- Trouver un jeu de données pour entraîner un modèle d'OCR (https://huggingface.co/datasets/DonkeySmall/OCR-English-Printed-12)
- Créer un modèle neuronal capable de reconnaître des caractères isolés (`train_ocr_model.ipynb`)
- Sauvegarder le modèle pour ne pas avoir à le réentraîner (car c'est trèèèèèèèès long) ! (`OCR_20000_words.h5`)
- Segmenter une image de mot en images de caractères (`word_to_chars`)

#### A faire :

- Segmenter une image de ligne en mots (`line_to_words`)
- Segmenter une image de document (plusieurs lignes) en lignes (`doc_to_lines`)

## Interfaces

#### Objectif

Proposer une interface programmatique et une interface web pour permettre à différents types d'utilisateurs d'utiliser notre système d'OCR.

#### Fait :

- Rien lol

#### A faire :

- Interface programmatique sous la forme d’une API REST utilisable par un serveur ASGI
- Interface utilisateur web (HTML + CSS + js)
