# Méthodologie

## OCR caractères latins

Afin de parvenir à notre objectif, à savoir obtenir un script capable de reconnaître les caractères d'une image grâce à un modèle neuronal pré-entraîné, nous avons procédé avec la méthodologie suivante.

**1. Données**

Nous avons commencé par trouver un jeu de données permettant d'entraîner un modèle d'OCR. Nous avons utilisé le dataset `DonkeySmall/OCR-English-Printed-12` trouvé sur HuggingFace, contenant 1.000.000 images de mots avec leur transcription en chaîne de caractères.
Voici des exemples d'images du dataset :

![Image dataset](systeme/data/doc/extrait_dataset.png "Extrait du jeu de données")

**2. Prétraitement**

Une fois nos données récupérées, il nous a fallu les prétraiter. En effet, nous voulion un modèle capable de reconnaître des images de caractères. Or, le jeu de données étant constitué de mots et non de caractères uniques, nous avons du trouver un moyen de segmenter les images de mots en images de caractères. Pour cela, nous avons utilisé le module `PyTesseract`, qui est un outil de reconnaissance de caractères. Nous avons uniquement utilisé sa méthode `image_to_boxes`, qui segmente une image de texte en images de caractères. Nous avons appliqué cette méthode aux mots de notre dataset et avons conservé uniquement les cas où le nombre de caractères trouvés par segmentation correspondait au nombre de caractères du mot, afin de supprimer les cas où des caractères seraient restés "collés" ou au contraire où un caractère aurait été "coupé en deux".
Nous avons ainsi obtenu une liste de tuples où le premier élément de chaque tuple correspond à la liste des images de caractères du mot et le second aux caractères correspondants. Nous avons alors pu séparer notre dataset en un ensemble d'entraînement (80%) et un ensemble de test (20%).

**3. Modèle**

Nous avons ensuite commencé à construire notre modèle. Nous avons testé différentes architectures de modèle, mais celle qui semblait donner les meilleurs résultats est la suivante.

La première couche extrait des caractéristiques locales des images de caractères en appliquant des filtres de 3x3 pixels à l'image d'input (28x28 pixels). On utilise ensuite une fonction d'activation ReLU. La couche suivante, MaxPooling2D, réduit de moitié la map de features en prenant la valeur maximale de chaque patch de features pour conserver les caractéristiques les plus importantes. Cela permet de rendre le réseau plus efficace computationnellement et de réduire le sur-apprentissage. On utilise ensuite une seconde couche de convolution qui prend en entrée la sortie de la première couche et on effectue un second MaxPooling. On applique ensuite une couche Flatten aux sorties 2D des couches de convolution et de pooling pour les transformer en un vecteur à 1 dimension. Cela nous permet de pouvoir connecter nos couches précédentes à une couche dense entièrement connectée qui a pour rôle d'apprendre des relations entre les caractéristiques extraites par les différentes couches. On applique ensuite une couche de Dropout, qui désactive aléatoirement 50% des neurones, afin encore une fois d'éviter le sur-apprentissage. On utilise enfin une seconde couche dense avec une fonction d'activation softmax, qui correspond à notre couche de sortie.

Nous avons utilisé un optimiseur Adam et une categorical_crossentropy pour la loss car il s'agit d'un problème de classification multi-classes.

Pour l'entraînement du modèle, nous avons testé différents paramètres mais ceux qui semblaient donner les meilleures performances est un entraînement avec 20 époques et un batch_size. Cela signifie que le modèle parcourt 20 fois l'ensemble des données d'entraînement, et qu'il met à jour ses poids tous les 32 exemples.

Pour réaliser ce modèle, assez complexe, nous nous sommes beaucoup inspirées des ressources suivantes :
- Simple Convolutional Neural Network (CNN) for Dummies in PyTorch: A step-by-step guide. Medium. https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80
- Cours de CNN de Cédric Gendrot
- Hariyeh. CNN model implementing OCR. Kaggle. https://www.kaggle.com/code/harieh/cnn-model-implementing-ocr
- DataCorner. Image processing - Partie 7. DataCorner. https://datacorner.fr/image-processing-7/

**4. Vote des modèles**

Notre jeu de données contient, on le rappelle, 1.000.000 images de mots avec leur transcription. Ce chiffre étant d'autant plus grand que pour chaque mot, nous effectuons un prétraitement constistant en la segmentation de ses caractères, nous avons rapidement été confrontées à des contraintes matérielles. En effet, au-delà de 50000 mots, nos PC étaient dépassés. Cherchant une solution pour entraîner nostre modèle sur une plus grande partie des données, nous sommes finalement parvenues à une solution pour contourner le problème. Au lieu d'essayer d'entraîner le modèle sur plus de données, nous avons fait le choix d'entraîner le même modèle mais sur 5 sous-ensembles distincts du jeu de données. Ainsi, le premier modèle est entraîné sur les 50000 premiers mots, le deuxième sur les 50000 à 10000 mots suivants, etc. Nous avons ainsi obtenu 5 modèles identiques dans leur architecture et dans leurs paramètres, mais ayant été entraîné sur des données différentes. Nous avons ensuite défini une fonction de vote des modèles. Cela signifie que lorsque l'utilisateur donne une image de texte en entrée, chaque modèle va effectuer une prédiction du contenu textuel de l'image. Puis, notre fonction itère sur chaque caractère prédit par chaque modèle et compte simplement pour chacun lequel est le plus fréquent.

Prenons l'exemple du mot "agreement". Admettons que les modèles prédisent les textes suivants :

- model_1 : "aqreenent"
- model_2 : "agreeoent"
- model_3 : "agreement"
- model_4 : "agreemeot"
- model_5 : "aqreement"

Pour le premier caractère, pas de souci, les 5 modèles ont tous prédit un "a".

Pour le second caractère en revanche, deux des modèles ont prédit un "q" tandis que les trois autres ont prédit un "g". On va donc choisir un "g" pour la prédiction finale car c'est le caractère prédit pour la majorité des modèles, etc.

Cette astuce nous permet d'augmenter la fiabilité de la prédiction textuelle.

#### Résultats

- Les résultats (fichiers output, visualisations…) et une discussion sur ces résultats (ce que vous auriez aimé faire et ce que vous avez pu faire par exemple)

## OCR caractères tamouls

Notre objectif pour ce modèle est de pouvoir reconnaître des caractères tamouls manuscrits, en complément de notre système de reconnaissance de caractères latins.

**1. Données**

Nous avons utililé le dataset uTHCD Unconstrained Tamil Handwritten Database (Shaffi 2021) disponible en libre accès sur [Kaggle](https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database). Ce jeu de données contient 90950 images de caractères tamouls manuscrits, couvrant 156 caractères (et donc 156 classes) différents.

**2. Prétraitement**

*Normalisation des images* :
   - Redimensionnement uniforme à 64x64 pixels
   - Conversion en niveaux de gris
   - Normalisation des valeurs de pixels entre -1 et 1 (mean=0.5, std=0.5)

*Gestion des classes* :
   - Création d'un mapping entre les IDs de classes et le code Unicode correspondant aux caractères tamouls
   - Dataset prédivisé en ensembles d'entraînement (70%) et de test (30%), ce split est le même que celui utilisé dans l'étude de Shaffi 2021 afin de pouvoir comparer nos résultats sur un pied d'égalité

**3. Modèle**

Pour la reconnaissance des caractères tamouls, nous avons développé un réseau de neurones convolutif (CNN) en nous inspirant de celui utilisé par Shaffi 2021. L'architecture du modèle est la suivante :

*Couches de convolution* :
   - 4 blocs de convolution successifs
   - Chaque bloc contient :
     - Une couche Conv2D avec des filtres de taille 3x3
     - Une activation ReLU
     - Un MaxPooling2D pour réduire la dimensionnalité
   - Nombre de filtres progressif : 32 → 64 → 128 → 256

*Couches de classification* :
   - Pooling adaptatif pour gérer les dimensions variables
   - Dropout (50%) pour réduire le surapprentissage
   - Deux couches denses :
     - Une couche intermédiaire de 512 neurones
     - Une couche de sortie avec softmax (156 classes)

*Paramètres d'entraînement* :
   - Optimiseur : Adam avec learning rate fixe de 0.001
   - Fonction de perte : Categorical Cross-Entropy
   - Batch size : 32
   - Epochs : 25 avec early stopping (patience de 7 époques)
   - Validation split : 20%

**4. Résultats**

Les performances du modèle sur le jeu données de test sont les suivantes :

*Métriques globales* :
   - Accuracy : 0.9520
   - Loss : 0.1569
   - Précision moyenne (macro) : 0.95
   - Rappel moyen (macro) : 0.95
   - F1-Score moyen (macro) : 0.95

*Analyse des erreurs* :

Notre modèle confond facilement les caractères composés qui ne se différencient seulement par un point une longueur de trait par exemple ஜ் VS ஜ ou bien ஙு VS ங. Un tableau des erreurs dont la confiance de la prédiction est >80% peut être retrouvé [ici](systeme/OCR_tamil_char/plots/high_confidence_mistakes.csv).
Nous avons également plotté la matrice de confusion disponible [ici](systeme/OCR_tamil_char/plots/confusion_matrix.png) et un graph des Accuracy par classes [ici](systeme/OCR_tamil_char/plots/class_accuracies.png).

*Conclusion sur le modèle d'OCR du tamoul* :

Shaffi 2021 avait obtenu une accuracy de 0.91 sur le jeu de donnée test, nous avons réussi à obtenir une accuracy de *0.95* soit une amélioration de 0.04. Cependant, notre modèle emet des erreurs avec une confiance très haute notamment sur les caractères composés, et il reste très peu performant sur les données extérieurs au test set.

## Interface Web et API

Nous avons choisi de créer un interface web avec Streamlit qui permet de créer des sites web directement en python. Pour l'API nous avons utilisé FastAPI.

## Références

- Shaffi, N., & Hajamohideen, F. (2021). uTHCD: A new benchmarking for Tamil handwritten OCR. IEEE Access, 9, 101469–101493. https://doi.org/10.1109/access.2021.3096823