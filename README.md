# 🔍 Système OCR 🔍
Système de reconnaissance optique de caractères (OCR) supportant les caractères latins et tamouls.

## 📋 Table des matières ��
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Interface Web](#interface)
  - [API REST](#api-rest)
  - [Scripts en ligne de commande](#scripts)
- [Modèles et Performances](#models)

## 🛠 Installation 🛠 <a name="installation"></a>

1. Cloner le repository :
```bash
git clone https://github.com/JeevArn/Projet_OCR.git
cd projet_OCR
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 💻 Utilisation 💻 <a name="utilisation"></a>

### 🌐 Interface Web 🌐 <a name="interface"></a>

Notre interface web est disponible à cette adresse :  
🚀 **https://projetocr-nsnrgcfdbknwyzgxl3hgpz.streamlit.app** 🚀

### 🔌 API REST 🔌 <a name="api-rest"></a>

Lancer le serveur API :
```bash
uvicorn interfaces.api.app:app --reload
```
Exemple de requête cURL :
```
curl -X POST http://127.0.0.1:8000/ocr/ \
-H "Content-Type: application/json" \
-d '{
    "image_path": "systeme/OCR_tamil_char/data/images_for_testing/text1.png",
    "ocr_type": "doc",
    "language": "tamil"
}'
```
plus du documentation sur l'API est disponible sur notre [site](https://projetocr-nsnrgcfdbknwyzgxl3hgpz.streamlit.app).

### 📜 Scripts en ligne de commande 📜 <a name="scripts"></a>

#### OCR Latin
```bash
python systeme/OCR_latin_char/scripts/ocr_script.py <chemin_image> <type>
```
- `<chemin_image>` : Chemin vers l'image à analyser
- `<type>` : Type de document ('word', 'line', ou 'doc')  
(des images à tester sont disponibles dans [ici](systeme/OCR_latin_char/data/images/))

#### OCR Tamoul
```bash
python systeme/OCR_tamil_char/src/predict.py <chemin_image> <type>
```
- `<chemin_image>` : Chemin vers l'image à analyser
- `<type>` : Type de document ('word', 'line', ou 'doc')  
(des images à tester sont disponibles dans [ici](systeme/OCR_tamil_char/data/images_for_testing/))

## 🤖 Modèles et Performances 🤖 <a name="models"></a>

Nos modèles d'OCR de caractères latins et tamouls ont une accuracy de respectivement 75% et 95%, plus d'infos sur nos modèles dans le fichier [methodo](methodo.md).
