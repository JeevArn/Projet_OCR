"""
Usage
-----
uvicorn interfaces.api.app:app --reload

http://127.0.0.1:8000

TEST
----

For Latin OCR:
curl -X POST http://127.0.0.1:8000/ocr/ \
-H "Content-Type: application/json" \
-d '{"image_path": "systeme/OCR_latin_char/data/images/doc1.png", "ocr_type": "doc", "language": "latin"}'

For Tamil OCR:
curl -X POST http://127.0.0.1:8000/ocr/ \
-H "Content-Type: application/json" \
-d '{"image_path": "systeme/OCR_tamil_char/data/images_for_testing/text1.png", "ocr_type": "doc", "language": "tamil"}'

"""
import subprocess
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum

app = FastAPI()

class Language(str, Enum):
    LATIN = "latin"
    TAMIL = "tamil"

class OCRRequest(BaseModel):
    image_path: str
    ocr_type: str
    language: Language

@app.post("/ocr/")
async def perform_ocr(request: OCRRequest):
    # Vérifier si le fichier existe
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image non trouvée.")

    try:
        # Définir le PYTHONPATH pour inclure le répertoire parent de 'systeme'
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Sélectionner le script OCR approprié en fonction de la langue
        if request.language == Language.LATIN:
            script_path = 'systeme/OCR_latin_char/scripts/ocr_script.py'
        else:  # Tamil
            script_path = 'systeme/OCR_tamil_char/src/predict.py'

        # Utiliser subprocess pour appeler la fonction main via la ligne de commande
        command = [
            'python', script_path,
            request.image_path, request.ocr_type
        ]

        # Exécuter la commande avec l'environnement modifié
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)

        # Retourner le texte obtenu depuis la sortie du script
        return {"text": result.stdout, "language": request.language}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=e.stderr)
