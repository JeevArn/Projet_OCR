"""
Usage
-----
uvicorn interfaces.api.app:app --reload

http://127.0.0.1:8000

TEST
----

curl -X POST http://127.0.0.1:8000/ocr/ \
-H "Content-Type: application/json" \
-d '{"image_path": "systeme/data/images/doc1.png", "ocr_type": "doc"}'

"""
import subprocess
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class OCRRequest(BaseModel):
    image_path: str
    ocr_type: str

@app.post("/ocr/")
async def perform_ocr(request: OCRRequest):
    # Vérifier si le fichier existe
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image non trouvée.")

    try:
        # Définir le PYTHONPATH pour inclure le répertoire parent de 'systeme'
        env = os.environ.copy()
        env["PYTHONPATH"] = "/Users/jeevyaroun/Desktop/projet_OCR/Reseaux_neurones"

        # Utiliser subprocess pour appeler la fonction main via la ligne de commande
        command = [
            'python', 'systeme/scripts/ocr_script.py', 
            request.image_path, request.ocr_type
        ]

        # Exécuter la commande avec l'environnement modifié
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)

        # Retourner le texte obtenu depuis la sortie du script
        return {"text": result.stdout}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=e.stderr)
