from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import os
import shutil
import zipfile

from fastapi.responses import FileResponse
from app.services.stitching_service import generate_panorama, extract_images

app = FastAPI(title="API de Montagem de Imagem 360° com IA",
              description="Recebe imagens e gera uma imagem panorâmica 360° com técnicas de IA",
              version="2.0")

UPLOAD_FOLDER = "temp_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/gerar_panorama/", summary="Gerar Imagem 360° com IA")
def gerar_panorama(files: List[UploadFile] = File(...)):
    """Recebe imagens e gera um panorama 360° com alinhamento aprimorado por IA."""
    imagens = []
    
    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Se for um ZIP, extrair as imagens
        if file.filename.endswith(".zip"):
            imagens.extend(extract_images(filepath, UPLOAD_FOLDER))
        else:
            imagens.append(filepath)
    
    if len(imagens) < 2:
        raise HTTPException(status_code=400, detail="É necessário pelo menos duas imagens para gerar um panorama.")
    
    try:
        resultado_path = generate_panorama(imagens, OUTPUT_FOLDER)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem com IA: {str(e)}")
    
    return FileResponse(resultado_path, media_type="image/jpeg", filename="panorama.jpg")

# Executar com: uvicorn main:app --reload