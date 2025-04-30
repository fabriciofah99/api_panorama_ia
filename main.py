# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import os
import shutil
from fastapi.responses import FileResponse
from app.services.stitching_service import generate_panorama, extract_images

app = FastAPI(title="API de Montagem de Imagem 360° com OpenCV",
              description="Recebe imagens e gera uma imagem panorâmica 360° com OpenCV puro",
              version="1.0")

UPLOAD_FOLDER = "temp_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/gerar_panorama/", summary="Gerar Panorama com OpenCV")
def gerar_panorama_endpoint(files: List[UploadFile] = File(...)):
    imagens = []

    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.endswith(".zip"):
            imagens.extend(extract_images(filepath, UPLOAD_FOLDER))
        else:
            imagens.append(filepath)

    if len(imagens) < 2:
        raise HTTPException(status_code=400, detail="Pelo menos duas imagens são necessárias.")

    try:
        resultado_path = generate_panorama(imagens, OUTPUT_FOLDER)
        return FileResponse(resultado_path, media_type="image/jpeg", filename="panorama_resultado.jpg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
