import os
import shutil
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from app.services.stitching_service import gerar_panorama_por_frames, extract_images

app = FastAPI(title="API Panorama 360° por Frames",
              description="Gera panorama 360° com blocos verticais sincronizados por índice",
              version="3.0")

UPLOAD_FOLDER = "temp_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/gerar_panorama_por_frames/", summary="Panorama com chao/meio/ceu sincronizados por índice")
def gerar_endpoint(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)

    imagens = []
    for file in files:
        filepath = os.path.join(session_folder, file.filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if filepath.lower().endswith(".zip"):
            imagens.extend(extract_images(filepath, session_folder))
        else:
            imagens.append(filepath)

    try:
        resultado_path = gerar_panorama_por_frames(imagens, OUTPUT_FOLDER)
        shutil.rmtree(session_folder, ignore_errors=True)
        return FileResponse(resultado_path, media_type="image/jpeg", filename="panorama_360_bloco.jpg")
    except Exception as e:
        shutil.rmtree(session_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
