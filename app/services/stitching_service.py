# app/services/stitching_service.py
import zipfile
import os
import cv2
import numpy as np
import requests
from app.utils.image_processing import enhance_image
from app.utils.superglue_model import match_images

def extract_images(zip_path, extract_folder):
    """Extrai imagens de um ZIP e ordena numericamente"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    image_files = [os.path.join(extract_folder, f)
                   for f in os.listdir(extract_folder)
                   if f.endswith(('jpg', 'jpeg', 'png'))]

    return sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

def call_lama_cleaner(image_np):
    """
    Envia imagem para o lama-cleaner usando os campos esperados pela rota /inpaint:
    - image: imagem original
    - mask: mesma imagem como "máscara completa" (branco = 255)
    - outros parâmetros obrigatórios do form, mesmo que vazios/default
    """
    _, img_encoded = cv2.imencode(".jpg", image_np)
    image_bytes = img_encoded.tobytes()

    # Criar uma máscara completamente branca (255) para preencher tudo
    mask = np.ones_like(image_np[:, :, 0], dtype=np.uint8) * 255
    _, mask_encoded = cv2.imencode(".jpg", mask)

    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg"),
        "mask": ("mask.jpg", mask_encoded.tobytes(), "image/jpeg")
    }

    data = {
        "ldmSteps": "20",
        "ldmSampler": "plms",
        "hdStrategy": "Crop",
        "zitsWireframe": "false",
        "hdStrategyCropMargin": "32",
        "hdStrategyCropTrigerSize": "800",
        "hdStrategyResizeLimit": "2048",
        "prompt": "",
        "negativePrompt": "",
        "useCroper": "false",
        "croperX": "0",
        "croperY": "0",
        "croperHeight": "0",
        "croperWidth": "0",
        "sdScale": "1.0",
        "sdMaskBlur": "4",
        "sdStrength": "0.75",
        "sdSteps": "20",
        "sdGuidanceScale": "7.5",
        "sdSampler": "ddim",
        "sdSeed": "-1",
        "sdMatchHistograms": "false",
        "cv2Flag": "INPAINT_TELEA",
        "cv2Radius": "3",
        "paintByExampleSteps": "20",
        "paintByExampleGuidanceScale": "7.5",
        "paintByExampleMaskBlur": "4",
        "paintByExampleSeed": "-1",
        "paintByExampleMatchHistograms": "false",
        "p2pSteps": "20",
        "p2pImageGuidanceScale": "1.5",
        "p2pGuidanceScale": "7.5",
        "controlnet_conditioning_scale": "1.0",
        "controlnet_method": ""
    }

    response = requests.post("http://localhost:8081/inpaint", files=files, data=data)
    if response.status_code != 200:
        raise Exception(f"Erro ao chamar lama-cleaner: {response.text}")

    nparr = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def crop_black_borders(img):
    """Remove as bordas pretas do panorama usando threshold e bounding box."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w]
    return img

def generate_panorama(image_paths, output_folder):
    """Gera panorama 360 com alinhamento, IA e preenchimento"""
    images = [cv2.imread(img) for img in image_paths]

    # Etapa 1: Alinhamento com ORB + Homografia
    aligned_images = [images[0]]
    for i in range(1, len(images)):
        aligned = match_images(aligned_images[-1], images[i])
        aligned_images.append(aligned)

    # Etapa 2: Stitching inicial
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, panorama = stitcher.stitch(aligned_images)
    if status != cv2.Stitcher_OK:
        raise Exception("Falha ao gerar panorama inicial.")

    # Etapa 3: Melhorias visuais com IA
    panorama = enhance_image(panorama)

    # Etapa 4: Preenchimento com lama-cleaner (IA generativa)
    panorama = call_lama_cleaner(panorama)

    # Etapa 5: Stitching final com panorama + originais
    final_images = [panorama] + images
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, final_panorama = stitcher.stitch(final_images)
    if status != cv2.Stitcher_OK:
        raise Exception("Falha ao gerar panorama final.")

    # Etapa 6: Remoção de bordas pretas
    final_panorama = crop_black_borders(final_panorama)

    # Etapa 7: Salva imagem final
    resultado_path = os.path.join(output_folder, "panorama.jpg")
    cv2.imwrite(resultado_path, final_panorama)
    
    # Etapa 8: Limpeza dos arquivos temporários
    for f in os.listdir("temp_images"):
        try:
            file_path = os.path.join("temp_images", f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Erro ao remover arquivo temporário {f}: {str(e)}")
    
    return resultado_path