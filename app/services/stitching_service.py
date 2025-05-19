import zipfile
import os
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import requests
import re

from app.services.stitching_core import stitch_images
from app.utils.image_processing import enhance_image

register_heif_opener()
cv2.ocl.setUseOpenCL(False)
cv2.setUseOptimized(True)

MAX_WIDTH = 1920
MAX_HEIGHT = 1080

def extract_images(zip_path, extract_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    return sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(extract_folder)
        for f in files
        if f.lower().endswith(('jpg', 'jpeg', 'png', 'heic'))
    ])

def redimensionar_imagem(imagem, max_largura=MAX_WIDTH, max_altura=MAX_HEIGHT):
    altura, largura = imagem.shape[:2]
    if largura > max_largura or altura > max_altura:
        fator = min(max_largura / largura, max_altura / altura)
        nova_largura = int(largura * fator)
        nova_altura = int(altura * fator)
        return cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return imagem

def gerar_mascara_bordas_pretas(image_np, tolerancia=10):
    return cv2.inRange(image_np, (0, 0, 0), (tolerancia, tolerancia, tolerancia))

def is_inside_docker():
    return os.path.exists("/.dockerenv")

def get_lama_url():
    return "http://lama:8081/inpaint" if is_inside_docker() else "http://localhost:8081/inpaint"

def call_lama_cleaner(image_np):
    _, img_encoded = cv2.imencode(".jpg", image_np)
    mask = gerar_mascara_bordas_pretas(image_np)
    _, mask_encoded = cv2.imencode(".jpg", mask)

    files = {
        "image": ("image.jpg", img_encoded.tobytes(), "image/jpeg"),
        "mask": ("mask.jpg", mask_encoded.tobytes(), "image/jpeg")
    }

    data = {
        "ldmSteps": "20",
        "ldmSampler": "plms",
        "hdStrategy": "Original",
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
        "sdStrength": "1.0",
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

    response = requests.post(get_lama_url(), files=files, data=data)
    if response.status_code != 200:
        raise Exception(f"Erro ao chamar lama-cleaner: {response.text}")

    return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

def ajustar_proporcao_equiretangular(imagem):
    largura = imagem.shape[1]
    nova_altura = largura // 2
    return cv2.resize(imagem, (largura, nova_altura), interpolation=cv2.INTER_AREA)

def limitar_resolucao(imagem, max_largura=23000, max_altura=11500):
    altura, largura = imagem.shape[:2]
    if largura > max_largura or altura > max_altura:
        escala = min(max_largura / largura, max_altura / altura)
        nova_largura = int(largura * escala)
        nova_altura = int(altura * escala)
        return cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return imagem

def gerar_panorama_por_frames(image_paths, output_folder):
    grupos = {}
    for path in image_paths:
        filename = os.path.basename(path).lower()
        match = re.match(r"(chao|meio|ceu)[^0-9]*([0-9]+)", filename)
        if match:
            faixa, indice = match.groups()
            if indice not in grupos:
                grupos[indice] = {}
            grupos[indice][faixa] = path

    blocos = []
    for indice in sorted(grupos.keys()):
        bloco = grupos[indice]
        if not all(k in bloco for k in ['chao', 'meio', 'ceu']):
            continue

        imgs = [cv2.imread(bloco['chao']),
                cv2.imread(bloco['meio']),
                cv2.imread(bloco['ceu'])]

        if any(i is None for i in imgs):
            continue

        largura_min = min(i.shape[1] for i in imgs)
        imgs_redimensionadas = [
            cv2.resize(i, (largura_min, int(i.shape[0] * largura_min / i.shape[1])))
            for i in imgs
        ]

        bloco_vertical = cv2.vconcat(imgs_redimensionadas)
        blocos.append(bloco_vertical)

    if len(blocos) < 2:
        raise Exception("Blocos verticais insuficientes para gerar panorama.")

    # 🔧 Código corrigido aqui:
    panorama = stitch_images(blocos)  # costura via stitching_core
    panorama = enhance_image(panorama)  # realce antes do lama-cleaner
    panorama = call_lama_cleaner(panorama)  # mantém lama-cleaner normal
    panorama = ajustar_proporcao_equiretangular(panorama)
    panorama = limitar_resolucao(panorama)

    resultado_path = os.path.join(output_folder, "panorama_360_bloco.jpg")
    cv2.imwrite(resultado_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return resultado_path
