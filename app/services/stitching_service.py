# app/services/stitching_service.py
import zipfile
import os
import cv2
from PIL import Image
import numpy as np
from pillow_heif import register_heif_opener
import zipfile

import requests

register_heif_opener()
cv2.ocl.setUseOpenCL(False)     # Desativa uso de OpenCL (GPU)
cv2.setUseOptimized(True)       # Ativa otimizações padrão

MAX_WIDTH = 1920  # largura máxima para redimensionamento
MAX_HEIGHT = 1080

def extract_images(zip_path, extract_folder):
    """Extrai imagens de um ZIP e ordena numericamente"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    return sorted([
        os.path.join(extract_folder, f)
        for f in os.listdir(extract_folder)
        if f.lower().endswith(('jpg', 'jpeg', 'png', 'heic'))
    ])

def redimensionar_imagem(imagem, max_largura, max_altura):
    altura, largura = imagem.shape[:2]
    if largura > max_largura or altura > max_altura:
        fator = min(max_largura / largura, max_altura / altura)
        nova_largura = int(largura * fator)
        nova_altura = int(altura * fator)
        return cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    return imagem

def call_lama_cleaner(image_np):
    """
    Envia imagem para o lama-cleaner usando os campos esperados pela rota /inpaint:
    - image: imagem original
    - mask: mesma imagem como "máscara completa" (branco = 255)
    - outros parâmetros obrigatórios do form, mesmo que vazios/default
    """
    _, img_encoded = cv2.imencode(".jpg", image_np)
    image_bytes = img_encoded.tobytes()

    mask = gerar_mascara_bordas_pretas(image_np)
    _, mask_encoded = cv2.imencode(".jpg", mask)

    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg"),
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
        "prompt": "preencher suavemente as bordas pretas com o mesmo estilo da imagem",
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

    nparr = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def is_inside_docker():
    return os.path.exists("/.dockerenv")

def get_lama_url():
    if is_inside_docker():
        return "http://lama:8081/inpaint"  # dentro do Docker, usa o nome do serviço
    return "http://localhost:8081/inpaint"  # fora do Docker (execução local)

def gerar_mascara_bordas_pretas(image_np, tolerancia=10):
    """Cria uma máscara onde as bordas pretas da imagem serão 255 (preencher), o restante 0"""
    # Cria uma máscara onde os pixels pretos ou quase pretos (com tolerância) são marcados como 255
    mask = cv2.inRange(image_np, (0, 0, 0), (tolerancia, tolerancia, tolerancia))
    return mask

def generate_panorama(image_paths, output_folder):
    pasta_convertidos = os.path.join(output_folder, 'convertidos')
    os.makedirs(pasta_convertidos, exist_ok=True)
    caminhos_imagens = []

    for caminho_arquivo in sorted(image_paths):
        nome_arquivo = os.path.basename(caminho_arquivo)
        nome_base, extensao = os.path.splitext(nome_arquivo)
        extensao = extensao.lower()

        if extensao == '.heic':
            try:
                imagem = Image.open(caminho_arquivo)
                novo_caminho = os.path.join(pasta_convertidos, f"{nome_base}.jpg")
                imagem.save(novo_caminho, "JPEG")
                caminhos_imagens.append(novo_caminho)
            except Exception as e:
                print(f"Erro ao converter {nome_arquivo}: {e}")
        elif extensao in ['.jpg', '.jpeg', '.png']:
            caminhos_imagens.append(caminho_arquivo)

    imagens = []
    for caminho in caminhos_imagens:
        if os.path.exists(caminho):
            img = cv2.imread(caminho)
            if img is not None:
                img = redimensionar_imagem(img, MAX_WIDTH, MAX_HEIGHT)
                imagens.append(img)

    if len(imagens) < 2:
        raise Exception("Imagens insuficientes para criar panorama.")

    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(imagens)
    
    panorama = call_lama_cleaner(panorama)

    if status == cv2.Stitcher_OK:
        resultado_path = os.path.join(output_folder, "panorama_resultado.jpg")
        cv2.imwrite(resultado_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return resultado_path
    else:
        raise Exception(f"Erro ao criar panorama. Código: {status}")
