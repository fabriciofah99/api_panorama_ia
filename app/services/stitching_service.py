import zipfile
import cv2
import os
import numpy as np
from app.utils.image_processing import enhance_image, inpaint_missing_areas
from app.utils.superglue_model import match_images

def extract_images(zip_path, extract_folder):
    """Extrai imagens de um ZIP e converte HEIC para JPG automaticamente, se necessário."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    image_files = [os.path.join(extract_folder, f) for f in os.listdir(extract_folder) if f.endswith(('jpg', 'jpeg', 'png'))]

    # Ordenar corretamente os arquivos de imagem
    image_files = sorted(
        image_files,
        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
    )

    return image_files

def generate_panorama(image_paths, output_folder):
    """Gera uma imagem panorâmica 360° a partir das imagens fornecidas."""
    
    images = [cv2.imread(img) for img in image_paths]

    # 🔹 Etapa 1: Alinhar imagens com SuperGlue antes do stitching
    aligned_images = []
    for i in range(len(images) - 1):
        aligned_img = match_images(images[i], images[i + 1])
        aligned_images.append(aligned_img)
    aligned_images.append(images[-1])

    # 🔹 Etapa 2: Criar o stitcher e configurar parâmetros
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    stitcher.setPanoConfidenceThresh(0.2)  # Ajusta a confiança mínima

    # 🔹 Etapa 3: Primeiro Stitching
    status, panorama = stitcher.stitch(aligned_images)

    if status != cv2.Stitcher_OK:
        raise Exception("Falha ao gerar a primeira imagem 360°.")

    # 🔹 Etapa 4: Aplicar melhorias de IA
    panorama = enhance_image(panorama)

    # 🔹 Etapa 5: Preenchimento Inteligente de Falhas
    panorama_filled = inpaint_missing_areas(panorama)

    # 🔹 Etapa 6: Segundo Stitching com a imagem final + originais
    final_images = [panorama_filled] + images
    status, final_panorama = stitcher.stitch(final_images)

    if status != cv2.Stitcher_OK:
        raise Exception("Falha ao gerar a imagem final 360°.")