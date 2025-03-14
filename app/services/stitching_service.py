import zipfile
import cv2
import os
import numpy as np
from app.utils.image_processing import enhance_image

def extract_images(zip_path, extract_folder):
    """Extrai imagens de um arquivo ZIP para um diretório e ordena corretamente os arquivos."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    # Listar e ordenar corretamente os arquivos de imagem
    image_files = sorted(
        [os.path.join(extract_folder, f) for f in os.listdir(extract_folder) if f.endswith(('jpg', 'png', 'jpeg'))],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
    )

    return image_files

def generate_panorama(image_paths, output_folder):
    """Gera uma imagem panorâmica 360° a partir das imagens fornecidas."""
    
    images = [cv2.imread(img) for img in image_paths]
    
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)  # AGORA panorama está definido antes de usar!

    if status != cv2.Stitcher_OK:
        raise Exception("Falha ao gerar a imagem 360°. As imagens podem estar desalinhadas.")

    # Aplicar correção de perspectiva APÓS a geração do panorama
    panorama = cv2.warpPerspective(panorama, np.eye(3), (panorama.shape[1], panorama.shape[0]))

    # Aplicar melhorias de IA na imagem final
    panorama = enhance_image(panorama)

    resultado_path = os.path.join(output_folder, "panorama.jpg")
    cv2.imwrite(resultado_path, panorama)
    
    return resultado_path