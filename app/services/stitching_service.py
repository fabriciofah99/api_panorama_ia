import os
import cv2
import numpy as np
import zipfile
from app.utils.controlnet_pipeline import gerar_panorama_com_controlnet
from app.utils.blip_prompt import gerar_prompt_blip

def extract_images(zip_path, extract_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    return sorted(
        [os.path.join(extract_folder, f)
         for f in os.listdir(extract_folder)
         if f.lower().endswith(('jpg', 'jpeg', 'png', 'heic'))],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
    )

def montar_canvas(imagens, margem=40):
    alturas = [img.shape[0] for img in imagens]
    larguras = [img.shape[1] for img in imagens]
    altura = max(alturas)
    largura = sum(larguras) + margem * (len(imagens) - 1)
    canvas = np.zeros((altura, largura, 3), dtype=np.uint8)

    x = 0
    for i, img in enumerate(imagens):
        h, w = img.shape[:2]
        canvas[:h, x:x+w] = img
        x += w + margem
    return canvas

def generate_panorama(image_paths, output_folder):
    imagens = []
    for path in image_paths:
        if path.lower().endswith(".zip"):
            imagens.extend(extract_images(path, "temp_images"))
        else:
            imagens.append(path)

    images = [cv2.imread(p) for p in imagens if cv2.imread(p) is not None]
    if len(images) < 2:
        raise Exception("Mínimo de duas imagens necessárias.")

    canvas = montar_canvas(images)
    prompt = gerar_prompt_blip(canvas)

    resultado_np = gerar_panorama_com_controlnet(canvas, prompt)

    resultado_path = os.path.join(output_folder, "panorama_controlnet.jpg")
    cv2.imwrite(resultado_path, resultado_np)

    # Limpeza da pasta temporária
    for f in os.listdir("temp_images"):
        try:
            os.remove(os.path.join("temp_images", f))
        except Exception as e:
            print(f"Erro ao limpar {f}: {e}")

    return resultado_path