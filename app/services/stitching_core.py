import cv2
import numpy as np

def stitch_images(imagens):
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(imagens)
    
    if status != cv2.Stitcher_OK:
        raise Exception(f"Erro ao costurar imagens: Código {status}")
    
    return panorama

def resize_to_width(image, target_width):
    altura, largura = image.shape[:2]
    fator = target_width / largura
    nova_altura = int(altura * fator)
    return cv2.resize(image, (target_width, nova_altura), interpolation=cv2.INTER_AREA)
