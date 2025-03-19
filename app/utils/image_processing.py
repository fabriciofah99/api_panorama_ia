import cv2
import numpy as np

def enhance_image(image):
    """Aplica melhorias na imagem usando técnicas de IA."""

    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    alpha = 1.2  # Contraste
    beta = 10    # Brilho
    adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

    return adjusted

def inpaint_missing_areas(image):
    """Preenche lacunas na imagem usando OpenCV Inpainting."""

    mask = cv2.inRange(image, (0, 0, 0), (5, 5, 5))  # Criar máscara para detectar áreas pretas

    # Aplicar Inpainting (Testamos dois métodos)
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return inpainted_image