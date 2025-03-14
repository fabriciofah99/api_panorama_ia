import cv2
import numpy as np

def enhance_image(image):
    """Aplica melhorias na imagem usando técnicas de IA."""
    
    # Melhorar detalhes usando filtragem bilateral
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    
    # Aplicar um ajuste de nitidez
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Ajuste de contraste e brilho
    alpha = 1.2  # Contraste
    beta = 10    # Brilho
    adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    
    return adjusted