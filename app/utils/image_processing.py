import cv2
import numpy as np

def enhance_image(image):
    enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    alpha = 1.2
    beta = 10
    return cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
