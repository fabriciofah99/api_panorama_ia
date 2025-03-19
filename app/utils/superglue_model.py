import cv2
import numpy as np

def match_images(img1, img2):
    """Usa SuperGlue para melhorar o alinhamento das imagens."""
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)  # Aumentamos a quantidade de features
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        return img2  # Retorna imagem original se não encontrar pontos

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)[:50]  # Mantemos apenas os 50 melhores

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned_img = cv2.warpPerspective(img2, h, (img1.shape[1], img1.shape[0]))
        return aligned_img
    else:
        return img2  # Retorna imagem original se não encontrar boa correspondência