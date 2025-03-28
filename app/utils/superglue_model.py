# app/utils/superglue_model.py
import cv2
import numpy as np

def match_images(img1, img2):
    """
    Alinha img2 com base na img1 usando correspondência de pontos com ORB e Homografia.

    - Extraímos keypoints com ORB
    - Casamos os descritores com BFMatcher
    - Filtramos os melhores matches
    - Estimamos uma matriz de homografia (transformação)
    - Aplicamos warpPerspective para alinhar
    """
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        return img2

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        h, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(img2, h, (img1.shape[1], img1.shape[0]))
        return aligned
    else:
        return img2