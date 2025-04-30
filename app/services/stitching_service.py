import os
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
import zipfile

register_heif_opener()
cv2.ocl.setUseOpenCL(False)     # Desativa uso de OpenCL (GPU)
cv2.setUseOptimized(True)       # Ativa otimizações padrão

MAX_WIDTH = 1920  # largura máxima para redimensionamento
MAX_HEIGHT = 1080

def extract_images(zip_path, extract_folder):
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

    if status == cv2.Stitcher_OK:
        resultado_path = os.path.join(output_folder, "panorama_resultado.jpg")
        cv2.imwrite(resultado_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return resultado_path
    else:
        raise Exception(f"Erro ao criar panorama. Código: {status}")
