# app/services/stitching_service.py
import zipfile
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
    """Extrai imagens de um ZIP e ordena numericamente"""
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

def call_lama_cleaner(image_np):
    """
    Envia imagem para o lama-cleaner usando os campos esperados pela rota /inpaint:
    - image: imagem original
    - mask: mesma imagem como "máscara completa" (branco = 255)
    - outros parâmetros obrigatórios do form, mesmo que vazios/default
    """
    _, img_encoded = cv2.imencode(".jpg", image_np)
    image_bytes = img_encoded.tobytes()

    # Criar uma máscara completamente branca (255) para preencher tudo
    mask = np.ones_like(image_np[:, :, 0], dtype=np.uint8) * 255
    _, mask_encoded = cv2.imencode(".jpg", mask)

    files = {
        "image": ("image.jpg", image_bytes, "image/jpeg"),
        "mask": ("mask.jpg", mask_encoded.tobytes(), "image/jpeg")
    }

    data = {
        "ldmSteps": "20",
        "ldmSampler": "plms",
        "hdStrategy": "Crop",
        "zitsWireframe": "false",
        "hdStrategyCropMargin": "32",
        "hdStrategyCropTrigerSize": "800",
        "hdStrategyResizeLimit": "2048",
        "prompt": "",
        "negativePrompt": "",
        "useCroper": "false",
        "croperX": "0",
        "croperY": "0",
        "croperHeight": "0",
        "croperWidth": "0",
        "sdScale": "1.0",
        "sdMaskBlur": "4",
        "sdStrength": "0.75",
        "sdSteps": "20",
        "sdGuidanceScale": "7.5",
        "sdSampler": "ddim",
        "sdSeed": "-1",
        "sdMatchHistograms": "false",
        "cv2Flag": "INPAINT_TELEA",
        "cv2Radius": "3",
        "paintByExampleSteps": "20",
        "paintByExampleGuidanceScale": "7.5",
        "paintByExampleMaskBlur": "4",
        "paintByExampleSeed": "-1",
        "paintByExampleMatchHistograms": "false",
        "p2pSteps": "20",
        "p2pImageGuidanceScale": "1.5",
        "p2pGuidanceScale": "7.5",
        "controlnet_conditioning_scale": "1.0",
        "controlnet_method": ""
    }

    response = requests.post("http://localhost:8081/inpaint", files=files, data=data)
    if response.status_code != 200:
        raise Exception(f"Erro ao chamar lama-cleaner: {response.text}")

    nparr = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def crop_black_borders(img):
    """Remove as bordas pretas do panorama usando threshold e bounding box."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return img[y:y+h, x:x+w]
    return img

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
