from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import numpy as np

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def gerar_prompt_blip(np_img):
    image = Image.fromarray(cv2rgb(np_img))
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

def cv2rgb(image):
    return image[:, :, ::-1]  # BGR to RGB