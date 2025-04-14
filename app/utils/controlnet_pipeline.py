from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch
import numpy as np
import cv2
from PIL import Image

def gerar_panorama_com_controlnet(image_np, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet_tile = ControlNetModel.from_pretrained(
        "xinsir/controlnet-tile-sdxl-1.0",
        torch_dtype=torch.float16
    ).to(device)

    controlnet_depth = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[controlnet_tile, controlnet_depth],
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.set_progress_bar_config(disable=True)

    # Prepara imagem para os dois ControlNet
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).resize((1024, 1024))

    # Chamada
    result = pipe(
        prompt=prompt,
        negative_prompt="distortion, bad stitching, duplicate objects, blurry",
        image=[image, image],  # Tile + Depth
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)