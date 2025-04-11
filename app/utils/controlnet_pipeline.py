from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import numpy as np
import cv2
from PIL import Image

def gerar_panorama_com_controlnet(image_np, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet_tile = ControlNetModel.from_pretrained(
        "lllyasviel/controlnet-tile-sdxl-1.0",
        torch_dtype=torch.float16
    ).to(device)

    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/controlnet-depth-sdxl-1.0",
        torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=[controlnet_tile, controlnet_depth],
        torch_dtype=torch.float16
    ).to(device)

    pipe.enable_xformers_memory_efficient_attention()

    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).resize((1024, 512))

    result = pipe(
        prompt=prompt,
        negative_prompt="distortion, bad stitching, duplicate objects, blurry",
        image=[image, image],
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)