from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
import torch
from PIL import Image

background = Image.open("background.jpg")
mask = Image.open("mask_image.png")
control_pose = Image.open("pose_image.png")

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")

output = pipe(
        prompt="A man weave his hand",
        negative_prompt="",
        num_inference_steps=25,
        strength=0.98,
        guidance_scale=16,

        image=background,
        control_image=control_pose,
        mask_image=mask,
    ).images[0]
    
output.save("output.jpg")