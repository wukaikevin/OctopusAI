from modelscope import QwenImageLayeredPipeline
import torch
from PIL import Image
import sys

source = sys.argv[1]
output_dir = sys.argv[2]

pipeline = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered")
pipeline = pipeline.to("cuda", torch.bfloat16)
pipeline.set_progress_bar_config(disable=None)

image = Image.open(source).convert("RGBA")
inputs = {
    "image": image,
    "generator": torch.Generator(device='cuda').manual_seed(777),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "num_images_per_prompt": 1,
    "layers": 4,
    "resolution": 640,      # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
    "cfg_normalize": True,  # Whether enable cfg normalization.
    "use_en_prompt": True,  # Automatic caption language if user does not provide caption
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]

for i, image in enumerate(output_image):
    image.save(f"{output_dir}/{i}.png")