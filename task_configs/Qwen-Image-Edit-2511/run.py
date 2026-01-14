import os
import sys
import json
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
prompt = sys.argv[1]
image_paths = json.loads(sys.argv[2])
output_path = sys.argv[3]
num_inference_steps = sys.argv[4]
guidance_scale = sys.argv[5]
seed = sys.argv[6]
true_cfg_scale = sys.argv[7]
negative_prompt = sys.argv[8]
num_images_per_prompt = sys.argv[9]

image_array = []
for image_path in image_paths:
    img_data = Image.open(image_path)
    if img_data is not None:
        image_array.append(img_data)

inputs = {
    "image": image_array,
    "prompt": prompt,
    "generator": torch.manual_seed(int(seed)),
    "true_cfg_scale": int(true_cfg_scale),
    "negative_prompt": negative_prompt,
    "num_inference_steps": int(num_inference_steps),
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save(output_path)
    print("image saved at", os.path.abspath(output_path))
