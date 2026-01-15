import os
import sys
import json
import torch
import filetype
from PIL import Image
try:
    from pillow_heif import register_heif_opener, open_heif
    register_heif_opener()
    print("HEIF/AVIF support registered via pillow-heif")
    HAS_HEIF = True
except ImportError:
    print("Warning: pillow-heif not available")
    HAS_HEIF = False
    open_heif = None

# Try OpenCV as backup for AVIF
try:
    import cv2
    HAS_OPENCV = True
    print("OpenCV available as backup for image loading")
except ImportError:
    HAS_OPENCV = False
    print("OpenCV not available")

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
    print(f"Loading image: {image_path}")
    # Detect actual file type
    kind = filetype.guess(image_path)
    if kind is not None:
        print(f"Detected file type: {kind.mime} ({kind.extension})")
    else:
        print("Could not detect file type, trying to open as image...")

    try:
        img_data = None
        # Try different methods based on file type
        if kind and kind.extension in ['heif', 'heic', 'avif']:
            print(f"Detected {kind.extension.upper()} format, attempting to load...")
            loaded = False

            # Method 1: Try pillow-heif first
            if HAS_HEIF and open_heif:
                try:
                    print(f"  Trying pillow-heif...")
                    heif_file = open_heif(image_path)
                    img_data = heif_file.to_pillow()
                    print(f"  Successfully loaded via pillow-heif")
                    loaded = True
                except RuntimeError as e:
                    if "Support for this compression format has not been built in" in str(e):
                        print(f"  pillow-heif lacks codec support, trying alternative methods...")
                    else:
                        raise

            # Method 2: Try OpenCV as backup if pillow-heif failed
            if not loaded and HAS_OPENCV:
                try:
                    print(f"  Trying OpenCV...")
                    # OpenCV reads as BGR, need to convert to RGB
                    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if img_bgr is not None:
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        # Convert to PIL Image
                        img_data = Image.fromarray(img_rgb)
                        print(f"  Successfully loaded via OpenCV")
                        loaded = True
                    else:
                        print(f"  OpenCV could not read the file")
                except Exception as e:
                    print(f"  OpenCV failed: {e}")

            if not loaded:
                raise Exception(f"Unable to load {kind.extension.upper()} file - no working codec found")

        else:
            # Standard PIL open for other formats
            img_data = Image.open(image_path)

        if img_data is not None:
            print(f"Successfully loaded image: {image_path}, mode: {img_data.mode}, size: {img_data.size}")
            # Convert to RGB if necessary
            if img_data.mode != 'RGB':
                img_data = img_data.convert('RGB')
                print(f"Converted image to RGB mode")
            image_array.append(img_data)
        else:
            print(f"Warning: Could not load image {image_path}")
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        # Try to get file info for debugging
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            print(f"File exists, size: {file_size} bytes")
        raise

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
