import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = "cuda"

# =========================================================
# MODEL
# =========================================================

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined",
    torch_dtype=torch.float16
).to(device)

# =========================================================
# PATHS
# =========================================================

IMAGE_DIR = "seq16/Images"
SAVE_DIR  = "seq16/Prediction528"

os.makedirs(SAVE_DIR, exist_ok=True)

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png") and os.path.isfile(os.path.join(IMAGE_DIR, f))
]
image_files.sort()
image_files = [f for f in image_files if not os.path.exists(os.path.join(SAVE_DIR, f))]



# =========================================================
# PROMPTS
# =========================================================

building_prompts = ["building"]
road_prompts     = ["street"]
car_prompts      = ["car"]
tree_prompts     = ["trees"]
veg_prompts      = ["low vegetation"]
human_prompts    = ["human"]
bg_prompts       = ["background clutter"]
"""
building_prompts = ["building", "roof", "house", "flat roof", "terace roof", "drone view of roof"]
road_prompts     = ["asphalt road", "street", "road", "road surface", "driving lane"]
car_prompts      = ["car"]
tree_prompts     = ["tree", "tree canopy"]
veg_prompts      = ["grass", "low vegetation"]
human_prompts    = ["pedestrian", "human"]
bg_prompts       = ["background clutter", "sidewalk", "pavement",
                    "footpath", "parking lot", "square", "lantern", "sign", "stopsign", "walkway", "promenade"]
"""
prompt_groups = [
    building_prompts, road_prompts, car_prompts,
    tree_prompts, veg_prompts, human_prompts,
    bg_prompts,
]

prompts_flat = [p for group in prompt_groups for p in group]

class_names = [
    "Building", "Road", "Car",
    "Tree", "Low Vegetation", "Human",
    "Background Clutter",
]

UAVID_COLOURS = {
    "Building":           (128, 0, 0),
    "Road":               (128, 64, 128),
    "Car":                (192, 0, 192),
    "Tree":               (0, 128, 0),
    "Low Vegetation":     (128, 128, 0),
    "Human":              (64, 64, 0),
    "Background Clutter": (0, 0, 0),    
}

# =========================================================
# SEGMENTATION PARAMETERS
# =========================================================

scales = [528]
batch_size = 8
  
# =========================================================
# PROCESS ALL IMAGES
# =========================================================

for image_name in image_files:

    print(f"\nProcessing {image_name}")

    image_path = os.path.join(IMAGE_DIR, image_name)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    final_preds = torch.zeros((len(prompts_flat), h, w), dtype=torch.float32)
    weight_sums = torch.zeros((len(prompts_flat), h, w), dtype=torch.float32)

    for patch_size in scales:

        stride = int(patch_size * 1)

        xs = list(range(0, w - patch_size, stride)) + [w - patch_size]
        ys = list(range(0, h - patch_size, stride)) + [h - patch_size]

        xs = [max(0, x) for x in xs]
        ys = [max(0, y) for y in ys]

        patch_coords = list(dict.fromkeys(
            (x, y, x + patch_size, y + patch_size)
            for x in xs for y in ys
        ))

        cy, cx = patch_size // 2, patch_size // 2
        gy = torch.exp(-0.5 * ((torch.arange(patch_size).float() - cy) / (patch_size * 0.35)) ** 2)
        gx = torch.exp(-0.5 * ((torch.arange(patch_size).float() - cx) / (patch_size * 0.35)) ** 2)
        gaussian = (gy.unsqueeze(1) * gx.unsqueeze(0))

        with tqdm(total=len(patch_coords), desc=f"{image_name}") as pbar:

            for i in range(0, len(patch_coords), batch_size):

                batch_coords_sub = patch_coords[i : i + batch_size]
                batch_images = [image.crop(coords) for coords in batch_coords_sub]

                texts = prompts_flat * len(batch_images)
                images_input = [img for img in batch_images for _ in range(len(prompts_flat))]

                inputs = processor(
                    text=texts,
                    images=images_input,
                    padding=True,
                    return_tensors="pt"
                ).to(device)

                inputs = {
                    k: v.to(device, dtype=torch.float16) if v.is_floating_point() else v.to(device)
                    for k, v in inputs.items()
                }

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits.view(len(batch_images), len(prompts_flat), 352, 352)
                logits = logits.cpu().to(torch.float32)

                if patch_size != 352:
                    logits = F.interpolate(
                        logits,
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False
                    )

                for idx, (x1, y1, x2, y2) in enumerate(batch_coords_sub):

                    actual_h = min(y2, h) - y1
                    actual_w = min(x2, w) - x1

                    final_preds[:, y1:y1+actual_h, x1:x1+actual_w] += \
                        logits[idx, :, :actual_h, :actual_w] * gaussian[:actual_h, :actual_w].unsqueeze(0)

                    weight_sums[:, y1:y1+actual_h, x1:x1+actual_w] += \
                        gaussian[:actual_h, :actual_w].unsqueeze(0)

                pbar.update(len(batch_coords_sub))

    # =========================================================
    # POST PROCESS
    # =========================================================

    weight_sums[weight_sums < 1e-8] = 1.0
    final_preds = final_preds / weight_sums

    final_probs = torch.sigmoid(final_preds)

    class_probs = []
    idx = 0

    for group in prompt_groups:
        group_probs = final_probs[idx : idx + len(group)]
        class_probs.append(group_probs.max(dim=0).values)
        idx += len(group)

    class_probs = torch.stack(class_probs, dim=0)

    class_probs[2] *= 1.5
    class_probs[5] *= 1.5

    class_probs = torch.softmax(class_probs * 4.0, dim=0)

    # smoothing
    class_probs_smooth = []

    for i in range(class_probs.shape[0]):
        t = class_probs[i].unsqueeze(0).unsqueeze(0)
        t = F.avg_pool2d(t, kernel_size=5, stride=1, padding=2)
        class_probs_smooth.append(t.squeeze())

    class_probs_smooth = torch.stack(class_probs_smooth, dim=0)

    winner_map = class_probs_smooth.argmax(dim=0).numpy()

    # =========================================================
    # SAVE UAVid FORMAT MASK
    # =========================================================

    label_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, name in enumerate(class_names):
        rgb = UAVID_COLOURS[name]
        label_mask[winner_map == class_idx] = rgb

    save_path = os.path.join(SAVE_DIR, image_name)

    Image.fromarray(label_mask).save(save_path)

    print(f"Saved → {save_path}")

print("\nFinished processing all images.")