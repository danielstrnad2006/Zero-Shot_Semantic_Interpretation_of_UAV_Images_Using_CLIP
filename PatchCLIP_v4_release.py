import torch
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
import os
from support.scoring_general import save_and_evaluate_single_image
import time 
import pandas as pd

# --- 4070 CONTEXT INITIALIZATION ---
if torch.cuda.is_available():
    # [AI]: Warm up and select the CUDA device early to reduce startup
    # [AI]: latency and ensure deterministic device assignment for tensors.
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.ones(1, device="cuda") @ torch.ones(1, device="cuda")

def segment_mask_to_rgb(global_seg_map, labels_order):
    # [AI]: Keep this mapping local to the visualisation code so that
    # [AI]: saved prediction images match the dataset colour convention
    # [AI]: and downstream evaluation tools can read them back reliably.
    """Converts a 2D segmentation map of integer indices into a 3D RGB image array."""
    uavid_gt_colors = {
        "building":    [128, 0, 0],
        "road":        [128, 64, 128],
        "tree":        [0, 128, 0],
        "low_veg":     [128, 128, 0],
        "clutter":     [0, 0, 0],
        "car":         [192, 0, 192],
        "human":       [64, 64, 0]
    }
    num_classes = len(labels_order)
    color_matrix = np.zeros((num_classes, 3), dtype=np.uint8)
    for i, label in enumerate(labels_order):
        if label in uavid_gt_colors:
            color_matrix[i] = uavid_gt_colors[label]
        else:
            color_matrix[i] = [255, 255, 255]
    return color_matrix[global_seg_map]

def exact_multi_scale_ensemble_matrix(image_path, clip_prompts, mapping_keys, scale_class_matrix, scale_threshold_matrix=None, temperature=1.0, sw_plot=True):
    """
    Multi-Scale ensemble that applies a specific weight to each class at each scale,
    with a Hard Activation Gate to mathematically crush low-confidence hallucinations.
    """
    # [AI]: Choose computation device here to ensure all tensors are
    # [AI]: placed consistently; this also keeps the function portable
    # [AI]: when running on CPU-only machines for debugging.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 1. Global Setup
    original_image = Image.open(image_path).convert("RGB")
    W_orig, H_orig = original_image.size
    num_classes = len(clip_prompts)
    
    # 3D Probability Tensor: (Height, Width, Num_Classes)
    fused_prob_tensor = np.zeros((H_orig, W_orig, num_classes), dtype=np.float32)

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Extract the patch scales from the matrix keys and sort descending
    patch_scales = sorted(list(scale_class_matrix.keys()), reverse=True)
    
    if scale_threshold_matrix is None:
        scale_threshold_matrix = {}

    # 2. Iterate through each Spatial Scale
    for p_size in patch_scales:
        print(f"\n--- Processing Scale: {p_size}x{p_size} ---")
        
        # Extract the specific class weights for THIS scale
        scale_weights_dict = scale_class_matrix[p_size]
        current_weight_array = np.array(
            [scale_weights_dict.get(k, 1.0) for k in mapping_keys], 
            dtype=np.float32
        )
        
        # Extract specific class hard-thresholds for THIS scale
        scale_thresholds_dict = scale_threshold_matrix.get(p_size, {})
        current_threshold_array = torch.tensor(
            [scale_thresholds_dict.get(k, 0.0) for k in mapping_keys], 
            device=device, dtype=torch.float32
        )
        
        pad_w = (p_size - (W_orig % p_size)) % p_size
        pad_h = (p_size - (H_orig % p_size)) % p_size
        padded_w, padded_h = W_orig + pad_w, H_orig + pad_h
        
        padded_img = Image.new("RGB", (padded_w, padded_h), color=(0, 0, 0))
        padded_img.paste(original_image, (0, 0))
        
        cols, rows = padded_w // p_size, padded_h // p_size
        padded_scale_tensor = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)

        patches = []
        boxes = [] 
        
        for r in range(rows):
            for c in range(cols):
                left, upper = c * p_size, r * p_size
                right, lower = left + p_size, upper + p_size
                
                patch = padded_img.crop((left, upper, right, lower))
                patches.append(preprocess(patch))
                boxes.append((upper, lower, left, right))

        batch_tensor = torch.stack(patches).to(device)
        batch_size = 128 
        
        all_probs = []
        with torch.no_grad():
            logit_scale = model.logit_scale.exp()
            for i in tqdm(range(0, len(batch_tensor), batch_size), desc=f"Evaluating Patches"):
                chunk = batch_tensor[i : i + batch_size]
                
                with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(chunk)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Compute Raw Logits
                    raw_logits = logit_scale * (image_features @ text_features.T)
                    
                    # 1. RAW Probability (Strict confidence for evaluating thresholds)
                    raw_probs = torch.softmax(raw_logits, dim=-1)
                    
                    # 2. SOFT Probability (If temperature is used, otherwise identical)
                    logits = raw_logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    
                    # --- THE HARD ACTIVATION GATE ---
                    # Crushes the probability to 0.0 if it doesn't meet the specified minimum bound
                    probs = torch.where(raw_probs < current_threshold_array, torch.zeros_like(probs), probs)
                    
                    all_probs.extend(probs.cpu().numpy())

        # Project probabilities into the padded tensor
        for (u, l, left, right), prob_dist in zip(boxes, all_probs):
            padded_scale_tensor[u:l, left:right, :] = prob_dist 

        # Slice off padding to get native dimensions
        local_scale_tensor = padded_scale_tensor[:H_orig, :W_orig, :]

        # 3. MATHEMATICAL FUSION (The Matrix Multiplication)
        weighted_local_tensor = local_scale_tensor * current_weight_array
        fused_prob_tensor += weighted_local_tensor

    # 4. Final Decision Phase
    # [AI]: Normalise the fused tensor to a common confidence scale so that
    # [AI]: thresholds and visualisations are comparable across images and
    # [AI]: scales (prevents high-scale dominance due to unbounded sums).
    if fused_prob_tensor.max() > 0:
        fused_prob_tensor /= fused_prob_tensor.max()
    
    fused_class_map = np.argmax(fused_prob_tensor, axis=-1).astype(np.int32)
    fused_confidence_map = np.max(fused_prob_tensor, axis=-1)

    # =========================================================================
    # 5. MATPLOTLIB VISUALIZATION BLOCK
    # =========================================================================
    if sw_plot:
        print("\nComposing Final Visual Map...")
        rgb_map_uint8 = segment_mask_to_rgb(fused_class_map, mapping_keys)
        rgb_map_float = rgb_map_uint8.astype(np.float32) / 255.0
        rgba_map = np.dstack((rgb_map_float, fused_confidence_map))
        gt_path = image_path.replace("Images", "Labels")
        gt_rgb = None
        if os.path.exists(gt_path):
            gt_rgb = np.array(Image.open(gt_path).convert("RGB"))

        original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        original_rgb = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
        original_gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
        fig, ax = plt.subplots(2, 2, figsize=(16, 14))

        ax[0, 0].imshow(original_rgb)
        ax[0, 0].set_title(f"Original Image ({W_orig}x{H_orig})", fontsize=14)
        ax[0, 0].axis('off')

        ax[0, 1].imshow(original_gray, cmap='gray')
        ax[0, 1].imshow(rgba_map)
        ax[0, 1].set_title(f"Matrix-Weighted Soft Voting Fusion ({W_orig}x{H_orig})", fontsize=14)
        ax[0, 1].axis('off')

        ax[1, 0].imshow(rgb_map_uint8)
        ax[1, 0].set_title("Prediction Mask (No Overlay)", fontsize=14)
        ax[1, 0].axis('off')

        if gt_rgb is not None:
            ax[1, 1].imshow(gt_rgb)
            ax[1, 1].set_title("Ground Truth Mask", fontsize=14)
        else:
            ax[1, 1].text(0.5, 0.5, "Ground Truth not found", ha='center', va='center', fontsize=12)
            ax[1, 1].set_title("Ground Truth Mask", fontsize=14)
        ax[1, 1].axis('off')

        uavid_gt_colors = {
            "building": [128, 0, 0], "road": [128, 64, 128], "tree": [0, 128, 0],
            "low_veg": [128, 128, 0], "clutter": [0, 0, 0], "car": [192, 0, 192], "human": [64, 64, 0]
        }
        
        patches_legend = [
            mpatches.Patch(
                color=np.array(uavid_gt_colors.get(mapping_keys[i], [255, 255, 255])) / 255.0, 
                label=clip_prompts[i].split()[-1].capitalize()
            ) 
            for i in range(len(clip_prompts))
        ]
        ax[0, 1].legend(handles=patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    return fused_class_map, fused_confidence_map

# =============================================================================
# --- EXECUTION ---
# =============================================================================

"""clip_prompts = [
    "drone view of a building", "drone view of a road", "drone view of a tree",
    "drone view of low vegetation", "drone view of background clutter", 
    "drone view of a car", "drone view of a human"
]"""
clip_prompts = [
    "building", "road", "tree",
    "low vegetation", "background clutter", 
    "car", "human"
    ]

mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

# --- THE SCALE-CLASS WEIGHT MATRIX ---
# Format: { Patch_Size: { "class_key": weight_multiplier } }

# Long prompts weights

scale_class_matrix = {
    448: {
        "building": 1.0182,
        "road": 0.3347,
        "tree": 0.3707,
        "low_veg": 0.5934,
        "clutter": 1.2224,
        "car": 0.0000,
        "human": 0.0000,
    },
    224: {
        "building": 2.2611,
        "road": 2.3432,
        "tree": 2.3511,
        "low_veg": 0.5339,
        "clutter": 2.2603,
        "car": 0.0000,
        "human": 0.0000,
    },
    112: {
        "building": 2.3151,
        "road": 0.5154,
        "tree": 2.6982,
        "low_veg": 0.6610,
        "clutter": 0.8028,
        "car": 0.6423,
        "human": 0.0177,
    },
    56: {
        "building": 0.1974,
        "road": 0.4617,
        "tree": 0.8353,
        "low_veg": 0.5938,
        "clutter": 0.5614,
        "car": 2.8276,
        "human": 4.6350,
    },
}

scale_threshold_matrix = {
    112: {
        "building": 0.0000,
        "road": 0.0000,
        "tree": 0.0000,
        "low_veg": 0.0000,
        "clutter": 0.0000,
        "car": 0.8500,
        "human": 0.0000,
    },
    56: {
        "building": 0.0000,
        "road": 0.0000,
        "tree": 0.0000,
        "low_veg": 0.0000,
        "clutter": 0.0000,
        "car": 0.8448,
        "human": 0.8922,
    },
}

#Short prompt weights:
"""scale_class_matrix = {
    448: {
        "building": 1.5976,
        "road": 0.3790,
        "tree": 0.3184,
        "low_veg": 0.3659,
        "clutter": 0.3778,
        "car": 0.0000,
        "human": 0.0000,
    },
    224: {
        "building": 2.4247,
        "road": 2.5252,
        "tree": 2.1396,
        "low_veg": 0.4932,
        "clutter": 5.2522,
        "car": 0.0000,
        "human": 0.0000,
    },
    112: {
        "building": 2.8658,
        "road": 0.4342,
        "tree": 2.3137,
        "low_veg": 0.5577,
        "clutter": 0.7276,
        "car": 0.6056,
        "human": 0.0182,
    },
    56: {
        "building": 0.8012,
        "road": 0.4539,
        "tree": 0.3487,
        "low_veg": 0.4727,
        "clutter": 0.5277,
        "car": 4.0141,
        "human": 11.7080,
    },
}

scale_threshold_matrix = {
    112: {
        "building": 0.0000,
        "road": 0.0000,
        "tree": 0.0000,
        "low_veg": 0.0000,
        "clutter": 0.0000,
        "car": 0.8533,
        "human": 0.0000,
    },
    56: {
        "building": 0.0000,
        "road": 0.0000,
        "tree": 0.0000,
        "low_veg": 0.0000,
        "clutter": 0.0000,
        "car": 0.8480,
        "human": 0.6426,
    },
}
"""
test1_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1longprompt\Images"
test2_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1shortprompt\Images"
cv_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images"

sw_plot = True  # Set to False to skip visualization and just save predictions
image_dir = test1_dir
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][::40]
    
miou_lst = []
weighted_miou_lst = []
per_class_lst = []
time_lst = []

# Execute

for image_path in image_paths:
    start_time = time.time()

    fused_class_map, fused_conf_map = exact_multi_scale_ensemble_matrix(
        image_path,
        clip_prompts,
        mapping_keys,
        scale_class_matrix,
        scale_threshold_matrix=scale_threshold_matrix,
        temperature=1.0,
        sw_plot=sw_plot
    )

    results, miou, weighted_miou = save_and_evaluate_single_image(
        image_path=image_path,
        seg_map=fused_class_map,
        mapping_keys=mapping_keys
    )

    if results is not None:
        miou_lst.append(miou)
        weighted_miou_lst.append(weighted_miou)
        per_class_lst.append(results)
    time_lst.append(time.time() - start_time)

if miou_lst:
    print(f"\n=== FINAL AVERAGE mIoU across {len(miou_lst)} images: {np.mean(miou_lst):.4f} ===")
    print(f"=== FINAL AVERAGE Weighted mIoU across {len(weighted_miou_lst)} images: {np.mean(weighted_miou_lst):.4f} ===")
    print(f"=== FINAL AVERAGE Inference Time across {len(time_lst)} images: {np.mean(time_lst):.4f} seconds ===")

    # per_class_lst is a list of {class_name: {metric: value}} dicts.
    # Convert each to a DataFrame with classes as rows, metrics as columns,
    # then average across images.
    dataframes = [pd.DataFrame(run).T for run in per_class_lst]  # .T → rows=classes, cols=metrics
    avg_df = sum(dataframes) / len(dataframes)

    header = f"\n{'='*84}"
    header += f"\nImage: Average Results"
    header += f"\n{'='*84}"
    print(header)

    col_w = 20
    print(f"\n{'Category':<{col_w}} {'IoU':>8} {'F0.5':>8} {'F1':>8} {'F2':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 76)

    for name, m in avg_df.iterrows():  # iterrows() since rows=classes now
        print(f"{name:<{col_w}} {m['IoU']:>8.4f} {m['F0.5']:>8.4f} {m['F1']:>8.4f} "
              f"{m['F2']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}")

    print("-" * 76)
    avg_miou = avg_df['IoU'].mean()
    avg_weighted_miou = np.mean(weighted_miou_lst)
    print(f"\nmIoU:          {avg_miou:.4f}  ({avg_miou*100:.2f}%)")
    print(f"Weighted mIoU: {avg_weighted_miou:.4f}  ({avg_weighted_miou*100:.2f}%)\n")