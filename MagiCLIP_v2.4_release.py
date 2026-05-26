"""
BlackMagiCLIP: Multi-Scale Bayesian Panoptic Engine
===================================================
This script implements a zero-shot semantic segmentation pipeline using a frozen CLIP (RN50) model.
It utilizes:
1. SpatialLayerHooks for dense Grad-CAM feature extraction from hidden layers.
2. Multi-Scale Patching with interpolation-free padding.
3. A Scale-Class Weight Matrix to natively handle the "Thing vs. Stuff" spatial dichotomy.
4. Hard Activation Gates to crush hallucinations and drastically speed up backpropagation.
"""

import torch
import torch.nn.functional as F
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
import os
import time
import pandas as pd
from support.scoring_general import save_and_evaluate_single_image

# [AI]: Reuse the canonical evaluation utilities to ensure saved masks
# [AI]: and computed metrics stay consistent across scripts.

# =============================================================================
# --- 1. HARDWARE & CONTEXT INITIALIZATION ---
# =============================================================================

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    # Warm up the GPU matrix cores
    _ = torch.ones(1, device="cuda") @ torch.ones(1, device="cuda")

# [AI]: Warming up the CUDA device reduces the first-inference latency
# [AI]: (initial kernel/JIT overhead) and helps avoid sporadic OOMs.

# =============================================================================
# --- 2. THE SPATIAL HOOK (GRAD-CAM ENGINE) ---
# =============================================================================

class SpatialLayerHook:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.activations = output
        self.hook_grad = output.register_hook(self.save_gradient)
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def close(self):
        self.hook.remove()

# [AI]: The hook captures activations and gradients non-invasively so
# [AI]: we can compute Grad-CAM maps without modifying the model graph.

# =============================================================================
# --- 3. HELPER FUNCTIONS ---
# =============================================================================

def segment_mask_to_rgb(global_seg_map, labels_order):
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

# =============================================================================
# --- 4. THE CORE ARCHITECTURE ---
# =============================================================================

def exact_multi_scale_ensemble_matrix(image_path, clip_prompts, mapping_keys, 
                                       scale_class_matrix, model, preprocess, hook,
                                       scale_threshold_matrix=None, sw_plot=True):
    # [AI]: Select device once to ensure all tensors and model ops
    # [AI]: are colocated; this reduces cross-device transfers.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    original_image = Image.open(image_path).convert("RGB")
    W_orig, H_orig = original_image.size
    num_classes = len(clip_prompts)
    
    # Initialize the 3D Probability Tensor at EXACT native resolution
    fused_prob_tensor = np.zeros((H_orig, W_orig, num_classes), dtype=np.float32)

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    patch_scales = sorted(list(scale_class_matrix.keys()), reverse=True)

    if scale_threshold_matrix is None:
        scale_threshold_matrix = {}

    for p_size in patch_scales:
        print(f"\n--- Processing Scale: {p_size}x{p_size} (Grad-CAM Enabled) ---")
        
        # 1. Extract the specific class weights for THIS scale
        scale_weights_dict = scale_class_matrix[p_size]
        current_weight_array = np.array(
            [scale_weights_dict.get(k, 1.0) for k in mapping_keys], 
            dtype=np.float32
        )
        
        # 2. Extract specific class hard-thresholds for THIS scale
        scale_thresholds_dict = scale_threshold_matrix.get(p_size, {})
        current_threshold_array = torch.tensor(
            [scale_thresholds_dict.get(k, 0.0) for k in mapping_keys], 
            device=device, dtype=torch.float32
        )
        
        # 3. Calculate necessary padding for interpolation-free extraction
        pad_w = (p_size - (W_orig % p_size)) % p_size
        pad_h = (p_size - (H_orig % p_size)) % p_size
        padded_w, padded_h = W_orig + pad_w, H_orig + pad_h
        
        padded_img = Image.new("RGB", (padded_w, padded_h), color=(0, 0, 0))
        padded_img.paste(original_image, (0, 0))
        
        cols, rows = padded_w // p_size, padded_h // p_size
        padded_scale_tensor = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)

        patches, boxes = [], []
        
        for r in range(rows):
            for c in range(cols):
                left, upper = c * p_size, r * p_size
                right, lower = left + p_size, upper + p_size
                
                patch = padded_img.crop((left, upper, right, lower))
                patches.append(preprocess(patch))
                boxes.append((upper, lower, left, right))

        batch_tensor = torch.stack(patches).to(device)
        
        # Batch size lowered to 32 to accommodate backpropagation passes
        batch_size = 32 
        all_dense_probs = []
        
        # 4. Inference & Batched Backpropagation
        for i in tqdm(range(0, len(batch_tensor), batch_size), desc=f"Evaluating Patches"):
            chunk = batch_tensor[i : i + batch_size].type(model.dtype)
            chunk.requires_grad = True # Mandatory for backprop
            
            # Forward Pass
            image_features = model.encode_image(chunk).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = model.logit_scale.exp().float() * (image_features @ text_features.T)
            probs = torch.softmax(logits, dim=-1) # Raw Probability Distribution
            
            # --- THE HARD ACTIVATION GATE ---
            # Crushes the probability to 0.0 if it doesn't meet the specified minimum bound
            # [AI]: Apply a hard gate to suppress low-confidence classes early,
            # [AI]: reducing backward work and decreasing false-positive noise.
            probs = torch.where(probs < current_threshold_array, torch.zeros_like(probs), probs)
            
            # Storage for the dense, pixel-level heatmaps for this batch
            batch_dense_cams = torch.zeros((chunk.shape[0], num_classes, p_size, p_size), device=device)
            
            # --- THE GRAD-CAM BACKWARD LOOP ---
            active_classes = [
                c_idx for c_idx, key in enumerate(mapping_keys)
                if scale_weights_dict.get(key, 1.0) > 0.0
]
            for c_idx in active_classes:
                if probs[:, c_idx].sum() == 0:
                    continue
                
                model.zero_grad()
                
                # Score for this specific class across the whole batch
                score = (image_features @ text_features[c_idx]) 
                score.sum().backward(retain_graph=True)
                
                if hook.gradients is not None:
                    g, a = hook.gradients.clone().float(), hook.activations.clone().float()
                    w = torch.mean(g, dim=[2, 3], keepdim=True)
                    cam = F.relu(torch.sum(w * a, dim=1, keepdim=True)) # Shape: (Batch, 1, 7, 7)
                    
                    # Min-Max Normalization per patch
                    cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                    cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                    
                    # Upsample using nearest neighbor to preserve strict boundaries
                    cam_up = F.interpolate(cam_norm, size=(p_size, p_size), mode='nearest')
                    
                    # Modulate the spatial heatmap by the patch's overall probability
                    class_prob = probs[:, c_idx].view(-1, 1, 1, 1)
                    batch_dense_cams[:, c_idx, :, :] = (cam_up * class_prob).squeeze(1)
                    # [AI]: Nearest-neighbour upsampling + per-patch modulation
                    # [AI]: preserves crisp object boundaries and encodes patch
                    # [AI]: level confidence spatially into the dense heatmap.
                
            # Move to CPU and format as (Batch, P_size, P_size, Num_Classes)
            all_dense_probs.extend(batch_dense_cams.permute(0, 2, 3, 1).detach().cpu().numpy())
            
            # Clear cache to prevent RTX 4070 OOM
            del chunk, image_features, logits, probs, batch_dense_cams
            torch.cuda.empty_cache()

        # 5. Native Block Projection
        for (u, l, left, right), dense_prob_block in zip(boxes, all_dense_probs):
            padded_scale_tensor[u:l, left:right, :] = dense_prob_block 

        # Slice off padding to get native dimensions
        local_scale_tensor = padded_scale_tensor[:H_orig, :W_orig, :]

        # 6. Mathematical Fusion (Tensor Accumulation with Matrix Weights)
        weighted_local_tensor = local_scale_tensor * current_weight_array
        fused_prob_tensor += weighted_local_tensor


    # --- FINAL DECISION PHASE ---
    # [AI]: Normalise fused scores to a common scale so confidence maps
    # [AI]: are comparable across images and scales; this avoids bias
    # [AI]: from differing patch counts or weight magnitudes.
    if fused_prob_tensor.max() > 0:
        fused_prob_tensor /= fused_prob_tensor.max()
    
    fused_class_map = np.argmax(fused_prob_tensor, axis=-1).astype(np.int32)
    fused_confidence_map = np.max(fused_prob_tensor, axis=-1)

    # =========================================================================
    # --- VISUALIZATION BLOCK ---
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
        ax[0, 1].set_title(f"BlackMagiCLIP Dense Matrix Fusion ({W_orig}x{H_orig})", fontsize=14)
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
# --- 5. PIPELINE EXECUTION ---
# =============================================================================

if __name__ == "__main__":
    
    clip_prompts = [
    "drone view of a building", "drone view of a road", "drone view of a tree",
    "drone view of low vegetation", "drone view of background clutter", 
    "drone view of a car", "drone view of a human"
    ]
    
    mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

    #Long label optimized weights 
    scale_class_matrix = {
        448: {
            "building": 0.9810,
            "road": 0.6498,
            "tree": 0.7276,
            "low_veg": 0.3784,
            "clutter": 1.4474,
            "car": 0.0000,
            "human": 0.0000,
        },
        224: {
            "building": 2.8208,
            "road": 0.5397,
            "tree": 3.2826,
            "low_veg": 3.0060,
            "clutter": 0.7609,
            "car": 2.6541,
            "human": 0.4433,
        },
    }
    scale_threshold_matrix = {
        224: {
            "building": 0.2660,
            "road": 0.0000,
            "tree": 0.2510,
            "low_veg": 0.3964,
            "clutter": 0.1756,
            "car": 0.8893,
            "human": 0.6498,
        },
    }

    #short label optimized weights
        
    """scale_class_matrix = {
        448: {
            "building": 2.4467,
            "road": 0.6056,
            "tree": 0.5274,
            "low_veg": 0.4086,
            "clutter": 0.6043,
            "car": 0.0000,
            "human": 0.0000,
        },
        224: {
            "building": 2.6240,
            "road": 0.5052,
            "tree": 2.9124,
            "low_veg": 3.0818,
            "clutter": 4.1069,
            "car": 3.8081,
            "human": 2.7756,
        },
    }
    scale_threshold_matrix = {
        224: {
            "building": 0.2714,
            "road": 0.0000,
            "tree": 0.2719,
            "low_veg": 0.5298,
            "clutter": 0.0925,
            "car": 0.8959,
            "human": 0.5747,
        },
    }"""




    # Define Image Directory
    test1_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1longpromptMagiCLIP\Images"
    test2_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1shortpromptMagiCLIP\Images"
    cv_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images"


    image_dir = test1_dir  # Change this to test2_dir or cv_dir as needed
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))][::40]
    

    # Execute Pipeline
    sw_plot = True  # Set to False to skip visualization and process silently
    miou_lst = []
    weighted_miou_lst = []
    per_class_lst = []
    time_lst = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    hook = SpatialLayerHook(model.visual.layer4)

    for image_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        fused_class_map, fused_conf_map = exact_multi_scale_ensemble_matrix(
            image_path, 
            clip_prompts, 
            mapping_keys, 
            scale_class_matrix,
            model,
            preprocess,
            hook,
            scale_threshold_matrix=scale_threshold_matrix,
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
        # ADD THIS after each image:
        del fused_class_map, fused_conf_map
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Forces CUDA to actually flush all operations and free memory before the next image starts processing                                

    # --- FINAL PANDAS REPORTING ---
    if miou_lst:
        print(f"\n=== FINAL AVERAGE mIoU across {len(miou_lst)} images: {np.mean(miou_lst):.4f} ===")
        print(f"=== FINAL AVERAGE Weighted mIoU across {len(weighted_miou_lst)} images: {np.mean(weighted_miou_lst):.4f} ===")
        print(f"=== FINAL AVERAGE Inference Time across {len(time_lst)} images: {np.mean(time_lst):.4f} seconds ===")

        # .T so rows=classes, cols=metrics — consistent with PatchCLIP
        dataframes = [pd.DataFrame(run).T for run in per_class_lst]
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