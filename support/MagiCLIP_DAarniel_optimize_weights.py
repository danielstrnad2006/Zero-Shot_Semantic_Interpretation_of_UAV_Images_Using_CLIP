"""
optimize_weights_magiclip.py
────────────────────────────────────────────────────────────────────────────────
Adam-based gradient optimizer for Daniel_MagiCLIP.py's
scale_class_matrix and scale_threshold_matrix.

DIFFERENCE FROM optimize_weights.py (Daniel_Patch.py version)
──────────────────────────────────────────────────────────────
Daniel_Patch (ViT-B/32) produces one probability vector per patch.
Daniel_MagiCLIP (RN50 + Grad-CAM) produces a spatial (7×7) heatmap per
class per patch. The 7×7 map comes from layer4 of the ResNet and is
upsampled inside each patch to produce sub-patch spatial predictions.

HOW THIS OPTIMIZER HANDLES THAT
────────────────────────────────
1.  CLIP + Grad-CAM is run ONCE (expensive), storing per patch:
        cams_7x7  : (N_patches, C, 7, 7)  — normalised Grad-CAM maps
        raw_probs : (N_patches, C)         — softmax probabilities

2.  A spatial index is precomputed that maps each 56-pixel base patch to
    the exact (coarse_patch_index, cam_row, cam_col) within the 7×7 map
    for every scale.  This makes the per-epoch forward pass trivially cheap.

3.  The differentiable forward pass:
        gated_probs = raw_probs × sigmoid(k × (raw_probs − threshold))
        dense_val   = cam_7x7  × gated_probs          (spatial modulation)
        sampled     = dense_val sampled at base-patch positions
        fused      += sampled × class_weight

    This is end-to-end differentiable w.r.t. threshold and weight params.

4.  Frozen zeros, log-space weights, logit-space thresholds, steepness
    annealing — identical to the ViT-B/32 optimiser.
────────────────────────────────────────────────────────────────────────────────
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

from scoring_general import (
    CATEGORY_COLOURS,
    validate_colour_map,
    build_colour_index,
    image_to_label_array,
)

# [AI]: Grad-CAM + CLIP inference is expensive; cache compact CAM maps
# [AI]: and raw probabilities once so optimisation iterates cheaply.

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 – GRAD-CAM HOOK  (mirrors Daniel_MagiCLIP.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialLayerHook:
    def __init__(self, module):
        self.activations = None
        self.gradients   = None
        self._fwd        = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activations = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def close(self):
        self._fwd.remove()

# [AI]: The hook captures activations/gradients to compute Grad-CAM
# [AI]: without changing the original model graph; keeping it minimal
# [AI]: avoids accidental side-effects during the caching pass.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – GT / CLIP INDEX HELPERS  (shared with ViT-B/32 optimiser)
# ══════════════════════════════════════════════════════════════════════════════

def _gt_to_clip_index_map(gt_categories: list, mapping_keys: list) -> list:
    _ALIASES = {
        "background_clutter": "clutter",
        "low_vegetation":     "low_veg",
        "static_car":         "car",
        "moving_car":         "car",
    }
    def _norm(s):
        s = s.lower().replace(" ", "_")
        return _ALIASES.get(s, s)

    clip_lookup = {k: i for i, k in enumerate(mapping_keys)}
    result = []
    for name in gt_categories:
        n = _norm(name)
        if n not in clip_lookup:
            raise ValueError(
                f"GT category '{name}' (normalised: '{n}') has no matching "
                f"CLIP key in {mapping_keys}"
            )
        result.append(clip_lookup[n])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – GRAD-CAM CACHE
# ══════════════════════════════════════════════════════════════════════════════

_CAM_SIZE = 7   # RN50 layer4 spatial resolution

def precompute_gradcam_cache(
    image_paths:        list,
    clip_prompts:       list,
    mapping_keys:       list,
    patch_scales:       list,
    scale_class_matrix: dict,
    colour_to_idx:      dict,
    num_classes_gt:     int,
    device:             str,
    batch_size:         int = 32,
) -> list:
    """
    Run RN50 + Grad-CAM once and store the compact 7×7 CAM maps.

    Per image, per scale, stores:
        cams     : (N_coarse, C, 7, 7)  normalised Grad-CAM (before prob. modulation)
        raw_probs: (N_coarse, C)         softmax probabilities (before threshold)

    Additionally, for the base scale (finest patch):
        gt_dist  : (N_base, C_gt)        GT class fractions per base patch

    And a spatial_idx that maps each base patch to the correct coarse patch
    and 7×7 CAM cell for every scale.
    """
    patch_scales_desc = sorted(patch_scales, reverse=True)
    base_scale        = min(patch_scales)

    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    hook = SpatialLayerHook(model.visual.layer4)

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    C_clip = len(clip_prompts)
    cache  = []

    for image_path in tqdm(image_paths, desc="Caching Grad-CAM"):
        pred_path = image_path.replace("Images", "Predictions")
        gt_path   = pred_path.replace("Predictions", "Labels")
        if not os.path.exists(gt_path):
            print(f"  [skip] no GT: {os.path.basename(image_path)}")
            continue

        original_image = Image.open(image_path).convert("RGB")
        W_orig, H_orig = original_image.size
        gt_label       = image_to_label_array(gt_path, colour_to_idx, num_classes_gt)

        # ── Base-scale GT distribution ────────────────────────────────────────
        B         = base_scale
        cols_base = math.ceil(W_orig / B)
        rows_base = math.ceil(H_orig / B)
        N_base    = rows_base * cols_base

        gt_dists = np.zeros((N_base, num_classes_gt), dtype=np.float32)
        for rb in range(rows_base):
            for cb in range(cols_base):
                x0, y0 = cb * B, rb * B
                x1, y1 = min(x0 + B, W_orig), min(y0 + B, H_orig)
                patch_gt = gt_label[y0:y1, x0:x1]
                valid    = patch_gt[patch_gt >= 0]
                if len(valid):
                    for cls in range(num_classes_gt):
                        gt_dists[rb * cols_base + cb, cls] = np.sum(valid == cls) / len(valid)

        # ── Per-scale Grad-CAM inference ──────────────────────────────────────
        cams_per_scale  = {}
        probs_per_scale = {}
        spatial_idx     = {}

        for p_size in patch_scales_desc:
            pad_w      = (-W_orig) % p_size
            pad_h      = (-H_orig) % p_size
            pw, ph     = W_orig + pad_w, H_orig + pad_h
            cols, rows = pw // p_size, ph // p_size
            N_patches  = rows * cols

            padded_img = Image.new("RGB", (pw, ph), color=(0, 0, 0))
            padded_img.paste(original_image, (0, 0))

            patches = [
                preprocess(padded_img.crop((cc * p_size, rr * p_size,
                                            (cc + 1) * p_size, (rr + 1) * p_size)))
                for rr in range(rows) for cc in range(cols)
            ]
            batch_tensor = torch.stack(patches).to(device)

            # Only compute Grad-CAM for classes that are active at this scale
            active_classes = [
                i for i, k in enumerate(mapping_keys)
                if scale_class_matrix[p_size].get(k, 0.0) > 0.0
            ]

            # [AI]: Limiting Grad-CAM to active classes reduces backward
            # [AI]: work and memory, which speeds up caching considerably.

            all_cams  = torch.zeros(N_patches, C_clip, _CAM_SIZE, _CAM_SIZE)
            all_probs = torch.zeros(N_patches, C_clip)

            for i in tqdm(range(0, N_patches, batch_size),
                          desc=f"  Scale {p_size}", leave=False):
                chunk      = batch_tensor[i : i + batch_size].type(model.dtype)
                chunk_size = chunk.shape[0]
                chunk.requires_grad_(True)

                # Forward
                img_feats  = model.encode_image(chunk).float()
                img_feats  = img_feats / img_feats.norm(dim=-1, keepdim=True)
                logits     = model.logit_scale.exp().float() * (img_feats @ text_features.T)
                raw_probs  = torch.softmax(logits, dim=-1)

                all_probs[i : i + chunk_size] = raw_probs.detach().cpu()

                # Grad-CAM backward per active class
                for c_idx in active_classes:
                    if raw_probs[:, c_idx].sum().item() < 1e-9:
                        continue

                    model.zero_grad()
                    score = img_feats @ text_features[c_idx]
                    score.sum().backward(retain_graph=True)

                    if hook.gradients is None:
                        continue

                    g   = hook.gradients.clone().float()
                    a   = hook.activations.clone().float()
                    w   = g.mean(dim=[2, 3], keepdim=True)
                    cam = F.relu((w * a).sum(dim=1, keepdim=True))   # (B, 1, 7, 7)

                    cam_min  = cam.flatten(1).min(1)[0].view(-1, 1, 1, 1)
                    cam_max  = cam.flatten(1).max(1)[0].view(-1, 1, 1, 1)
                    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

                    all_cams[i : i + chunk_size, c_idx] = cam_norm.squeeze(1).detach().cpu()

            cams_per_scale[p_size]  = all_cams    # (N_coarse, C, 7, 7)
            probs_per_scale[p_size] = all_probs   # (N_coarse, C)

            # ── Spatial index: base patch → (coarse_idx, cam_row, cam_col) ──
            coarse_idxs = np.empty(N_base, dtype=np.int64)
            cam_ys      = np.empty(N_base, dtype=np.int64)
            cam_xs      = np.empty(N_base, dtype=np.int64)

            for rb in range(rows_base):
                for cb in range(cols_base):
                    b_idx    = rb * cols_base + cb
                    cy, cx   = rb * B + B // 2, cb * B + B // 2   # patch centre

                    cr = min(cy // p_size, rows - 1)
                    cc = min(cx // p_size, cols - 1)
                    coarse_idxs[b_idx] = cr * cols + cc

                    # Position within coarse patch → CAM cell (nearest-neighbour)
                    within_y = cy % p_size
                    within_x = cx % p_size
                    cam_ys[b_idx] = within_y * _CAM_SIZE // p_size
                    cam_xs[b_idx] = within_x * _CAM_SIZE // p_size

            spatial_idx[p_size] = (
                torch.from_numpy(coarse_idxs),
                torch.from_numpy(cam_ys),
                torch.from_numpy(cam_xs),
            )

        # [AI]: Precomputing spatial indices lets the forward pass sample
        # [AI]: directly by integer indexing rather than computing positions
        # [AI]: on every epoch — a small preprocessing step with big gains.

        cache.append({
            "cams":        cams_per_scale,            # {p_size: (N_coarse, C, 7, 7)}
            "raw_probs":   probs_per_scale,            # {p_size: (N_coarse, C)}
            "gt_dist":     torch.from_numpy(gt_dists), # (N_base, C_gt)
            "spatial_idx": spatial_idx,                # {p_size: (idx, y, x)}
        })

    hook.close()
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DIFFERENTIABLE FUSION MODULE  (identical structure to ViT-B/32)
# ══════════════════════════════════════════════════════════════════════════════

class FusionWeights(nn.Module):
    """
    Learnable weights and thresholds with frozen-zero masks.
    Weights in log-space (always ≥ 0), thresholds in logit-space (always in (0,1)).
    """
    def __init__(self, scale_class_matrix, scale_threshold_matrix, mapping_keys):
        super().__init__()
        self.mapping_keys = mapping_keys
        self.patch_scales = sorted(scale_class_matrix.keys(), reverse=True)

        def _logit(t):
            if t <= 0.0: return -10.0
            t = max(min(t, 1 - 1e-6), 1e-6)
            return math.log(t / (1.0 - t))

        for p in self.patch_scales:
            w_init = [scale_class_matrix[p].get(k, 0.0)             for k in mapping_keys]
            t_init = [scale_threshold_matrix.get(p, {}).get(k, 0.0) for k in mapping_keys]

            self.register_buffer(
                f"w_freeze_{p}",
                torch.tensor([1.0 if w == 0.0 else 0.0 for w in w_init])
            )
            self.register_buffer(
                f"t_freeze_{p}",
                torch.tensor([1.0 if t == 0.0 else 0.0 for t in t_init])
            )
            self.register_parameter(
                f"log_w_{p}",
                nn.Parameter(torch.tensor(
                    [math.log(max(w, 1e-6)) for w in w_init], dtype=torch.float32
                ))
            )
            self.register_parameter(
                f"raw_t_{p}",
                nn.Parameter(torch.tensor(
                    [_logit(t) for t in t_init], dtype=torch.float32
                ))
            )

        # [AI]: Store weights/thresholds in log/logit spaces so optimisation
        # [AI]: naturally respects non-negativity and (0,1) bounds.

    def weights(self, p: int) -> torch.Tensor:
        return torch.exp(getattr(self, f"log_w_{p}")) * (1.0 - getattr(self, f"w_freeze_{p}"))

    def thresholds(self, p: int) -> torch.Tensor:
        return torch.sigmoid(getattr(self, f"raw_t_{p}")) * (1.0 - getattr(self, f"t_freeze_{p}"))

    def forward(self, item: dict, gate_steepness: float = 50.0) -> torch.Tensor:
        """
        item: one cache entry.
        Returns fused (N_base, C).

        For each scale:
          1. Soft gate on raw_probs         → differentiable threshold
          2. Modulate 7×7 CAM by gated prob → dense spatial value
          3. Sample at base-patch positions → (N_base, C)
          4. Accumulate weighted by class weights
        """
        dev    = next(self.parameters()).device
        N_base = item["gt_dist"].shape[0]
        C      = len(self.mapping_keys)
        fused  = torch.zeros(N_base, C, device=dev)

        for p in self.patch_scales:
            raw_probs = item["raw_probs"][p].to(dev)   # (N_coarse, C)
            cams      = item["cams"][p].to(dev)         # (N_coarse, C, 7, 7)
            w         = self.weights(p)                 # (C,)
            thr       = self.thresholds(p)              # (C,)

            # Differentiable soft gate
            gate = torch.sigmoid(
                gate_steepness * (raw_probs - thr.unsqueeze(0))
            )  # (N_coarse, C)

            # [AI]: Applying the gate to the CAM maps turns coarse-class
            # [AI]: confidence into spatial importance, enabling spatially
            # [AI]: aware weighting during optimisation.

            # Modulate CAM spatially: (N_coarse, C, 7, 7) × (N_coarse, C, 1, 1)
            gated_cams = cams * gate.unsqueeze(-1).unsqueeze(-1)

            # Sample at base-patch centre positions
            coarse_idx, cam_y, cam_x = item["spatial_idx"][p]
            coarse_idx = coarse_idx.to(dev)
            cam_y      = cam_y.to(dev)
            cam_x      = cam_x.to(dev)

            # Permute to (N_coarse, 7, 7, C) for clean integer indexing
            gc_perm = gated_cams.permute(0, 2, 3, 1)        # (N_coarse, 7, 7, C)
            sampled = gc_perm[coarse_idx, cam_y, cam_x, :]   # (N_base, C)

            fused = fused + sampled * w.unsqueeze(0)

        return fused  # (N_base, C)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – SOFT mIoU LOSS  (identical to ViT-B/32 optimiser)
# ══════════════════════════════════════════════════════════════════════════════

def soft_miou_loss(
    fused_probs: torch.Tensor,   # (N, C_clip)
    gt_dist:     torch.Tensor,   # (N, C_gt)
    gt_to_clip:  list,
) -> torch.Tensor:
    dev    = fused_probs.device
    N, C   = fused_probs.shape

    # L1-normalise predictions
    p_norm = fused_probs / (fused_probs.sum(dim=-1, keepdim=True) + 1e-8)

    # GT in CLIP class order
    g = torch.zeros(N, C, device=dev)
    for gt_idx, clip_idx in enumerate(gt_to_clip):
        g[:, clip_idx] += gt_dist[:, gt_idx].to(dev)

    iou_per_class = []
    for c in range(C):
        p_c = p_norm[:, c];  g_c = g[:, c]
        tp  = (p_c * g_c).sum()
        fp  = (p_c * (1.0 - g_c)).sum()
        fn  = ((1.0 - p_c) * g_c).sum()
        iou_per_class.append(tp / (tp + fp + fn + 1e-8))

    return torch.stack(iou_per_class).mean()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – MAIN OPTIMISER LOOP
# ══════════════════════════════════════════════════════════════════════════════

def optimize_weights(
    image_paths:            list,
    clip_prompts:           list,
    mapping_keys:           list,
    scale_class_matrix:     dict,
    scale_threshold_matrix: dict,
    *,
    num_epochs:      int   = 150,
    lr:              float = 3e-2,
    steepness_start: float = 10.0,
    steepness_end:   float = 80.0,
    print_every:     int   = 10,
    batch_size:      int   = 32,     # keep low for Grad-CAM backprop
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    validate_colour_map(CATEGORY_COLOURS)
    gt_categories, colour_to_idx = build_colour_index(
        CATEGORY_COLOURS, merge_cars=True, merge_vegetation=False
    )
    num_classes_gt = len(gt_categories)
    print(f"[*] GT categories ({num_classes_gt}): {gt_categories}")

    gt_to_clip = _gt_to_clip_index_map(gt_categories, mapping_keys)
    print(f"[*] GT→CLIP: {list(zip(gt_categories, [mapping_keys[i] for i in gt_to_clip]))}")

    patch_scales = sorted(scale_class_matrix.keys(), reverse=True)

    # ── One-time Grad-CAM cache ───────────────────────────────────────────────
    cache = precompute_gradcam_cache(
        image_paths, clip_prompts, mapping_keys, patch_scales,
        scale_class_matrix, colour_to_idx, num_classes_gt,
        device, batch_size=batch_size,
    )
    if not cache:
        raise RuntimeError("No valid (image, GT) pairs found. Check path structure.")
    print(f"[*] Cached {len(cache)} image(s).\n")

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model     = FusionWeights(scale_class_matrix, scale_threshold_matrix, mapping_keys).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    best_miou  = -1.0
    best_state = None

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        t_frac    = (epoch - 1) / max(num_epochs - 1, 1)
        steepness = steepness_start + t_frac * (steepness_end - steepness_start)

        optimizer.zero_grad()
        total_miou = 0.0

        for item in cache:
            fused    = model(item, gate_steepness=steepness)
            miou_val = soft_miou_loss(fused, item["gt_dist"], gt_to_clip)
            (-miou_val).backward()
            total_miou += miou_val.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        avg_miou = total_miou / len(cache)
        if avg_miou > best_miou:
            best_miou  = avg_miou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"  Epoch {epoch:4d}/{num_epochs}  |  "
                f"Soft mIoU: {avg_miou:.4f}  |  Best: {best_miou:.4f}  |  "
                f"Steepness: {steepness:.1f}  |  LR: {scheduler.get_last_lr()[0]:.5f}"
            )

    model.load_state_dict(best_state)
    return _extract_matrices(model, mapping_keys, patch_scales,
                              scale_class_matrix, scale_threshold_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – RESULT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_matrices(model, mapping_keys, patch_scales, orig_w, orig_t):
    new_w, new_t = {}, {}
    for p in patch_scales:
        w_vals = model.weights(p).detach().cpu().tolist()
        t_vals = model.thresholds(p).detach().cpu().tolist()
        new_w[p] = {
            k: (round(w, 6) if orig_w[p].get(k, 0.0) != 0.0 else 0.0)
            for k, w in zip(mapping_keys, w_vals)
        }
        new_t[p] = {
            k: (round(t, 6) if orig_t.get(p, {}).get(k, 0.0) != 0.0 else 0.0)
            for k, t in zip(mapping_keys, t_vals)
        }
    return new_w, new_t


def print_optimised_matrices(new_w, new_t):
    print("\n" + "═" * 72)
    print("  OPTIMISED  scale_class_matrix  (copy-paste into Daniel_MagiCLIP.py)")
    print("═" * 72)
    print("scale_class_matrix = {")
    for p, d in sorted(new_w.items(), reverse=True):
        print(f"    {p}: {{")
        for k, v in d.items():
            print(f'        "{k}": {v:.4f},')
        print("    },")
    print("}")

    print("\n" + "═" * 72)
    print("  OPTIMISED  scale_threshold_matrix")
    print("═" * 72)
    print("scale_threshold_matrix = {")
    for p, d in sorted(new_t.items(), reverse=True):
        if any(v > 1e-5 for v in d.values()):
            print(f"    {p}: {{")
            for k, v in d.items():
                print(f'        "{k}": {v:.4f},')
            print("    },")
    print("}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    """clip_prompts = [
        "drone view of a building", "drone view of a road", "drone view of a tree",
        "drone view of low vegetation", "drone view of background clutter",
        "drone view of a car", "drone view of a human",
    ]"""

    clip_prompts = [
    "building", "road", "tree",
    "low vegetation", "background clutter", 
    "car", "human"
    ]

    mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

    scale_class_matrix = {
        448: {
            "building": 1.1, "road": 1.5,  "tree": 1.2, "low_veg": 1.0,
            "clutter":  1.2, "car":  0.1,  "human": 0.1,
        },
        224: {
            "building": 1.0, "road": 1.3,  "tree": 1.1, "low_veg": 1.0,
            "clutter":  1.2, "car":  1.0,  "human": 1.2,
        },
    }

    scale_threshold_matrix = {
        448: {
            "building": 0.1, "road": 0.1, "tree": 0.1, "low_veg": 0.1,
            "clutter":  0.1, "car":  0.1, "human": 0.1,
        },
        224: {
            "building": 0.5, "road": 0.1, "tree": 0.5, "low_veg": 0.4,
            "clutter":  0.1, "car":  0.85, "human": 0.75,
        },
    }

    IMAGE_DIRS = [
        r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images",
        # Add more sequences here for better optimisation coverage
    ]

    image_paths = []
    for d in IMAGE_DIRS:
        image_paths += [
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    print(f"[*] Found {len(image_paths)} images.")

    opt_w, opt_t = optimize_weights(
        image_paths            = image_paths,
        clip_prompts           = clip_prompts,
        mapping_keys           = mapping_keys,
        scale_class_matrix     = scale_class_matrix,
        scale_threshold_matrix = scale_threshold_matrix,
        num_epochs             = 2000,
        lr                     = 1e-3,
        steepness_start        = 10.0,
        steepness_end          = 80.0,
        print_every            = 10,
        batch_size             = 32,    # must stay low for Grad-CAM backprop
    )

    print_optimised_matrices(opt_w, opt_t)
