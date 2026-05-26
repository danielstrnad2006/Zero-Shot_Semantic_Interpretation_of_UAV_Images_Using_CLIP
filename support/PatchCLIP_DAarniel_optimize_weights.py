"""
optimize_weights.py
────────────────────────────────────────────────────────────────────────────────
Adam-based gradient optimizer for scale_class_matrix and scale_threshold_matrix.

KEY IDEAS
─────────
1.  CLIP is run ONCE (expensive).  The raw logit vectors for every patch at
    every scale are cached in RAM.  After that each epoch is cheap pure-Python
    / torch arithmetic.

2.  The pipeline is made end-to-end differentiable:
      • Weights  → stored in log-space (exp keeps them ≥ 0)
      • Thresholds → stored in logit-space (sigmoid keeps them in (0,1))
      • Hard gate  → replaced by a differentiable sigmoid gate:
            gate = sigmoid(steepness × (raw_prob − threshold))
        where steepness is gradually annealed upward so gradients don't
        vanish at the start of training.
      • argmax → replaced by soft per-class IoU computed from raw probabilities.

3.  FROZEN ZEROS  – any weight/threshold that is 0 in the original matrices
    stays 0 throughout optimisation (freeze mask).

4.  PATCH-LEVEL COMPUTATION at the finest scale (56 px).
    Each coarser patch contains several fine patches; we precompute which
    coarse patch each fine patch belongs to, so the spatial bookkeeping costs
    nothing during training.

5.  After optimisation the learned values are printed in copy-pasteable format
    so you can drop them straight back into Daniel_Patch.py.

USAGE
─────
    python optimize_weights.py

    Adjust IMAGE_DIR and hyper-parameters at the bottom of this file.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import clip
from PIL import Image
from tqdm import tqdm

from support.scoring_general import (
    CATEGORY_COLOURS,
    validate_colour_map,
    build_colour_index,
    image_to_label_array,
)

# [AI]: CLIP forward passes are expensive; we cache per-image features
# [AI]: once so subsequent optimisation epochs are pure, cheap tensor math.

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – FEATURE CACHING
# ══════════════════════════════════════════════════════════════════════════════

def _gt_to_clip_index_map(gt_categories: list, mapping_keys: list) -> list:
    """
    Build gt_to_clip[gt_idx] = clip_idx.

    Normalises names and applies a small alias table so the automatic mapping
    handles the naming differences between scoring_new categories and the CLIP
    prompt keys used in Daniel_Patch.
    """
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
                f"GT category '{name}' (normalised: '{n}') cannot be mapped "
                f"to any of the CLIP keys: {mapping_keys}"
            )
        result.append(clip_lookup[n])

    return result          # length == len(gt_categories)


def precompute_patch_cache(
    image_paths: list,
    clip_prompts: list,
    mapping_keys: list,
    patch_scales: list,
    colour_to_idx: dict,
    num_classes_gt: int,
    device: str,
    batch_size: int = 128,
) -> list:
    """
    Run CLIP once for every (image, scale) pair and cache results.

    The *base* spatial resolution used for optimisation is the FINEST scale
    (smallest patch size).  For each base-scale patch we record:
        • logits  at each scale  (N_base, C_clip)  – the coarser patch that
          contains this base patch shares its logit vector.
        • gt_dist (N_base, C_gt) – fraction of GT pixels belonging to each
          class inside this patch.

    Memory footprint per image ≈ #fine_patches × #scales × C × 4 bytes.
    For UAVid (3840×2160) with base=56 and 4 scales this is ~70 MB per image –
    easily manageable on a modern workstation.
    """
    patch_scales_desc = sorted(patch_scales, reverse=True)
    base_scale        = min(patch_scales)   # finest resolution

    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale   = model.logit_scale.exp()

    cache = []

    for image_path in tqdm(image_paths, desc="Caching CLIP features"):
        # ── Ground-truth path check ─────────────────────────────────────────
        pred_path = image_path.replace("Images", "Predictions")
        gt_path   = pred_path.replace("Predictions", "Labels")
        if not os.path.exists(gt_path):
            print(f"  [skip] no GT for {os.path.basename(image_path)}")
            continue

        original_image  = Image.open(image_path).convert("RGB")
        W_orig, H_orig  = original_image.size
        gt_label        = image_to_label_array(gt_path, colour_to_idx, num_classes_gt)

        # ── Base-scale patch grid ────────────────────────────────────────────
        B          = base_scale
        pad_w      = (-W_orig) % B
        pad_h      = (-H_orig) % B
        cols_base  = (W_orig + pad_w) // B
        rows_base  = (H_orig + pad_h) // B
        N_base     = rows_base * cols_base

        # GT class distribution per base patch
        gt_dists = np.zeros((N_base, num_classes_gt), dtype=np.float32)
        for idx_b, (r, c) in enumerate(
            (r, c) for r in range(rows_base) for c in range(cols_base)
        ):
            x0, y0 = c * B,  r * B
            x1, y1 = min(x0 + B, W_orig), min(y0 + B, H_orig)
            patch_gt = gt_label[y0:y1, x0:x1]
            valid    = patch_gt[patch_gt >= 0]
            if len(valid):
                for cls in range(num_classes_gt):
                    gt_dists[idx_b, cls] = float(np.sum(valid == cls)) / len(valid)

        # ── Per-scale CLIP logits ────────────────────────────────────────────
        logits_per_scale = {}

        for p_size in patch_scales_desc:
            pad_wp  = (-W_orig) % p_size
            pad_hp  = (-H_orig) % p_size
            pw      = W_orig + pad_wp
            ph      = H_orig + pad_hp
            cols    = pw // p_size
            rows    = ph // p_size

            padded_img = Image.new("RGB", (pw, ph), color=(0, 0, 0))
            padded_img.paste(original_image, (0, 0))

            patches = []
            for rr in range(rows):
                for cc in range(cols):
                    x0, y0 = cc * p_size, rr * p_size
                    patches.append(
                        preprocess(padded_img.crop((x0, y0, x0 + p_size, y0 + p_size)))
                    )

            # Encode in batches
            btensor   = torch.stack(patches).to(device)
            raw_list  = []
            with torch.no_grad():
                for i in range(0, len(btensor), batch_size):
                    chunk = btensor[i : i + batch_size]
                    with torch.amp.autocast("cuda"):
                        feats = model.encode_image(chunk)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        raw_list.append((logit_scale * (feats @ text_features.T)).float().cpu())
            patch_logits = torch.cat(raw_list, dim=0)   # (N_coarse, C)

            # Map each base patch to its containing coarse patch via centre point
            base_to_coarse = np.empty(N_base, dtype=np.int64)
            for idx_b, (rb, cb) in enumerate(
                (r, c) for r in range(rows_base) for c in range(cols_base)
            ):
                cy         = rb * B + B // 2
                cx         = cb * B + B // 2
                coarse_r   = min(cy // p_size, rows - 1)
                coarse_c   = min(cx // p_size, cols - 1)
                base_to_coarse[idx_b] = coarse_r * cols + coarse_c

            logits_per_scale[p_size] = patch_logits[base_to_coarse]   # (N_base, C)

        cache.append({
            "logits_per_scale": logits_per_scale,                         # dict
            "gt_dist":          torch.from_numpy(gt_dists),               # (N_base, C_gt)
            "image_path":       image_path,
        })

    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DIFFERENTIABLE FUSION MODULE
# ══════════════════════════════════════════════════════════════════════════════

class FusionWeights(nn.Module):
    """
    Learnable wrapper around scale_class_matrix and scale_threshold_matrix.

    Parameterisation
    ────────────────
    • Weights    stored in log-space  → exp(log_w) is always ≥ 0.
    • Thresholds stored in logit-space → sigmoid(raw_t) is always in (0,1).
    • Frozen zeros remain exactly 0 via a non-differentiable mask buffer.
    """

    def __init__(
        self,
        scale_class_matrix:     dict,
        scale_threshold_matrix: dict,
        mapping_keys:           list,
    ):
        super().__init__()
        self.mapping_keys  = mapping_keys
        self.patch_scales  = sorted(scale_class_matrix.keys(), reverse=True)

        for p in self.patch_scales:
            w_init = [scale_class_matrix[p].get(k, 0.0)              for k in mapping_keys]
            t_init = [scale_threshold_matrix.get(p, {}).get(k, 0.0)  for k in mapping_keys]

            # Freeze mask: 1.0 where original value is 0 → stays 0
            # [AI]: Keep zeros frozen to preserve original design constraints
            # [AI]: (e.g. permanently disabled classes) during optimisation.
            w_freeze = torch.tensor([1.0 if w == 0.0 else 0.0 for w in w_init])
            t_freeze = torch.tensor([1.0 if t == 0.0 else 0.0 for t in t_init])
            self.register_buffer(f"w_freeze_{p}", w_freeze)
            self.register_buffer(f"t_freeze_{p}", t_freeze)

            # Log-space weights (small ε for 0→log safety; masked out anyway)
            log_w = torch.tensor([math.log(max(w, 1e-6)) for w in w_init], dtype=torch.float32)
            self.register_parameter(f"log_w_{p}", nn.Parameter(log_w))

            # Logit-space thresholds
            def _logit(t):
                if t <= 0.0:
                    return -10.0        # sigmoid(-10) ≈ 4.5e-5 ≈ 0
                t = max(min(t, 1 - 1e-6), 1e-6)
                return math.log(t / (1.0 - t))

            raw_t = torch.tensor([_logit(t) for t in t_init], dtype=torch.float32)
            self.register_parameter(f"raw_t_{p}", nn.Parameter(raw_t))

    # ─── Accessors ───────────────────────────────────────────────────────────

    def weights(self, p_size: int) -> torch.Tensor:
        mask  = getattr(self, f"w_freeze_{p_size}")
        log_w = getattr(self, f"log_w_{p_size}")
        return torch.exp(log_w) * (1.0 - mask)     # (C,)

    def thresholds(self, p_size: int) -> torch.Tensor:
        mask  = getattr(self, f"t_freeze_{p_size}")
        raw_t = getattr(self, f"raw_t_{p_size}")
        return torch.sigmoid(raw_t) * (1.0 - mask) # (C,)

    # ─── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        logits_per_scale: dict,
        temperature:      float = 1.0,
        gate_steepness:   float = 50.0,
    ) -> torch.Tensor:
        """
        logits_per_scale: {p_size: Tensor(N_base, C_clip)}  (on any device)
        Returns: fused Tensor(N_base, C_clip)  — raw (un-normalised) scores.
        """
        dev   = next(self.parameters()).device
        N, C  = next(iter(logits_per_scale.values())).shape
        fused = torch.zeros(N, C, device=dev)

        for p in self.patch_scales:
            logits = logits_per_scale[p].to(dev)            # (N, C)
            w      = self.weights(p)                        # (C,)
            thr    = self.thresholds(p)                     # (C,)

            raw_probs = torch.softmax(logits, dim=-1)       # (N, C)

            # Differentiable soft gate  (replaces the hard-zero activation)
            # [AI]: Use a steepness-controlled sigmoid so training starts
            # [AI]: with smooth gradients (small steepness) and converges
            # [AI]: to near-hard gating as steepness increases.
            gate  = torch.sigmoid(gate_steepness * (raw_probs - thr.unsqueeze(0)))

            probs = torch.softmax(logits / temperature, dim=-1) * gate
            fused = fused + probs * w.unsqueeze(0)

        return fused                                        # (N, C)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DIFFERENTIABLE SOFT mIoU LOSS
# ══════════════════════════════════════════════════════════════════════════════

def soft_miou_loss(
    fused_probs: torch.Tensor,   # (N, C_clip)
    gt_dist:     torch.Tensor,   # (N, C_gt)
    gt_to_clip:  list,           # length C_gt; maps gt_idx → clip_idx
) -> torch.Tensor:
    """
    Soft (differentiable) mean-IoU.

    fused_probs need not be normalised; we L1-normalise per patch internally.
    gt_dist     is the fraction of GT pixels per class in each patch.

    Soft IoU per class c:
        TP_c = Σ_n  p_nc · g_nc
        FP_c = Σ_n  p_nc · (1 − g_nc)
        FN_c = Σ_n  (1 − p_nc) · g_nc
        IoU_c = TP_c / (TP_c + FP_c + FN_c + ε)
    """
    dev    = fused_probs.device
    N, C   = fused_probs.shape
    C_gt   = gt_dist.shape[1]

    # Normalise predictions
    p_norm = fused_probs / (fused_probs.sum(dim=-1, keepdim=True) + 1e-8)   # (N, C)

    # Remap GT distribution to CLIP class order
    # [AI]: Align GT columns with CLIP prompt order so IoU is computed
    # [AI]: between matching semantic categories rather than arbitrary indices.
    g = torch.zeros(N, C, device=dev)
    for gt_idx, clip_idx in enumerate(gt_to_clip):
        g[:, clip_idx] += gt_dist[:, gt_idx].to(dev)

    iou_per_class = []
    for c in range(C):
        p_c  = p_norm[:, c]
        g_c  = g[:, c]
        tp   = (p_c * g_c).sum()
        fp   = (p_c * (1.0 - g_c)).sum()
        fn   = ((1.0 - p_c) * g_c).sum()
        iou_per_class.append(tp / (tp + fp + fn + 1e-8))

    return torch.stack(iou_per_class).mean()    # scalar, maximise this


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – MAIN OPTIMISER LOOP
# ══════════════════════════════════════════════════════════════════════════════

def optimize_weights(
    image_paths:            list,
    clip_prompts:           list,
    mapping_keys:           list,
    scale_class_matrix:     dict,
    scale_threshold_matrix: dict,
    *,
    num_epochs:      int   = 150,
    lr:              float = 3e-3,
    temperature:     float = 1.0,
    # Gate steepness is annealed: starts low (smooth gradients) → ends high (sharp gate)
    steepness_start: float = 10.0,
    steepness_end:   float = 80.0,
    print_every:     int   = 10,
    batch_size:      int   = 128,
):
    """
    Entry-point.  Returns (optimized_scale_class_matrix, optimized_scale_threshold_matrix)
    in the exact same nested-dict format used by Daniel_Patch.py.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    # ── GT setup ─────────────────────────────────────────────────────────────
    validate_colour_map(CATEGORY_COLOURS)
    gt_categories, colour_to_idx = build_colour_index(
        CATEGORY_COLOURS, merge_cars=True, merge_vegetation=False
    )
    num_classes_gt = len(gt_categories)
    print(f"[*] GT categories ({num_classes_gt}): {gt_categories}")

    gt_to_clip = _gt_to_clip_index_map(gt_categories, mapping_keys)
    print(f"[*] GT→CLIP map: {list(zip(gt_categories, [mapping_keys[i] for i in gt_to_clip]))}")

    # ── Feature cache (CLIP forward, one-time) ────────────────────────────────
    # [AI]: Precompute cache once to amortise expensive model forwards
    # [AI]: across all optimisation epochs; this is how training stays fast.
    patch_scales = sorted(scale_class_matrix.keys(), reverse=True)
    cache = precompute_patch_cache(
        image_paths, clip_prompts, mapping_keys, patch_scales,
        colour_to_idx, num_classes_gt, device, batch_size=batch_size,
    )
    if not cache:
        raise RuntimeError("No valid (image, GT) pairs found. Check image_dir and path structure.")
    print(f"[*] Cached {len(cache)} image(s).\n")

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model     = FusionWeights(scale_class_matrix, scale_threshold_matrix, mapping_keys).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    best_miou  = -1.0
    best_state = None

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        # Anneal gate steepness: low → high over training
        t_frac    = (epoch - 1) / max(num_epochs - 1, 1)
        steepness = steepness_start + t_frac * (steepness_end - steepness_start)

        optimizer.zero_grad()

        total_miou = 0.0
        for item in cache:
            fused    = model(item["logits_per_scale"], temperature=temperature,
                             gate_steepness=steepness)
            miou_val = soft_miou_loss(fused, item["gt_dist"], gt_to_clip)
            (-miou_val).backward()                   # accumulate gradients
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
                f"Soft mIoU: {avg_miou:.4f}  |  "
                f"Best: {best_miou:.4f}  |  "
                f"Steepness: {steepness:.1f}  |  "
                f"LR: {scheduler.get_last_lr()[0]:.5f}"
            )

    # ── Extract results ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    return _extract_matrices(model, mapping_keys, patch_scales,
                              scale_class_matrix, scale_threshold_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – RESULT EXTRACTION + PRETTY-PRINT
# ══════════════════════════════════════════════════════════════════════════════

def _extract_matrices(model, mapping_keys, patch_scales, orig_w, orig_t):
    """Return optimised matrices in the original nested-dict format."""
    new_w = {}
    new_t = {}
    for p in patch_scales:
        w_vals = model.weights(p).detach().cpu().tolist()
        t_vals = model.thresholds(p).detach().cpu().tolist()

        orig_w_row = orig_w[p]
        orig_t_row = orig_t.get(p, {})

        new_w[p] = {
            k: (round(w, 6) if orig_w_row.get(k, 0.0) != 0.0 else 0.0)
            for k, w in zip(mapping_keys, w_vals)
        }
        new_t[p] = {
            k: (round(t, 6) if orig_t_row.get(k, 0.0) != 0.0 else 0.0)
            for k, t in zip(mapping_keys, t_vals)
        }
    return new_w, new_t


def print_optimised_matrices(new_w, new_t):
    print("\n" + "═" * 72)
    print("  OPTIMISED  scale_class_matrix  (copy-paste into Daniel_Patch.py)")
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
        active = {k: v for k, v in d.items() if v > 1e-5}
        if active:
            print(f"    {p}: {{")
            for k, v in d.items():
                print(f'        "{k}": {v:.4f},')
            print("    },")
    print("}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Copy your setup from Daniel_Patch.py ─────────────────────────────────
    
    """clip_prompts = [
        "drone view of a building",
        "drone view of a road",
        "drone view of a tree",
        "drone view of low vegetation",
        "drone view of background clutter",
        "drone view of a car",
        "drone view of a human",
    ]"""

    clip_prompts = [
    "building", "road", "tree",
    "low vegetation", "background clutter", 
    "car", "human"
    ]

    mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

    scale_class_matrix = {
        448: {"building": 0.8, "road": 0.8,  "tree": 0.8,  "low_veg": 0.8,
              "clutter": 0.8, "car": 0.0,  "human": 0.0},
        224: {"building": 1.0, "road": 1.0,  "tree": 1.0,  "low_veg": 1.0,
              "clutter": 1.5, "car": 0.0,  "human": 0.0},
        112: {"building": 1.2, "road": 1.2,  "tree": 1.1,  "low_veg": 1.1,
              "clutter": 1.7, "car": 1.4,  "human": 0.05},
        56:  {"building": 0.5, "road": 1.0,  "tree": 1.0,  "low_veg": 1.0,
              "clutter": 1.2, "car": 1.6,  "human": 5.0},
    }

    scale_threshold_matrix = {
        448: {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
              "clutter": 0.0, "car": 0.0, "human": 0.0},
        224: {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
              "clutter": 0.0, "car": 0.0, "human": 0.0},
        112: {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
              "clutter": 0.0, "car": 0.7, "human": 0.0},
        56:  {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
              "clutter": 0.0, "car": 0.7, "human": 0.85},
    }

    # ── Point at your validation images ──────────────────────────────────────
    # Add as many sequence directories as you have.
    IMAGE_DIRS = [
        r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images",
        # r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq16\Images",
    ]

    image_paths = []
    for d in IMAGE_DIRS:
        image_paths += [
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    print(f"[*] Found {len(image_paths)} images across {len(IMAGE_DIRS)} sequence(s).")

    # ── Run optimisation ──────────────────────────────────────────────────────
    opt_w, opt_t = optimize_weights(
        image_paths            = image_paths,
        clip_prompts           = clip_prompts,
        mapping_keys           = mapping_keys,
        scale_class_matrix     = scale_class_matrix,
        scale_threshold_matrix = scale_threshold_matrix,
        num_epochs             = 2000,
        lr                     = 1e-3,
        temperature            = 1.0,
        steepness_start        = 10.0,   # smooth at start → clean gradients
        steepness_end          = 80.0,   # sharp at end → matches hard-gate behaviour
        print_every            = 10,
        batch_size             = 128,
    )

    print_optimised_matrices(opt_w, opt_t)
