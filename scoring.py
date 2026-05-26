"""
UAVid Semantic Segmentation Evaluation Script
==============================================
Computes per-class IoU, mIoU, and F0.5 / F1 / F2 scores by comparing
a model output image against a UAVid ground-truth label image.

Both images must use the UAVid colour-mask format (RGB PNG).

Set GT_IMAGE_PATH and PRED_IMAGE_PATH below, then run:
    python evaluate_segmentation.py
"""

import os
import json
import argparse
import numpy as np
from PIL import Image

# =============================================================================
# INPUT FILE PATHS — set these before running
# =============================================================================
patch_long = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1longprompt"  
patch_short = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1shortprompt"
magi_long = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1longpromptMagiCLIP"
magi_short = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1shortpromptMagiCLIP"

image_dir = magi_short  # Change this to the directory containing your prediction images

# =============================================================================
# CATEGORY COLOUR MAP
# Fill in the hex colour codes that correspond to each UAVid category.
# Format: "Category Name": (R, G, B)
# The hex code #RRGGBB converts to (0xRR, 0xGG, 0xBB).
# =============================================================================

CATEGORY_COLOURS = {
    "Building":           (128, 0, 0),   # e.g. (128, 0, 0)   — fill in hex: #______
    "Road":               (128, 64, 128),   # e.g. (128, 64, 128) — fill in hex: #______
    "Static Car":         (192, 0, 192),   # e.g. (192, 0, 192)  — fill in hex: #______
    "Tree":               (0, 128, 0),   # e.g. (0, 128, 0)    — fill in hex: #______
    "Low Vegetation":     (128, 128, 0),   # e.g. (128, 128, 0)  — fill in hex: #______
    "Human":              (64, 64, 0),   # e.g. (64, 0, 0)     — fill in hex: #______
    "Moving Car":         (64, 0, 128),   # e.g. (0, 0, 192)    — fill in hex: #______
    "Background Clutter": (0, 0, 0),   # e.g. (0, 0, 0)      — fill in hex: #______
}

# [AI]: Centralise category colours so all evaluation code uses a single
# authoritative source — this prevents subtle mismatches between
# saved prediction colours and ground-truth labels during scoring.

# =============================================================================
# OPTIONAL: MERGE STATIC CAR AND MOVING CAR INTO ONE CATEGORY
# Set to True to treat both car types as a single "Car" class.
# When enabled, both colours map to the same index and results are
# reported under "Car" instead of the two separate categories.
# =============================================================================

MERGE_CARS = True
MERGE_VEGETATION = False

# [AI]: Allow merging related classes to simplify evaluation when the
# [AI]: dataset or downstream task treats them as a single semantic label.

# =============================================================================
# END OF CONFIGURATION — no changes needed below this line
# =============================================================================


def hex_to_rgb(hex_str):
    """Convert a hex string like '#FF00AA' or 'FF00AA' to an (R, G, B) tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def validate_colour_map(colour_map):
    """Check that all category colours have been filled in."""
    missing = [name for name, colour in colour_map.items() if colour is None]
    if missing:
        raise ValueError(
            f"The following categories have no colour assigned:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nPlease fill in CATEGORY_COLOURS at the top of the script."
        )


def build_colour_index(colour_map, merge_cars=False, merge_vegetation=False):
    """
    Build a lookup from RGB tuple -> category index.
    Also returns the ordered list of category names.

    If merge_cars=True, both "Static Car" and "Moving Car" map to "Car".
    If merge_vegetation=True, both "Tree" and "Low Vegetation" map to "Vegetation".
    """
    if merge_cars or merge_vegetation:
        merged_categories = []
        seen_car = False
        seen_veg = False
        
        # Build the collapsed category list
        for name in colour_map:
            if merge_cars and name in ("Static Car", "Moving Car"):
                if not seen_car:
                    merged_categories.append("Car")
                    seen_car = True
            elif merge_vegetation and name in ("Tree", "Low Vegetation"):
                if not seen_veg:
                    merged_categories.append("Vegetation")
                    seen_veg = True
            else:
                merged_categories.append(name)

        # Assign indices based on merged list
        colour_to_idx = {}
        name_to_idx = {name: idx for idx, name in enumerate(merged_categories)}

        # Map each RGB colour to its new (or original) index
        for name, rgb in colour_map.items():
            if merge_cars and name in ("Static Car", "Moving Car"):
                colour_to_idx[rgb] = name_to_idx["Car"]
            elif merge_vegetation and name in ("Tree", "Low Vegetation"):
                colour_to_idx[rgb] = name_to_idx["Vegetation"]
            else:
                colour_to_idx[rgb] = name_to_idx[name]

        return merged_categories, colour_to_idx
    else:
        # Standard unmerged fallback
        categories = list(colour_map.keys())
        colour_to_idx = {rgb: idx for idx, (_, rgb) in enumerate(colour_map.items())}
        return categories, colour_to_idx


def image_to_label_array(image_path, colour_to_idx, num_classes):
    """
    Load an RGB label image and convert every pixel to a class index.
    Pixels whose colour does not match any category are assigned index -1.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = img.shape
    label = np.full((h, w), fill_value=-1, dtype=np.int32)

    for rgb, idx in colour_to_idx.items():
        # Build a boolean mask where all three channels match
        match = (
            (img[:, :, 0] == rgb[0]) &
            (img[:, :, 1] == rgb[1]) &
            (img[:, :, 2] == rgb[2])
        )
        label[match] = idx

    unmatched = np.sum(label == -1)
    if unmatched > 0:
        total = h * w
        print(f"  Warning: {unmatched}/{total} pixels ({100*unmatched/total:.2f}%) "
              f"in '{os.path.basename(image_path)}' did not match any category colour.")

    return label


def compute_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Compute a (num_classes x num_classes) confusion matrix where
    entry [i, j] is the number of pixels whose true class is i
    and predicted class is j.

    Pixels labelled -1 (unknown colour) are ignored in both images.
    """
    valid_mask = (gt_label >= 0) & (pred_label >= 0)
    gt_flat   = gt_label[valid_mask]
    pred_flat = pred_label[valid_mask]

    # Use numpy's bincount for efficiency
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    combined = gt_flat * num_classes + pred_flat
    counts = np.bincount(combined, minlength=num_classes * num_classes)
    confusion = counts.reshape((num_classes, num_classes))
    return confusion


def per_class_metrics(confusion, categories):
    """
    Derive per-class TP, FP, FN, TN from the confusion matrix, then compute
    IoU, F0.5, F1, and F2 for each category.

    For class i:
        TP = confusion[i, i]
        FP = sum of column i  - TP   (predicted as i but actually something else)
        FN = sum of row i     - TP   (actually i but predicted as something else)
        TN = total pixels - TP - FP - FN
    """
    num_classes = len(categories)
    total_pixels = confusion.sum()

    results = {}

    for i, name in enumerate(categories):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp   # column sum minus diagonal
        fn = confusion[i, :].sum() - tp   # row sum minus diagonal
        tn = total_pixels - tp - fp - fn

        # IoU (Jaccard Index)
        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F-scores: F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
        def f_score(beta, p, r):
            beta_sq = beta ** 2
            denom = beta_sq * p + r
            return (1 + beta_sq) * p * r / denom if denom > 0 else 0.0

        f05 = f_score(0.5, precision, recall)
        f1  = f_score(1.0, precision, recall)
        f2  = f_score(2.0, precision, recall)

        results[name] = {
            "TP":        int(tp),
            "FP":        int(fp),
            "FN":        int(fn),
            "TN":        int(tn),
            "Precision": precision,
            "Recall":    recall,
            "IoU":       iou,
            "F0.5":      f05,
            "F1":        f1,
            "F2":        f2,
        }

    return results


def compute_miou(per_class_results):
    """Mean IoU across all classes."""
    iou_values = [v["IoU"] for v in per_class_results.values()]
    return np.mean(iou_values)


def compute_weighted_miou(per_class_results):
    """
    Weighted mean IoU where each class is weighted by its share of total
    ground-truth pixels (TP + FN = GT pixel count for that class).
    Classes with zero GT pixels contribute zero weight and are effectively
    excluded from the average.
    """
    gt_counts  = np.array([v["TP"] + v["FN"] for v in per_class_results.values()], dtype=np.float64)
    iou_values = np.array([v["IoU"]           for v in per_class_results.values()], dtype=np.float64)

    total_gt = gt_counts.sum()
    if total_gt == 0:
        return 0.0

    weights = gt_counts / total_gt
    return float(np.dot(weights, iou_values))


def print_results(per_class_results, miou, image_name=None):
    """Pretty-print the evaluation results."""
    header = f"\n{'='*80}"
    if image_name:
        header += f"\nImage: {image_name}"
    header += f"\n{'='*80}"
    print(header)

    col_w = 20
    print(f"\n{'Category':<{col_w}} {'IoU':>8} {'F0.5':>8} {'F1':>8} {'F2':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 76)

    for name, m in per_class_results.items():
        print(f"{name:<{col_w}} {m['IoU']:>8.4f} {m['F0.5']:>8.4f} {m['F1']:>8.4f} "
              f"{m['F2']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}")

    print("-" * 76)
    print(f"\nmIoU: {miou:.4f}  ({miou*100:.2f}%)\n")

    # Confusion matrix counts
    # print(f"\n{'Category':<20} {'TP':>12} {'FP':>12} {'FN':>12} {'TN':>12}")
    # print("-" * 68)
    # for name, m in per_class_results.items():
    #     print(f"{name:<20} {m['TP']:>12,} {m['FP']:>12,} {m['FN']:>12,} {m['TN']:>12,}")


def evaluate_pair(gt_path, pred_path, colour_to_idx, categories):
    """Evaluate a single ground-truth / prediction image pair.

    Returns: per_class, miou, weighted_miou, confusion
    """
    num_classes = len(categories)

    gt_label   = image_to_label_array(gt_path,   colour_to_idx, num_classes)
    pred_label = image_to_label_array(pred_path, colour_to_idx, num_classes)

    if gt_label.shape != pred_label.shape:
        raise ValueError(
            f"Image size mismatch: GT is {gt_label.shape}, "
            f"prediction is {pred_label.shape}."
        )

    confusion = compute_confusion_matrix(gt_label, pred_label, num_classes)
    per_class = per_class_metrics(confusion, categories)
    miou      = compute_miou(per_class)
    weighted_miou = compute_weighted_miou(per_class)

    return per_class, miou, weighted_miou, confusion


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UAVid-style prediction masks against ground-truth labels.")
    parser.add_argument("--pred-dir", default=image_dir, help="Directory containing prediction images (default from file).")
    parser.add_argument("--root-dir", default=None, help="Root dataset directory containing 'Images', 'Labels', and 'Predictions' subfolders. If provided, overrides --pred-dir.")
    parser.add_argument("--no-pause", action="store_true", help="Do not prompt between images.")
    parser.add_argument("--output-json", default=None, help="Path to save JSON summary. If omitted, saves to <pred-dir>/evaluation_summary.json")
    args = parser.parse_args()

    # Resolve predictions directory.
    if args.root_dir:
        pred_dir = os.path.join(args.root_dir, 'Predictions')
    else:
        pred_dir = args.pred_dir

    # If the provided pred_dir looks like a dataset root (contains 'Images' or 'Labels'),
    # prefer its 'Predictions' subfolder when available.
    if os.path.isdir(pred_dir):
        # If pred_dir has no image files but has a 'Predictions' child, use it.
        files = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(files) == 0:
            alt = os.path.join(pred_dir, 'Predictions')
            if os.path.isdir(alt):
                pred_dir = alt

    # Finalize image_paths from resolved pred_dir
    if not os.path.isdir(pred_dir):
        print(f"Predictions directory not found: {pred_dir}")
        raise SystemExit(1)

    image_paths = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_paths) == 0:
        print(f"No images found in {pred_dir}")
        raise SystemExit(1)

    num_categories = 8 - (1 if MERGE_CARS else 0) - (1 if MERGE_VEGETATION else 0)
    per_class_running = np.zeros((num_categories, 10))
    miou_running = 0.0
    metrics_summary = {}

    categories, _ = build_colour_index(CATEGORY_COLOURS, merge_cars=MERGE_CARS, merge_vegetation=MERGE_VEGETATION)

    if MERGE_CARS:
        print("Note: 'Static Car' and 'Moving Car' are merged into 'Car'.")
    if MERGE_VEGETATION:
        print("Note: 'Tree' and 'Low Vegetation' are merged into 'Vegetation'.")

    for PRED_IMAGE_PATH in image_paths:
        base = os.path.basename(PRED_IMAGE_PATH)
        if args.root_dir:
            GT_IMAGE_PATH = os.path.join(args.root_dir, 'Labels', base)
        else:
            GT_IMAGE_PATH = PRED_IMAGE_PATH.replace("Predictions", "Labels")

        if not os.path.exists(GT_IMAGE_PATH):
            print(f"[-] Ground Truth not found for {base}. Skipping")
            continue

        per_class, miou, weighted_miou, confusion = evaluate_pair(
            GT_IMAGE_PATH, PRED_IMAGE_PATH, build_colour_index(CATEGORY_COLOURS, merge_cars=MERGE_CARS, merge_vegetation=MERGE_VEGETATION)[1],
            categories
        )

        # Build per-class list for running average
        per_class_list = []
        for category in per_class.keys():
            per_class_list.append([*per_class[category].values()])

        per_class_running += np.array(per_class_list)
        miou_running += miou

        # Compute global precision/recall (micro) across all classes
        total_TP = sum(v["TP"] for v in per_class.values())
        total_FP = sum(v["FP"] for v in per_class.values())
        total_FN = sum(v["FN"] for v in per_class.values())
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0

        # Save per-file metrics into the summary dict
        base = os.path.basename(PRED_IMAGE_PATH)
        metrics_summary[base] = [float(miou), float(weighted_miou), float(precision), float(recall)]

        # Print per-image results
        print_results(per_class, miou, image_name=base)


    # Finalize averages
    processed = len(metrics_summary)
    if processed == 0:
        print("No images processed. Exiting.")
        raise SystemExit(0)

    per_class_running /= processed
    miou_running /= processed

    per_class_overall = {}
    last_per_class = per_class
    for i, category in enumerate(last_per_class.keys()):
        per_class_overall[category] = {}
        for j, metric in enumerate(last_per_class[category].keys()):
            per_class_overall[category][metric] = per_class_running[i][j]

    print_results(per_class_overall, miou_running, image_name='Overall')

    # Write JSON summary
    out_json = args.output_json or os.path.join(args.pred_dir, 'evaluation_summary.json')
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(metrics_summary, fh, indent=2)
    print(f"Saved per-file JSON summary to: {out_json}")