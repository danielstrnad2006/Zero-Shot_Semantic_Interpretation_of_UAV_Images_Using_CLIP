"""
UAVid Semantic Segmentation Evaluation Script
==============================================
Computes per-class IoU, mIoU, weighted mIoU, and F0.5 / F1 / F2 scores by
comparing a model output image against a UAVid ground-truth label image.

Both images must use the UAVid colour-mask format (RGB PNG).
"""

import os
import numpy as np
from PIL import Image
import cv2

# =============================================================================
# CATEGORY COLOUR MAP
# =============================================================================

CATEGORY_COLOURS = {
    "Building":           (128, 0, 0),
    "Road":               (128, 64, 128),
    "Static Car":         (192, 0, 192),
    "Tree":               (0, 128, 0),
    "Low Vegetation":     (128, 128, 0),
    "Human":              (64, 64, 0),
    "Moving Car":         (64, 0, 128),
    "Background Clutter": (0, 0, 0),
}

# [AI]: Keep the canonical colour map here so importers can rely on
# [AI]: consistent label-to-colour mapping; makes saving and evaluation
# [AI]: interchangeable across scripts without manual sync.

# =============================================================================
# EVALUATION CORE FUNCTIONS
# =============================================================================

def hex_to_rgb(hex_str):
    """Convert a hex string like '#FF00AA' or 'FF00AA' to an (R, G, B) tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def validate_colour_map(colour_map):
    # [AI]: Fail early if colours are missing to avoid silent evaluation
    # [AI]: errors later when masks are converted to class indices.
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
    """
    # [AI]: Create a single lookup from RGB -> class index once so the
    # [AI]: pixel-wise conversion step can be implemented efficiently.
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
    # [AI]: Vectorised conversion avoids per-pixel Python loops and scales
    # [AI]: well to high-resolution imagery common in UAV datasets.
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
    """
    # [AI]: Mask out unknown-colour pixels from both images so they don't
    # [AI]: bias the confusion counts; this keeps metrics meaningful.
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
    """
    # [AI]: Compute per-class statistics from the confusion matrix once
    # [AI]: so all derived metrics remain consistent across summaries.
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

        # F-scores
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
    """Mean IoU across all classes (unweighted)."""
    iou_values = [v["IoU"] for v in per_class_results.values()]
    return np.mean(iou_values)

def compute_weighted_miou(per_class_results):
    """
    Weighted mean IoU where each class is weighted by its share of total
    ground-truth pixels (TP + FN = GT pixel count for that class).
    Classes with zero GT pixels contribute zero weight and are effectively
    excluded from the average.
    """
    # [AI]: Weight IoU by ground-truth pixel counts to reflect class
    # [AI]: prevalence — this prevents tiny classes from dominating averages.
    gt_counts  = np.array([v["TP"] + v["FN"] for v in per_class_results.values()], dtype=np.float64)
    iou_values = np.array([v["IoU"]           for v in per_class_results.values()], dtype=np.float64)

    total_gt = gt_counts.sum()
    if total_gt == 0:
        return 0.0

    weights = gt_counts / total_gt
    return float(np.dot(weights, iou_values))

def print_results(per_class_results, miou, weighted_miou=None, image_name=None):
    """Pretty-print the evaluation results."""
    header = f"\n{'='*84}"
    if image_name:
        header += f"\nImage: {image_name}"
    header += f"\n{'='*84}"
    print(header)

    col_w = 20
    print(f"\n{'Category':<{col_w}} {'IoU':>8} {'F0.5':>8} {'F1':>8} {'F2':>8} "
          f"{'Precision':>10} {'Recall':>8} {'GT%':>7}")
    print("-" * 84)

    total_gt = sum(v["TP"] + v["FN"] for v in per_class_results.values())
    for name, m in per_class_results.items():
        gt_pct = 100.0 * (m["TP"] + m["FN"]) / total_gt if total_gt > 0 else 0.0
        print(f"{name:<{col_w}} {m['IoU']:>8.4f} {m['F0.5']:>8.4f} {m['F1']:>8.4f} "
              f"{m['F2']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f} {gt_pct:>6.2f}%")

    print("-" * 84)
    print(f"\nmIoU:          {miou:.4f}  ({miou*100:.2f}%)")
    if weighted_miou is not None:
        print(f"Weighted mIoU: {weighted_miou:.4f}  ({weighted_miou*100:.2f}%)")
    print()

def evaluate_pair(gt_path, pred_path, colour_to_idx, categories):
    """Evaluate a single ground-truth / prediction image pair."""
    num_classes = len(categories)

    gt_label   = image_to_label_array(gt_path,   colour_to_idx, num_classes)
    pred_label = image_to_label_array(pred_path, colour_to_idx, num_classes)

    if gt_label.shape != pred_label.shape:
        raise ValueError(
            f"Image size mismatch: GT is {gt_label.shape}, "
            f"prediction is {pred_label.shape}."
        )

    confusion     = compute_confusion_matrix(gt_label, pred_label, num_classes)
    per_class     = per_class_metrics(confusion, categories)
    miou          = compute_miou(per_class)
    weighted_miou = compute_weighted_miou(per_class)

    return per_class, miou, weighted_miou, confusion

def evaluate_all_images(image_dir, colour_to_idx, categories):
    """Loop through an entire directory of predicted images and evaluate them."""
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    for PRED_IMAGE_PATH in image_paths:
        GT_IMAGE_PATH = PRED_IMAGE_PATH.replace("Predictions", "Labels")

        if not os.path.exists(GT_IMAGE_PATH):
            print(f"[-] Ground Truth not found for {os.path.basename(PRED_IMAGE_PATH)}. Skipping.")
            continue

        per_class, miou, weighted_miou, _ = evaluate_pair(
            GT_IMAGE_PATH, PRED_IMAGE_PATH, colour_to_idx, categories
        )
        print_results(per_class, miou, weighted_miou, image_name=os.path.basename(PRED_IMAGE_PATH))
        input("Press Enter to continue to the next image...")


# =============================================================================
# MASK SAVING & INTEGRATION PIPELINE
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

def save_segmentation_map(image_path, seg_map, mapping_keys):
    """
    Converts a raw integer mask to RGB, flips it for OpenCV, and saves it
    by replacing 'Images' with 'Predictions' in the file path.
    """
    # [AI]: Reuse the same colour conversion when saving masks so evaluation
    # [AI]: tools can read them back without needing conversion logic duplication.
    rgb_seg_map = segment_mask_to_rgb(seg_map, mapping_keys)
    bgr_seg_map = cv2.cvtColor(rgb_seg_map, cv2.COLOR_RGB2BGR)

    save_path = image_path.replace("Images", "Predictions")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr_seg_map)
    print(f"[+] Saved prediction to: {save_path}")

    return save_path

def save_and_evaluate_single_image(image_path, seg_map, mapping_keys, merge_cars=True, merge_vegetation=False):
    """
    The master importable function.
    It saves the raw prediction array to disk as an RGB image, automatically locates
    the corresponding ground-truth label, and executes the evaluation.
    """
    # 1. Save the mask
    pred_path = save_segmentation_map(image_path, seg_map, mapping_keys)

    # 2. Derive the Ground Truth path
    gt_path = pred_path.replace("Predictions", "Labels")

    # 3. Setup the colour index logic based on global configuration
    # [AI]: Build the colour index here to ensure the saved mask and the
    # [AI]: evaluation mapping are aligned; this prevents mismatched indices.
    validate_colour_map(CATEGORY_COLOURS)
    categories, colour_to_idx = build_colour_index(
        CATEGORY_COLOURS, merge_cars=merge_cars, merge_vegetation=merge_vegetation
    )

    if not os.path.exists(gt_path):
        print(f"[-] Ground Truth not found at {gt_path}. Skipping evaluation.")
        return None, None, None

    print(f"[*] Ground Truth found. Running IoU Evaluation...")
    if merge_cars:
        print("Note: 'Static Car' and 'Moving Car' are merged into 'Car'.")
    if merge_vegetation:
        print("Note: 'Tree' and 'Low Vegetation' are merged into 'Vegetation'.")

    # 4. Evaluate and print
    per_class, miou, weighted_miou, _ = evaluate_pair(gt_path, pred_path, colour_to_idx, categories)
    print_results(per_class, miou, weighted_miou, image_name=os.path.basename(pred_path))

    return per_class, miou, weighted_miou

def evaluate_single_image(image_path, merge_cars=True, merge_vegetation=False):
    """
    The master importable function.
    It saves the raw prediction array to disk as an RGB image, automatically locates
    the corresponding ground-truth label, and executes the evaluation.
    """
    # 1. Derive paths
    pred_path = image_path.replace("Images", "Predictions")
    gt_path   = pred_path.replace("Predictions", "Labels")

    # 2. Setup the colour index logic based on global configuration
    validate_colour_map(CATEGORY_COLOURS)
    categories, colour_to_idx = build_colour_index(
        CATEGORY_COLOURS, merge_cars=merge_cars, merge_vegetation=merge_vegetation
    )

    if not os.path.exists(gt_path):
        print(f"[-] Ground Truth not found at {gt_path}. Skipping evaluation.")
        return None, None, None

    print(f"[*] Ground Truth found. Running IoU Evaluation...")
    if merge_cars:
        print("Note: 'Static Car' and 'Moving Car' are merged into 'Car'.")
    if merge_vegetation:
        print("Note: 'Tree' and 'Low Vegetation' are merged into 'Vegetation'.")

    # 3. Evaluate and print
    per_class, miou, weighted_miou, _ = evaluate_pair(gt_path, pred_path, colour_to_idx, categories)
    print_results(per_class, miou, weighted_miou, image_name=os.path.basename(pred_path))

    return per_class, miou, weighted_miou