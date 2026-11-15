import time
import numpy as np
import cv2

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection over Union (IoU / Jaccard Index):
    IoU = |Pred intersect GT| / |Pred union GT| = TP / (TP + FP + FN)
    Measures spatial overlap accuracy against ground truth segmentation.
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Dice Similarity Coefficient (F1-score of segmentation):
    Dice = (2 * |Pred intersect GT|) / (|Pred| + |GT|)
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total_elements = pred_bin.sum() + gt_bin.sum()

    if total_elements == 0:
        return 1.0
    return float(2.0 * intersection) / float(total_elements)

def compute_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """
    Computes pixel-level precision and recall.
    """
    pred_bin = (pred_mask > 0).astype(bool)
    gt_bin = (gt_mask > 0).astype(bool)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()

    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def compute_psnr(orig: np.ndarray, processed: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB):
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    mse = np.mean((orig.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))

def compute_edge_contrast(edge_image: np.ndarray, threshold: int = 50) -> float:
    """
    Computes genuine binary edge density (% of significant edge pixels).
    Thresholds continuous gradient maps (Sobel, Laplacian) to reject noise floor,
    preventing artificially inflated 90%+ density readings.
    """
    if edge_image is None or edge_image.size == 0:
        return 0.0
    
    # Apply thresholding to isolate true edge pixels
    edge_pixels = np.count_nonzero(edge_image >= threshold)
    return float(edge_pixels / edge_image.size * 100.0)

def benchmark_pipeline(stage_times: dict) -> dict:
    """
    Summarizes processing latency and estimated FPS.
    """
    total_time = sum(stage_times.values())
    fps = 1.0 / total_time if total_time > 0 else 0.0
    return {
        'stage_times_ms': {k: round(v * 1000, 2) for k, v in stage_times.items()},
        'total_latency_ms': round(total_time * 1000, 2),
        'throughput_fps': round(fps, 1)
    }
