import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
from pipeline.color_segmentation import segment_colors
from pipeline.morphology import clean_mask
from pipeline.edge_detection import apply_gaussian_blur, detect_canny_edges
from pipeline.shape_detector import detect_traffic_signs
from pipeline.metrics import compute_iou, compute_dice, compute_precision_recall

def run_benchmark():
    sample_files = glob.glob("sample_data/*.png")
    if not sample_files:
        print("No sample images found in sample_data. Run generate_samples.py first.")
        return

    print("=" * 105)
    print(" DIGITAL IMAGE PROCESSING: ENHANCED TRAFFIC SIGN DETECTION BENCHMARK")
    print("=" * 105)

    results = []

    for img_path in sorted(sample_files):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        gt_path = os.path.join("sample_data", "ground_truth", f"{base_name}_gt.png")

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Ground truth
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None

        # Execute Enhanced Pipeline with precise timing
        t0 = time.perf_counter()
        comb_mask, color_masks, _ = segment_colors(rgb)
        t_color = time.perf_counter() - t0

        t1 = time.perf_counter()
        morph_mask = clean_mask(comb_mask, kernel_shape="ellipse", kernel_size=3, open_iterations=1, close_iterations=2)
        t_morph = time.perf_counter() - t1

        t2 = time.perf_counter()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = apply_gaussian_blur(gray, kernel_size=5, sigma=1.4)
        canny_edges = detect_canny_edges(blurred, low_thresh=50, high_thresh=150)
        t_edge = time.perf_counter() - t2

        t3 = time.perf_counter()
        annotated, detected_signs, pred_contour_mask = detect_traffic_signs(
            rgb,
            morph_mask,
            color_masks,
            min_area=600,
            min_solidity=0.78,
            use_template_matching=True,
            apply_nms=True
        )
        t_detect = time.perf_counter() - t3

        total_time_ms = (t_color + t_morph + t_edge + t_detect) * 1000

        # Quantitative Performance Measures
        if gt_mask is not None:
            iou = compute_iou(pred_contour_mask, gt_mask)
            dice = compute_dice(pred_contour_mask, gt_mask)
            prec, rec = compute_precision_recall(pred_contour_mask, gt_mask)
        else:
            iou, dice, prec, rec = np.nan, np.nan, np.nan, np.nan

        detected_labels = [f"{s['label']} ({s['heuristic_score']*100:.0f}%)" for s in detected_signs]
        num_detected = len(detected_signs)

        results.append({
            'Test Image': filename,
            'Signs Found': num_detected,
            'Recognized Classes & Heuristic Scores': "; ".join(detected_labels) if detected_labels else "None",
            'IoU (%)': round(iou * 100, 2) if not np.isnan(iou) else "-",
            'Dice (%)': round(dice * 100, 2) if not np.isnan(dice) else "-",
            'Precision (%)': round(prec * 100, 2) if not np.isnan(prec) else "-",
            'Recall (%)': round(rec * 100, 2) if not np.isnan(rec) else "-",
            'Latency (ms)': round(total_time_ms, 2)
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("=" * 105)
    
    # Save CSV
    csv_path = "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Updated quantitative benchmark saved to: {csv_path}")

    # Summary
    numeric_iou = pd.to_numeric(df['IoU (%)'], errors='coerce')
    numeric_dice = pd.to_numeric(df['Dice (%)'], errors='coerce')
    numeric_prec = pd.to_numeric(df['Precision (%)'], errors='coerce')
    numeric_rec = pd.to_numeric(df['Recall (%)'], errors='coerce')
    numeric_lat = pd.to_numeric(df['Latency (ms)'], errors='coerce')

    print(f"Mean Intersection-over-Union (mIoU):       {numeric_iou.mean():.2f}%")
    print(f"Mean Dice Similarity Coefficient (F1):      {numeric_dice.mean():.2f}%")
    print(f"Mean Pixel Precision (Zero False Positives): {numeric_prec.mean():.2f}%")
    print(f"Mean Pixel Recall:                          {numeric_rec.mean():.2f}%")
    print(f"Average Pipeline Latency:                   {numeric_lat.mean():.2f} ms (~{1000.0/numeric_lat.mean():.1f} FPS)")
    print("=" * 105)

if __name__ == "__main__":
    run_benchmark()
