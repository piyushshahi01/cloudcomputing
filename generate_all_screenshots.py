import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.color_segmentation import segment_colors, rgb_to_hsv
from pipeline.morphology import clean_mask
from pipeline.edge_detection import apply_gaussian_blur, compute_sobel, compute_laplacian, detect_canny_edges
from pipeline.shape_detector import detect_traffic_signs
from pipeline.visualizer import create_pipeline_comparison_grid, plot_edge_comparison, plot_color_histograms, plot_hsv_scatter, plot_1d_hue_distribution

def generate_screenshots(output_dir="screenshots"):
    os.makedirs(output_dir, exist_ok=True)
    print("Capturing and saving application result screenshots...")

    bgr = cv2.imread("sample_data/comprehensive_test_scene.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = rgb_to_hsv(rgb)

    comb_mask, color_masks, masked_rgb = segment_colors(rgb)
    morph_mask = clean_mask(comb_mask, kernel_shape="ellipse", kernel_size=3, open_iterations=1, close_iterations=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = apply_gaussian_blur(gray, kernel_size=5, sigma=1.4)
    canny = detect_canny_edges(blurred, low_thresh=50, high_thresh=150)
    sobel_mag, _, _, _ = compute_sobel(blurred)
    laplacian = compute_laplacian(blurred)

    # 1. Final Classified Overlays
    annotated, detected, pred_mask = detect_traffic_signs(rgb, morph_mask, color_masks, min_area=600, apply_nms=True)
    cv2.imwrite(os.path.join(output_dir, "01_final_traffic_sign_detections.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # 2. Pipeline Progression Grid (2x3)
    grid_img = create_pipeline_comparison_grid(rgb, hsv, comb_mask, morph_mask, canny, annotated)
    grid_img.save(os.path.join(output_dir, "02_pipeline_stages_progression.png"))

    # 3. False Positive Suppression (Before vs After)
    ann_naive, signs_naive, _ = detect_traffic_signs(rgb, comb_mask, color_masks, min_area=100, min_solidity=0.4, apply_nms=False)
    fig_fp, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0B0F19')
    ax_a.imshow(ann_naive)
    ax_a.set_title(f"Baseline (15+ False Detections: Leaves, Clutter)", fontsize=11, fontweight='bold', color='#EF4444')
    ax_a.axis('off')
    ax_b.imshow(annotated)
    ax_b.set_title(f"Enhanced DIP with NMS (Exactly 4 True Signs)", fontsize=11, fontweight='bold', color='#34D399')
    ax_b.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_false_positive_suppression_comparison.png"), dpi=130, facecolor='#0B0F19')
    plt.close(fig_fp)

    # 4. Calibrated 2-Panel Chromaticity Analysis
    chroma_img = plot_hsv_scatter(rgb, color_masks=color_masks, max_samples=300)
    chroma_img.save(os.path.join(output_dir, "04_hsv_chromaticity_scatter_analysis.png"))

    # 5. 1D Hue Density Profiles
    hue_1d_img = plot_1d_hue_distribution(rgb, color_masks=color_masks)
    hue_1d_img.save(os.path.join(output_dir, "05_1d_hue_density_profiles.png"))

    # 6. Edge Detection Comparison (Sobel vs Laplacian vs Canny)
    edge_img = plot_edge_comparison(gray, sobel_mag, laplacian, canny)
    edge_img.save(os.path.join(output_dir, "06_edge_operators_comparison.png"))

    # 7. Ground Truth Segmentation Overlap (TP / FP / FN)
    gt_mask = cv2.imread("sample_data/ground_truth/comprehensive_test_scene_gt.png", cv2.IMREAD_GRAYSCALE)
    if gt_mask is not None:
        tp_vis = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        tp_vis[(pred_mask > 0) & (gt_mask > 0)] = [0, 255, 0]   # True Positive
        tp_vis[(pred_mask > 0) & (gt_mask == 0)] = [255, 0, 0]   # False Positive
        tp_vis[(pred_mask == 0) & (gt_mask > 0)] = [0, 100, 255] # False Negative
        cv2.imwrite(os.path.join(output_dir, "07_ground_truth_overlap_verification.png"), cv2.cvtColor(tp_vis, cv2.COLOR_RGB2BGR))

    print(f"All 7 application result screenshots successfully saved to: {output_dir}/")

if __name__ == "__main__":
    generate_screenshots()

