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

def generate_visual_assets(output_dir="report_assets"):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating comprehensive visual assets for report and presentation...")

    # 1. Pipeline Stages Grid on the Comprehensive Test Scene
    bgr_comp = cv2.imread("sample_data/comprehensive_test_scene.png")
    rgb_comp = cv2.cvtColor(bgr_comp, cv2.COLOR_BGR2RGB)
    
    comb_mask, color_masks, _ = segment_colors(rgb_comp)
    morph_mask = clean_mask(comb_mask, kernel_shape="ellipse", kernel_size=3, open_iterations=1, close_iterations=2)
    gray = cv2.cvtColor(rgb_comp, cv2.COLOR_RGB2GRAY)
    blurred = apply_gaussian_blur(gray, kernel_size=5, sigma=1.4)
    canny = detect_canny_edges(blurred, low_thresh=50, high_thresh=150)
    annotated, _, _ = detect_traffic_signs(rgb_comp, morph_mask, color_masks, min_area=600, apply_nms=True)

    grid_img = create_pipeline_comparison_grid(
        rgb_comp,
        rgb_to_hsv(rgb_comp),
        comb_mask,
        morph_mask,
        canny,
        annotated
    )
    grid_img.save(os.path.join(output_dir, "pipeline_stages_grid.png"))

    # 2. Edge Operators Comparison
    sobel_mag, _, _, _ = compute_sobel(blurred)
    laplacian = compute_laplacian(blurred)
    edge_fig = plot_edge_comparison(gray, sobel_mag, laplacian, canny)
    edge_fig.save(os.path.join(output_dir, "edge_operators_comparison.png"))

    # 3. Color Histograms
    hist_fig = plot_color_histograms(rgb_comp)
    hist_fig.save(os.path.join(output_dir, "color_histograms.png"))

    # 4. 2-Panel Calibrated HSV Chromaticity Analysis
    chroma_fig = plot_hsv_scatter(rgb_comp, color_masks=color_masks, max_samples=300)
    chroma_fig.save(os.path.join(output_dir, "hsv_chromaticity_analysis.png"))

    # 5. 1D Hue Density Distribution
    hue_1d_fig = plot_1d_hue_distribution(rgb_comp, color_masks=color_masks)
    hue_1d_fig.save(os.path.join(output_dir, "hue_1d_distribution.png"))

    # 6. Multi-Sample Detection Showcase Grid
    sample_files = sorted(glob.glob("sample_data/*.png"))
    num_samples = len(sample_files)
    cols = 3
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = axes.flatten()

    for idx, fpath in enumerate(sample_files):
        bgr = cv2.imread(fpath)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        c_mask, c_dict, _ = segment_colors(rgb)
        m_mask = clean_mask(c_mask, kernel_shape="ellipse", kernel_size=3, open_iterations=1, close_iterations=2)
        ann, detected, _ = detect_traffic_signs(rgb, m_mask, c_dict, min_area=600, apply_nms=True)

        labels = [f"{d['label']} [{d['heuristic_score']*100:.0f}%]" for d in detected]
        axes[idx].imshow(ann)
        fname = os.path.basename(fpath).replace(".png", "")
        axes[idx].set_title(f"{fname}\nDetected: {'; '.join(labels) if labels else 'None'}", fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_results_showcase.png"), dpi=150)
    plt.close(fig)

    print(f"All updated report visual assets generated in '{output_dir}/'")

if __name__ == "__main__":
    generate_visual_assets()
