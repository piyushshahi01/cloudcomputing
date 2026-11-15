import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image

def create_pipeline_comparison_grid(
    original_rgb: np.ndarray,
    hsv_img: np.ndarray,
    color_mask: np.ndarray,
    clean_mask: np.ndarray,
    canny_edges: np.ndarray,
    final_annotated: np.ndarray
) -> Image.Image:
    """
    Creates a 2x3 comprehensive pipeline progression grid for visual analysis.
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='#0B0F19')
    
    # 1. Input Image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title("1. Input Road Scene", fontsize=11, fontweight='bold', color='#60A5FA')
    axes[0, 0].axis('off')

    # 2. HSV Color Space Representation
    axes[0, 1].imshow(hsv_img)
    axes[0, 1].set_title("2. HSV Chromatic Representation", fontsize=11, fontweight='bold', color='#60A5FA')
    axes[0, 1].axis('off')

    # 3. Raw HSV Color Mask
    axes[0, 2].imshow(color_mask, cmap='gray')
    axes[0, 2].set_title("3. Multi-Range Color Slicing", fontsize=11, fontweight='bold', color='#60A5FA')
    axes[0, 2].axis('off')

    # 4. Morphologically Cleaned Mask
    axes[1, 0].imshow(clean_mask, cmap='magma')
    axes[1, 0].set_title("4. Morphological Open + Close + Fill", fontsize=11, fontweight='bold', color='#34D399')
    axes[1, 0].axis('off')

    # 5. Canny Edge Detection
    axes[1, 1].imshow(canny_edges, cmap='gray')
    axes[1, 1].set_title("5. Canny Boundary Ridges", fontsize=11, fontweight='bold', color='#34D399')
    axes[1, 1].axis('off')

    # 6. Final Detected Signs Overlay
    axes[1, 2].imshow(final_annotated)
    axes[1, 2].set_title("6. Final NMS-Filtered Detections", fontsize=11, fontweight='bold', color='#34D399')
    axes[1, 2].axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=140, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_color_histograms(image_rgb: np.ndarray) -> Image.Image:
    """
    Plots RGB and HSV channel distribution histograms with modern styling.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), facecolor='#0B0F19')

    # RGB Histogram
    colors = ('#EF4444', '#10B981', '#3B82F6')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        label = ['Red Channel', 'Green Channel', 'Blue Channel'][i]
        ax1.plot(hist, color=col, linewidth=1.5, label=label)
    ax1.set_title("RGB Pixel Intensity Distribution", fontsize=11, fontweight='bold', color='#E2E8F0')
    ax1.set_xlim([0, 256])
    ax1.set_xlabel("Intensity (0-255)", color='#94A3B8')
    ax1.set_ylabel("Pixel Frequency", color='#94A3B8')
    ax1.grid(True, alpha=0.15, linestyle='--')
    ax1.legend(facecolor='#1E293B', edgecolor='none')
    ax1.set_facecolor('#131B2E')

    # HSV Histogram
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    ax2.plot(h_hist, color='#F59E0B', linewidth=1.5, label='Hue (0-180)')
    ax2.plot(s_hist, color='#A855F7', linewidth=1.5, label='Saturation (0-255)')
    ax2.plot(v_hist, color='#38BDF8', linewidth=1.5, label='Value (0-255)')
    ax2.set_title("HSV Channel Distribution", fontsize=11, fontweight='bold', color='#E2E8F0')
    ax2.set_xlim([0, 256])
    ax2.set_xlabel("Bin Value", color='#94A3B8')
    ax2.grid(True, alpha=0.15, linestyle='--')
    ax2.legend(facecolor='#1E293B', edgecolor='none')
    ax2.set_facecolor('#131B2E')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_edge_comparison(
    gray_img: np.ndarray,
    sobel_mag: np.ndarray,
    laplacian: np.ndarray,
    canny: np.ndarray
) -> Image.Image:
    """
    Comparative visualizer for Sobel, Laplacian, and Canny edge operators.
    """
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), facecolor='#0B0F19')
    
    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title("Grayscale Input", fontsize=10, fontweight='bold', color='#E2E8F0')
    axes[0].axis('off')

    axes[1].imshow(sobel_mag, cmap='inferno')
    axes[1].set_title("Sobel Magnitude (1st Deriv.)", fontsize=10, fontweight='bold', color='#F59E0B')
    axes[1].axis('off')

    axes[2].imshow(laplacian, cmap='magma')
    axes[2].set_title("Laplacian (2nd Deriv. Zero-X)", fontsize=10, fontweight='bold', color='#EC4899')
    axes[2].axis('off')

    axes[3].imshow(canny, cmap='gray')
    axes[3].set_title("Canny (NMS + Hysteresis)", fontsize=10, fontweight='bold', color='#10B981')
    axes[3].axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_hsv_scatter(
    image_rgb: np.ndarray,
    color_masks: dict = None,
    red_h1=(0, 10),
    red_h2=(165, 180),
    blue_h=(102, 130),
    yellow_h=(15, 35),
    s_min=100,
    max_samples=250
) -> Image.Image:
    """
    Generates a 2-Panel Academic Chromaticity Distribution Analysis:
    
    Panel 1: Global Scene Chromaticity vs. Theoretical Threshold Boundaries
             - Displays background scene pixels in muted slate.
             - Overlays calibrated decision boundary rectangles for Red 1, Red 2, Blue, and Yellow.
             
    Panel 2: Segmented In-Mask Pixel Clusters (Stratified Sampling)
             - Directly extracts true in-mask pixels for Red Band 1, Red Band 2, Blue, and Yellow.
             - Zero contamination from bounding-box background pixels.
             - Displays exact threshold intervals and empirical cluster statistics in legend.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), facecolor='#0B0F19')
    
    # -------------------------------------------------------------
    # PANEL 1: Global Scene Chromaticity vs. Decision Boundaries
    # -------------------------------------------------------------
    ax1.set_facecolor('#131B2E')
    
    # Subsample background scene pixels (step=7 for clean scatter density)
    bg_h = h_channel[::7, ::7].ravel()
    bg_s = s_channel[::7, ::7].ravel()
    
    ax1.scatter(bg_h, bg_s, c='#475569', alpha=0.30, s=8, label='Full Scene Ambient Pixels', rasterized=True)

    # Overlay Decision Boundary Bounding Boxes
    # 1. Red Band 1: [red_h1[0], s_min] to [red_h1[1], 255]
    rect_r1 = patches.Rectangle((red_h1[0], s_min), red_h1[1] - red_h1[0], 255 - s_min,
                                linewidth=1.5, edgecolor='#EF4444', facecolor='none', linestyle='--',
                                label=f'Red Band 1: H∈[{red_h1[0]},{red_h1[1]}], S≥{s_min}')
    ax1.add_patch(rect_r1)

    # 2. Red Band 2: [red_h2[0], s_min] to [red_h2[1], 255]
    rect_r2 = patches.Rectangle((red_h2[0], s_min), red_h2[1] - red_h2[0], 255 - s_min,
                                linewidth=1.5, edgecolor='#DC2626', facecolor='none', linestyle='--',
                                label=f'Red Band 2: H∈[{red_h2[0]},{red_h2[1]}], S≥{s_min}')
    ax1.add_patch(rect_r2)

    # 3. Yellow Band: [yellow_h[0], s_min] to [yellow_h[1], 255]
    rect_y = patches.Rectangle((yellow_h[0], s_min), yellow_h[1] - yellow_h[0], 255 - s_min,
                               linewidth=1.5, edgecolor='#F59E0B', facecolor='none', linestyle='--',
                               label=f'Yellow Band: H∈[{yellow_h[0]},{yellow_h[1]}], S≥{s_min}')
    ax1.add_patch(rect_y)

    # 4. Blue Band: [blue_h[0], s_min] to [blue_h[1], 255]
    rect_b = patches.Rectangle((blue_h[0], s_min), blue_h[1] - blue_h[0], 255 - s_min,
                               linewidth=1.5, edgecolor='#3B82F6', facecolor='none', linestyle='--',
                               label=f'Blue Band: H∈[{blue_h[0]},{blue_h[1]}], S≥{s_min}')
    ax1.add_patch(rect_b)

    ax1.set_title("Panel A: Global Scene Pixels & Calibrated Threshold Bands", fontsize=11, fontweight='bold', color='#E2E8F0')
    ax1.set_xlabel("Hue Angle H (0 - 180 in OpenCV)", color='#94A3B8')
    ax1.set_ylabel("Saturation Purity S (0 - 255)", color='#94A3B8')
    ax1.set_xlim([0, 180])
    ax1.set_ylim([0, 260])
    ax1.grid(True, alpha=0.15, linestyle='--')
    ax1.legend(facecolor='#1E293B', edgecolor='none', loc='upper right', fontsize=8)

    # -------------------------------------------------------------
    # PANEL 2: Segmented In-Mask Clusters (Clean Ground Truth)
    # -------------------------------------------------------------
    ax2.set_facecolor('#131B2E')
    
    # Background in Panel B for reference (fainter)
    ax2.scatter(bg_h, bg_s, c='#334155', alpha=0.15, s=6, rasterized=True)

    np.random.seed(42)

    if color_masks is not None:
        # 1. Red Mask Extraction (Separating Band 1 near 0 and Band 2 near 180)
        mask_red = color_masks.get('red', None)
        if mask_red is not None and np.count_nonzero(mask_red) > 0:
            red_coords = np.argwhere(mask_red > 0)
            red_h = h_channel[red_coords[:, 0], red_coords[:, 1]]
            red_s = s_channel[red_coords[:, 0], red_coords[:, 1]]

            # Separate into Band 1 (0 to 25) and Band 2 (150 to 180)
            idx_r1 = np.where((red_h >= red_h1[0]) & (red_h <= red_h1[1]))[0]
            idx_r2 = np.where((red_h >= red_h2[0]) & (red_h <= red_h2[1]))[0]

            # Sample Band 1
            if len(idx_r1) > 0:
                sample_r1 = np.random.choice(idx_r1, size=min(max_samples, len(idx_r1)), replace=False)
                ax2.scatter(red_h[sample_r1], red_s[sample_r1], c='#EF4444', s=24, alpha=0.90, edgecolors='#991B1B',
                            label=f'Red Band 1: H∈[{red_h1[0]},{red_h1[1]}] (n={len(idx_r1)}, μ_H={np.mean(red_h[idx_r1]):.1f})')

            # Sample Band 2
            if len(idx_r2) > 0:
                sample_r2 = np.random.choice(idx_r2, size=min(max_samples, len(idx_r2)), replace=False)
                ax2.scatter(red_h[sample_r2], red_s[sample_r2], c='#DC2626', s=24, alpha=0.90, edgecolors='#7F1D1D',
                            label=f'Red Band 2: H∈[{red_h2[0]},{red_h2[1]}] (n={len(idx_r2)}, μ_H={np.mean(red_h[idx_r2]):.1f})')

        # 2. Blue Mask Extraction
        mask_blue = color_masks.get('blue', None)
        if mask_blue is not None and np.count_nonzero(mask_blue) > 0:
            blue_coords = np.argwhere(mask_blue > 0)
            blue_h = h_channel[blue_coords[:, 0], blue_coords[:, 1]]
            blue_s = s_channel[blue_coords[:, 0], blue_coords[:, 1]]

            sample_b = np.random.choice(len(blue_h), size=min(max_samples, len(blue_h)), replace=False)
            ax2.scatter(blue_h[sample_b], blue_s[sample_b], c='#3B82F6', s=24, alpha=0.90, edgecolors='#1E40AF',
                        label=f'Blue Mandatory: H∈[{blue_h.min()},{blue_h.max()}] (n={len(blue_h)}, μ_H={np.mean(blue_h):.1f})')

        # 3. Yellow Mask Extraction
        mask_yellow = color_masks.get('yellow', None)
        if mask_yellow is not None and np.count_nonzero(mask_yellow) > 0:
            yellow_coords = np.argwhere(mask_yellow > 0)
            yellow_h = h_channel[yellow_coords[:, 0], yellow_coords[:, 1]]
            yellow_s = s_channel[yellow_coords[:, 0], yellow_coords[:, 1]]

            sample_y = np.random.choice(len(yellow_h), size=min(max_samples, len(yellow_h)), replace=False)
            ax2.scatter(yellow_h[sample_y], yellow_s[sample_y], c='#F59E0B', s=24, alpha=0.90, edgecolors='#B45309',
                        label=f'Yellow Warning: H∈[{yellow_h.min()},{yellow_h.max()}] (n={len(yellow_h)}, μ_H={np.mean(yellow_h):.1f})')

    ax2.set_title("Panel B: Segmented In-Mask Pixels (True Sign Clusters)", fontsize=11, fontweight='bold', color='#E2E8F0')
    ax2.set_xlabel("Hue Angle H (0 - 180 in OpenCV)", color='#94A3B8')
    ax2.set_ylabel("Saturation Purity S (0 - 255)", color='#94A3B8')
    ax2.set_xlim([0, 180])
    ax2.set_ylim([0, 260])
    ax2.grid(True, alpha=0.15, linestyle='--')
    ax2.legend(facecolor='#1E293B', edgecolor='none', loc='upper right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_1d_hue_distribution(image_rgb: np.ndarray, color_masks: dict = None) -> Image.Image:
    """
    Plots 1D empirical Hue density curves for background vs segmented sign pixels,
    demonstrating multi-modal separation and red boundary wrap-around.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor='#0B0F19')
    ax.set_facecolor('#131B2E')

    # Full Image Hue Histogram
    hist_all = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_all_norm = hist_all / (hist_all.sum() + 1e-6)
    ax.plot(hist_all_norm, color='#64748B', linewidth=1.5, linestyle=':', label='Full Image Hue Density (Ambient Background)')

    if color_masks is not None:
        if 'red' in color_masks and np.count_nonzero(color_masks['red']) > 0:
            hist_red = cv2.calcHist([hsv], [0], color_masks['red'], [180], [0, 180]).flatten()
            hist_red_norm = hist_red / (hist_red.sum() + 1e-6)
            ax.plot(hist_red_norm, color='#EF4444', linewidth=2.0, label='Segmented Red (Peaks at H=0 & H=175)')

        if 'blue' in color_masks and np.count_nonzero(color_masks['blue']) > 0:
            hist_blue = cv2.calcHist([hsv], [0], color_masks['blue'], [180], [0, 180]).flatten()
            hist_blue_norm = hist_blue / (hist_blue.sum() + 1e-6)
            ax.plot(hist_blue_norm, color='#3B82F6', linewidth=2.0, label='Segmented Blue (Peak at H=110)')

        if 'yellow' in color_masks and np.count_nonzero(color_masks['yellow']) > 0:
            hist_yellow = cv2.calcHist([hsv], [0], color_masks['yellow'], [180], [0, 180]).flatten()
            hist_yellow_norm = hist_yellow / (hist_yellow.sum() + 1e-6)
            ax.plot(hist_yellow_norm, color='#F59E0B', linewidth=2.0, label='Segmented Yellow (Peak at H=25)')

    ax.set_title("1D Hue Density Profiles: Background vs. Segmented Sign Pixels", fontsize=10, fontweight='bold', color='#E2E8F0')
    ax.set_xlabel("Hue Angle H (0 - 180)", color='#94A3B8')
    ax.set_ylabel("Probability Density", color='#94A3B8')
    ax.set_xlim([0, 180])
    ax.grid(True, alpha=0.15, linestyle='--')
    ax.legend(facecolor='#1E293B', edgecolor='none', loc='upper right', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
