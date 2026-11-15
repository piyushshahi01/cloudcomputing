# Digital Image Processing Pipeline for Smart Traffic Sign Detection
"""
Smart Traffic Sign Detection and Segmentation Pipeline
Modules:
- color_segmentation: RGB to HSV, multi-hue thresholding (Red, Blue, Yellow)
- morphology: Morphological operations (Opening, Closing, Dilation, Hole filling)
- edge_detection: Gaussian smoothing, Sobel gradients, Canny edge detection
- shape_detector: Contour geometry, Douglas-Peucker polygonal approx, Hough circles, NMS
- template_matcher: Normalized Cross-Correlation (NCC) template OCR
- degradation_simulator: Fog, Night, Rain, and CLAHE / Bilateral restoration
- color_probe: Interactive pixel inspection and decision boundary probe
- metrics: Quantitative evaluation (IoU, Dice, PSNR, SSIM, Latency)
- visualizer: Staged visual outputs and matplotlib plotting utilities
"""

from .color_segmentation import segment_colors, rgb_to_hsv, get_hsv_channels
from .morphology import clean_mask, fill_holes
from .edge_detection import apply_gaussian_blur, compute_sobel, detect_canny_edges, compute_laplacian
from .shape_detector import detect_traffic_signs, detect_hough_circles, apply_non_maximum_suppression
from .template_matcher import match_sign_template
from .degradation_simulator import simulate_fog, simulate_night, simulate_rain, simulate_sensor_noise, restore_degraded_image
from .color_probe import probe_pixel_color
from .metrics import compute_iou, compute_dice, compute_edge_contrast, benchmark_pipeline
from .visualizer import create_pipeline_comparison_grid, plot_color_histograms, plot_edge_comparison, plot_hsv_scatter, plot_1d_hue_distribution
