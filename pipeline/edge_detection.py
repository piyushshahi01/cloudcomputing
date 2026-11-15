import cv2
import numpy as np

def apply_gaussian_blur(gray_image: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Applies 2D Gaussian smoothing filter:
    G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    Suppresses high-frequency sensor and environmental noise before derivative operators.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)

def compute_sobel(image_gray: np.ndarray, ksize: int = 3):
    """
    Computes first-order spatial derivatives using the Sobel operator:
    Gx: Horizontal edge kernel [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    Gy: Vertical edge kernel   [-1, -2, -1; 0, 0, 0; 1, 2, 1]

    Magnitude: M = sqrt(Gx^2 + Gy^2)
    Direction: Theta = arctan(Gy / Gx)
    """
    if ksize not in [1, 3, 5, 7]:
        ksize = 3
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize to 0-255 uint8 for display
    norm_magnitude = np.uint8(np.clip(magnitude / (magnitude.max() + 1e-6) * 255, 0, 255))
    angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi) % 180
    return norm_magnitude, grad_x, grad_y, angle

def compute_laplacian(image_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Computes second-order spatial derivative using Laplacian operator:
    del^2 f = (d^2 f / dx^2) + (d^2 f / dy^2)
    Highlights regions of rapid intensity transitions (zero-crossings).
    """
    lap = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=ksize)
    lap_abs = np.uint8(np.clip(np.absolute(lap), 0, 255))
    return lap_abs

def detect_canny_edges(
    gray_image: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True
) -> np.ndarray:
    """
    Multi-stage Canny Edge Detection algorithm:
    1. Gaussian smoothing to suppress noise.
    2. Gradient calculation via Sobel.
    3. Non-Maximum Suppression (NMS) to thin edge ridges.
    4. Double thresholding (low_thresh, high_thresh).
    5. Edge tracking by Hysteresis to connect weak edges linked to strong edges.
    """
    edges = cv2.Canny(
        gray_image,
        threshold1=low_thresh,
        threshold2=high_thresh,
        apertureSize=aperture_size,
        L2gradient=l2_gradient
    )
    return edges

