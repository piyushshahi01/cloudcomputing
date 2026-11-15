import cv2
import numpy as np

def get_kernel(shape_name: str, ksize: int) -> np.ndarray:
    """
    Generate structuring element of specified shape and size.
    Supported shapes: 'ellipse', 'rect', 'cross'.
    """
    shape_dict = {
        'ellipse': cv2.MORPH_ELLIPSE,
        'rect': cv2.MORPH_RECT,
        'cross': cv2.MORPH_CROSS
    }
    morph_shape = shape_dict.get(shape_name.lower(), cv2.MORPH_ELLIPSE)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.getStructuringElement(morph_shape, (ksize, ksize))

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fills internal holes within segmented binary regions using contour reconstruction.
    Fills text, internal symbols, and border gaps without background leakage.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask

def clean_mask(
    mask: np.ndarray,
    kernel_shape: str = "ellipse",
    kernel_size: int = 5,
    open_iterations: int = 2,
    close_iterations: int = 2,
    do_fill_holes: bool = True
) -> np.ndarray:
    """
    Applies mathematical morphological operations:
    1. Opening (Erosion followed by Dilation): Eliminates isolated noise specks and thin branches.
    2. Closing (Dilation followed by Erosion): Joins broken boundaries.
    3. Hole Filling: Fills the hollow interior of traffic sign borders and icons.

    Mathematical definition:
    - Opening:  A o B = (A (-) B) (+) B
    - Closing:  A . B = (A (+) B) (-) B
    """
    kernel = get_kernel(kernel_shape, kernel_size)
    
    # Step 1: Morphological Opening to remove spurious noise
    if open_iterations > 0:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    else:
        cleaned = mask.copy()

    # Step 2: Morphological Closing to connect disjoint segments
    if close_iterations > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    # Step 3: Contour-based hole filling
    if do_fill_holes:
        cleaned = fill_holes(cleaned)

    return cleaned
