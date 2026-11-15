import cv2
import numpy as np

def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to HSV (Hue, Saturation, Value) color space.
    HSV decouples intensity (Value) from chromatic information (Hue & Saturation),
    providing invariance to illumination variations on roads.
    """
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

def get_hsv_channels(hsv_image: np.ndarray):
    """
    Split HSV image into individual H, S, V channels for visualization.
    """
    h, s, v = cv2.split(hsv_image)
    return h, s, v

def segment_colors(
    image_rgb: np.ndarray,
    detect_red: bool = True,
    detect_blue: bool = True,
    detect_yellow: bool = True,
    red_lower1=(0, 100, 70),
    red_upper1=(10, 255, 255),
    red_lower2=(165, 100, 70),
    red_upper2=(180, 255, 255),
    blue_lower=(102, 130, 60),
    blue_upper=(130, 255, 255),
    yellow_lower=(15, 110, 60),
    yellow_upper=(35, 255, 255)
):
    """
    Segments candidate traffic sign regions by color thresholding in HSV color space.
    Traffic signs predominantly use Red (regulatory/stop), Blue (mandatory/info),
    and Yellow/Amber (warning/caution).

    Because Red wraps around the 0-180 Hue boundary in OpenCV, two threshold ranges
    are computed and joined via bitwise-OR.

    Returns:
        combined_mask: np.ndarray (binary mask of all detected colors)
        color_masks: dict containing individual binary masks for 'red', 'blue', 'yellow'
        masked_rgb: np.ndarray (original image masked with combined_mask)
    """
    hsv = rgb_to_hsv(image_rgb)
    h, w, _ = image_rgb.shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    color_masks = {}

    # Red Segmentation (Dual Hue range for wrap-around)
    if detect_red:
        mask_red1 = cv2.inRange(hsv, np.array(red_lower1, dtype=np.uint8), np.array(red_upper1, dtype=np.uint8))
        mask_red2 = cv2.inRange(hsv, np.array(red_lower2, dtype=np.uint8), np.array(red_upper2, dtype=np.uint8))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        color_masks['red'] = mask_red
        combined_mask = cv2.bitwise_or(combined_mask, mask_red)
    else:
        color_masks['red'] = np.zeros((h, w), dtype=np.uint8)

    # Blue Segmentation (Higher saturation threshold avoids natural sky)
    if detect_blue:
        mask_blue = cv2.inRange(hsv, np.array(blue_lower, dtype=np.uint8), np.array(blue_upper, dtype=np.uint8))
        color_masks['blue'] = mask_blue
        combined_mask = cv2.bitwise_or(combined_mask, mask_blue)
    else:
        color_masks['blue'] = np.zeros((h, w), dtype=np.uint8)

    # Yellow Segmentation
    if detect_yellow:
        mask_yellow = cv2.inRange(hsv, np.array(yellow_lower, dtype=np.uint8), np.array(yellow_upper, dtype=np.uint8))
        color_masks['yellow'] = mask_yellow
        combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
    else:
        color_masks['yellow'] = np.zeros((h, w), dtype=np.uint8)

    # Extract masked RGB region
    masked_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=combined_mask)

    return combined_mask, color_masks, masked_rgb
