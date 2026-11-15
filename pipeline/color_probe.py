import cv2
import numpy as np

def probe_pixel_color(
    image_rgb: np.ndarray,
    x: int,
    y: int,
    red_h1=(0, 10),
    red_h2=(165, 180),
    blue_h=(102, 130),
    yellow_h=(15, 35),
    s_min=100,
    v_min=60
) -> dict:
    """
    Inspects exact chromatic and luminance coordinates of a single pixel (x, y),
    and evaluates threshold membership against calibrated decision boundaries.
    """
    h, w, _ = image_rgb.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    r, g, b = int(image_rgb[y, x, 0]), int(image_rgb[y, x, 1]), int(image_rgb[y, x, 2])
    hsv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = int(hsv_img[y, x, 0]), int(hsv_img[y, x, 1]), int(hsv_img[y, x, 2])

    # Decision membership evaluation
    classification = "Background / Unclassified Clutter"
    zone = "None"
    status_color = "#64748B"
    is_valid = False

    if sat >= s_min and val >= v_min:
        if red_h1[0] <= hue <= red_h1[1]:
            classification = "Red Traffic Sign (Band 1: Lower Hue)"
            zone = f"H∈[{red_h1[0]}, {red_h1[1]}], S≥{s_min}, V≥{v_min}"
            status_color = "#EF4444"
            is_valid = True
        elif red_h2[0] <= hue <= red_h2[1]:
            classification = "Red Traffic Sign (Band 2: Upper Wrap-around)"
            zone = f"H∈[{red_h2[0]}, {red_h2[1]}], S≥{s_min}, V≥{v_min}"
            status_color = "#DC2626"
            is_valid = True
        elif blue_h[0] <= hue <= blue_h[1]:
            classification = "Blue Traffic Sign (Mandatory / Direction)"
            zone = f"H∈[{blue_h[0]}, {blue_h[1]}], S≥{s_min}, V≥{v_min}"
            status_color = "#3B82F6"
            is_valid = True
        elif yellow_h[0] <= hue <= yellow_h[1]:
            classification = "Yellow Traffic Sign (Warning / Caution)"
            zone = f"H∈[{yellow_h[0]}, {yellow_h[1]}], S≥{s_min}, V≥{v_min}"
            status_color = "#F59E0B"
            is_valid = True
    else:
        if sat < s_min:
            zone = f"Desaturated Pixel (S={sat} < {s_min})"
        elif val < v_min:
            zone = f"Underexposed Pixel (V={val} < {v_min})"

    return {
        'coords': (x, y),
        'rgb': (r, g, b),
        'hsv': (hue, sat, val),
        'classification': classification,
        'decision_zone': zone,
        'status_color': status_color,
        'is_sign_candidate': is_valid
    }

