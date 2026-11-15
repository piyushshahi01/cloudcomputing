import cv2
import numpy as np
from .template_matcher import match_sign_template

def compute_box_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) and containment ratio between two bounding boxes.
    Boxes are in format: (x, y, w, h)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union_area = areaA + areaB - inter_area
    iou = inter_area / float(union_area) if union_area > 0 else 0.0
    containment = inter_area / float(min(areaA, areaB)) if min(areaA, areaB) > 0 else 0.0

    return iou, containment

def apply_non_maximum_suppression(detected_signs, iou_threshold=0.35, containment_threshold=0.70):
    """
    Applies Non-Maximum Suppression (NMS) to eliminate duplicate overlapping
    bounding boxes and concentric inner/outer border detections.
    """
    if len(detected_signs) <= 1:
        return detected_signs

    # Sort descending by heuristic geometric score * area
    sorted_signs = sorted(
        detected_signs,
        key=lambda s: s['heuristic_score'] * (1.0 + 0.1 * np.log10(max(10, s['area_pixels']))),
        reverse=True
    )

    keep_signs = []
    suppressed = set()

    for i in range(len(sorted_signs)):
        if i in suppressed:
            continue
        keep_signs.append(sorted_signs[i])

        for j in range(i + 1, len(sorted_signs)):
            if j in suppressed:
                continue
            iou, containment = compute_box_iou(sorted_signs[i]['bbox'], sorted_signs[j]['bbox'])
            # Suppress if boxes heavily overlap or one is nested inside the other
            if iou >= iou_threshold or containment >= containment_threshold:
                suppressed.add(j)

    return keep_signs

def classify_shape_and_color(
    contour,
    approx_vertices: int,
    circularity: float,
    solidity: float,
    aspect_ratio: float,
    dominant_color: str,
    color_fill_ratio: float,
    radial_var: float
) -> tuple[str, float]:
    """
    Classifies traffic sign shape using strict geometric invariants and color distribution.
    Returns (sign_category, heuristic_geometric_score).
    """
    heuristic_score = 0.50

    # 1. TRIANGLE: 3 vertices or triangular isoperimetric quotient
    if approx_vertices == 3 or (approx_vertices == 4 and 0.40 <= circularity <= 0.70 and solidity >= 0.78):
        if dominant_color == 'red':
            return "Pedestrian / Yield Sign (Triangle)", 0.91
        elif dominant_color == 'yellow':
            return "Pedestrian / Warning Sign (Triangle)", 0.90
        else:
            return "Triangular Sign", 0.75

    # 2. RED SIGNS (STOP vs SPEED LIMIT / CIRCLE)
    if dominant_color == 'red':
        # Stop sign is a solid red octagon (high fill ratio >= 0.58, 8 vertices or circularity ~ 0.85-0.90)
        if color_fill_ratio >= 0.58 and (approx_vertices in [7, 8, 9] or 0.78 <= circularity <= 0.95):
            heuristic_score = 0.94 if approx_vertices == 8 else 0.88
            return "STOP Sign (Octagon)", heuristic_score
        # Speed limit sign has a red circular ring with white interior (fill ratio 0.20-0.55, radial_var < 0.08)
        elif color_fill_ratio < 0.58 and (circularity >= 0.76 or radial_var < 0.075 or approx_vertices in [7, 8, 9]):
            heuristic_score = 0.93 if radial_var < 0.05 else 0.87
            return "Speed Limit Sign (Circle)", heuristic_score
        elif approx_vertices in [7, 8, 9]:
            return "STOP Sign (Octagon)", 0.82
        elif circularity >= 0.75:
            return "Prohibitory Sign (Circle)", 0.80

    # 3. BLUE SIGNS (DIRECTION BOARD vs MANDATORY CIRCLE)
    if dominant_color == 'blue':
        # Direction board: rectangular shape with 4 vertices
        if approx_vertices == 4 or (0.50 <= aspect_ratio <= 2.20 and solidity >= 0.82):
            return "Direction Board (Rectangle)", 0.92
        elif circularity >= 0.76 or radial_var < 0.075:
            return "Mandatory Sign (Blue Circle)", 0.90
        else:
            return "Information Sign (Blue)", 0.75

    # 4. YELLOW SIGNS (WARNING DIAMOND)
    if dominant_color == 'yellow':
        if approx_vertices == 4 and 0.75 <= aspect_ratio <= 1.30 and solidity >= 0.80:
            return "Warning Sign (Diamond)", 0.90
        elif approx_vertices == 3:
            return "Warning Sign (Triangle)", 0.88

    return "Unknown Traffic Sign", 0.50

def detect_hough_circles(
    gray_image: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 50,
    param1: int = 100,
    param2: int = 35,
    min_radius: int = 15,
    max_radius: int = 150
):
    """
    Parametric Hough Circle Transform: (x - a)^2 + (y - b)^2 = r^2
    """
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    return []

def detect_traffic_signs(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color_masks: dict,
    min_area: int = 600,
    max_area: int = 120000,
    min_solidity: float = 0.78,
    poly_epsilon_factor: float = 0.035,
    use_template_matching: bool = True,
    apply_nms: bool = True
):
    """
    Enhanced Traffic Sign Detection Pipeline:
    1. Multi-level geometric filtering (Area, Aspect Ratio, Solidity, Circularity)
    2. Douglas-Peucker polygonal approximation
    3. Template matching symbol recognition (Speed Limit '40', 'STOP', 'Pedestrian', 'Arrow')
    4. Non-Maximum Suppression (NMS) to eliminate duplicate/nested bounding boxes.
    """
    annotated = original_rgb.copy()
    h, w, _ = original_rgb.shape
    img_area = h * w
    contour_mask = np.zeros((h, w), dtype=np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_candidates = []

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # 1. Area Filtering: Reject tiny leaves / background noise and giant sky/road blobs
        if area < min_area or area > (0.35 * img_area):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # 2. Solidity Filtering: Traffic signs are strictly convex polygons
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue

        # 3. Circularity (Isoperimetric Quotient)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.38:  # Rejects ragged, fragmented contours
            continue

        # 4. Aspect Ratio Filtering
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / bh if bh > 0 else 0.0
        if aspect_ratio < 0.48 or aspect_ratio > 2.30:
            continue

        # 5. Radial Distance Variance from Centroid (Circle vs Polygon test)
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            pts = cnt.reshape(-1, 2)
            dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            radial_var = float(np.std(dists) / np.mean(dists)) if len(dists) > 0 and np.mean(dists) > 0 else 1.0
        else:
            radial_var = 1.0

        # 6. Douglas-Peucker Polygonal Approximation
        epsilon = poly_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices = len(approx)

        # 7. Dominant Color & Color Fill Ratio Analysis
        roi_red = color_masks.get('red', np.zeros((h, w), dtype=np.uint8))[y:y+bh, x:x+bw]
        roi_blue = color_masks.get('blue', np.zeros((h, w), dtype=np.uint8))[y:y+bh, x:x+bw]
        roi_yellow = color_masks.get('yellow', np.zeros((h, w), dtype=np.uint8))[y:y+bh, x:x+bw]

        red_pixels = cv2.countNonZero(roi_red)
        blue_pixels = cv2.countNonZero(roi_blue)
        yellow_pixels = cv2.countNonZero(roi_yellow)

        color_counts = {'red': red_pixels, 'blue': blue_pixels, 'yellow': yellow_pixels}
        dominant_color = max(color_counts, key=color_counts.get)
        if color_counts[dominant_color] == 0:
            dominant_color = 'unknown'

        color_fill_ratio = float(color_counts[dominant_color]) / float(bw * bh) if (bw * bh) > 0 else 0.0

        # Base geometric classification
        category, base_score = classify_shape_and_color(
            cnt, num_vertices, circularity, solidity, aspect_ratio, dominant_color, color_fill_ratio, radial_var
        )

        # Crop ROI for template matching
        pad = 4
        y1, y2 = max(0, y - pad), min(h, y + bh + pad)
        x1, x2 = max(0, x - pad), min(w, x + bw + pad)
        roi_rgb = original_rgb[y1:y2, x1:x2].copy()

        # 8. Template Matching for symbol/digit verification (OCR)
        final_label = category
        heuristic_score = base_score
        
        if use_template_matching:
            matched_symbol, tmpl_score = match_sign_template(roi_rgb, category)
            if matched_symbol:
                final_label = matched_symbol
                heuristic_score = min(0.98, max(base_score, 0.70 + 0.28 * tmpl_score))

        raw_candidates.append({
            'id': idx + 1,
            'label': final_label,
            'heuristic_score': heuristic_score,
            'dominant_color': dominant_color,
            'shape_vertices': num_vertices,
            'circularity': round(circularity, 3),
            'solidity': round(solidity, 3),
            'aspect_ratio': round(aspect_ratio, 2),
            'area_pixels': int(area),
            'bbox': (x, y, bw, bh),
            'approx_poly': approx,
            'contour': cnt,
            'roi': roi_rgb
        })

    # 9. Non-Maximum Suppression (NMS) to eliminate duplicate/nested boxes
    if apply_nms:
        detected_signs = apply_non_maximum_suppression(raw_candidates, iou_threshold=0.35, containment_threshold=0.65)
    else:
        detected_signs = raw_candidates

    # Re-index signs and render overlays
    for idx, sign in enumerate(detected_signs):
        sign['id'] = idx + 1
        x, y, bw, bh = sign['bbox']
        dom_color = sign['dominant_color']

        # Draw contour on binary output mask
        cv2.drawContours(contour_mask, [sign['contour']], -1, 255, -1)

        # High-visibility bounding box and polygon
        box_color = (0, 255, 0) if dom_color == 'red' else (255, 165, 0) if dom_color == 'blue' else (255, 255, 0)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)
        cv2.drawContours(annotated, [sign['approx_poly']], -1, (255, 0, 255), 2)

        # Label Banner
        label_text = f"{sign['label']} [{sign['heuristic_score']*100:.0f}%]"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x, max(0, y - th - 6)), (x + tw + 6, max(th + 6, y)), box_color, -1)
        cv2.putText(
            annotated,
            label_text,
            (x + 3, max(th + 2, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return annotated, detected_signs, contour_mask
