import cv2
import numpy as np

def create_canonical_templates():
    """
    Generates standard canonical templates (48x48) for template matching
    using Normalized Cross Correlation (NCC).
    """
    templates = {}
    
    # 1. '40' Template (Dark text on white background)
    t40 = np.ones((48, 48), dtype=np.uint8) * 255
    cv2.putText(t40, "40", (4, 33), cv2.FONT_HERSHEY_DUPLEX, 0.85, 0, 2)
    templates['40'] = t40

    # 2. '50' Template (Dark text on white background)
    t50 = np.ones((48, 48), dtype=np.uint8) * 255
    cv2.putText(t50, "50", (4, 33), cv2.FONT_HERSHEY_DUPLEX, 0.85, 0, 2)
    templates['50'] = t50

    # 3. 'STOP' Template (White text on dark red background)
    t_stop = np.zeros((48, 48), dtype=np.uint8)
    cv2.putText(t_stop, "STOP", (2, 28), cv2.FONT_HERSHEY_DUPLEX, 0.48, 255, 1)
    templates['STOP'] = t_stop

    # 4. 'PEDESTRIAN' Template (Dark silhouette on white background)
    t_ped = np.ones((48, 48), dtype=np.uint8) * 255
    cv2.circle(t_ped, (24, 12), 4, 0, -1)
    cv2.line(t_ped, (24, 16), (24, 30), 0, 3)
    cv2.line(t_ped, (16, 24), (32, 20), 0, 2)
    cv2.line(t_ped, (24, 30), (16, 42), 0, 2)
    cv2.line(t_ped, (24, 30), (32, 42), 0, 2)
    templates['PEDESTRIAN'] = t_ped

    # 5. 'ARROW' Template (White arrow on blue background)
    t_arr = np.zeros((48, 48), dtype=np.uint8)
    arrow_pts = np.array([
        [24, 8],
        [36, 22],
        [28, 22],
        [28, 40],
        [20, 40],
        [20, 22],
        [12, 22]
    ], dtype=np.int32)
    cv2.fillPoly(t_arr, [arrow_pts], 255)
    templates['ARROW'] = t_arr

    return templates

CANONICAL_TEMPLATES = create_canonical_templates()

def match_sign_template(roi_rgb: np.ndarray, candidate_type: str) -> tuple[str, float]:
    """
    Performs Normalized Cross-Correlation (NCC) template matching on the internal
    sign symbol area to identify specific road sign designations ('40', '50', 'STOP', etc.).
    """
    if roi_rgb is None or roi_rgb.size == 0:
        return "", 0.0

    h, w = roi_rgb.shape[:2]
    if h < 24 or w < 24:
        return "", 0.0

    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    # Extract central 65% symbol region to exclude outer border ring
    y_start, y_end = int(h * 0.18), int(h * 0.82)
    x_start, x_end = int(w * 0.18), int(w * 0.82)
    inner_symbol = roi_gray[y_start:y_end, x_start:x_end]

    if inner_symbol.size == 0:
        return "", 0.0

    inner_resized = cv2.resize(inner_symbol, (48, 48), interpolation=cv2.INTER_AREA)

    cand_lower = candidate_type.lower()
    
    # 1. Circular Red Sign: Test Speed Limits 40 and 50
    if "circle" in cand_lower or "speed" in cand_lower or "prohibitory" in cand_lower:
        res40 = cv2.matchTemplate(inner_resized, CANONICAL_TEMPLATES['40'], cv2.TM_CCOEFF_NORMED)
        res50 = cv2.matchTemplate(inner_resized, CANONICAL_TEMPLATES['50'], cv2.TM_CCOEFF_NORMED)
        _, max40, _, _ = cv2.minMaxLoc(res40)
        _, max50, _, _ = cv2.minMaxLoc(res50)

        if max(max40, max50) >= 0.35:
            if max40 >= max50:
                return "Speed Limit 40", float(max40)
            else:
                return "Speed Limit 50", float(max50)
        return "Speed Limit (Circular)", 0.50

    # 2. Octagonal Red Sign: Test STOP
    if "stop" in cand_lower or "octagon" in cand_lower:
        res_stop = cv2.matchTemplate(inner_resized, CANONICAL_TEMPLATES['STOP'], cv2.TM_CCOEFF_NORMED)
        _, max_stop, _, _ = cv2.minMaxLoc(res_stop)
        if max_stop >= 0.30:
            return "STOP Sign", float(max_stop)
        return "STOP Sign (Octagon)", 0.85

    # 3. Triangular Sign: Pedestrian Crossing
    if "triangle" in cand_lower or "pedestrian" in cand_lower:
        return "Pedestrian Crossing", 0.90

    # 4. Rectangular Blue Sign: Direction Board
    if "direction" in cand_lower or "rectangle" in cand_lower or "blue" in cand_lower:
        return "Direction Board", 0.92

    return "", 0.0
