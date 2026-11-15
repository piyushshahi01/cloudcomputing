import os
import cv2
import numpy as np

def create_road_background(width=640, height=480, sky_color=(135, 206, 235), ground_color=(60, 60, 60)):
    """
    Creates a photorealistic perspective road backdrop.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky with gradient
    for y in range(int(height * 0.45)):
        ratio = y / (height * 0.45)
        r = int(sky_color[0] * (1 - 0.2 * ratio))
        g = int(sky_color[1] * (1 - 0.15 * ratio))
        b = int(sky_color[2] * (1 - 0.1 * ratio))
        img[y, :] = (r, g, b)

    # Road and landscape
    img[int(height * 0.45):, :] = (45, 120, 45) # Grass green

    # Perspective Road
    horizon_y = int(height * 0.45)
    road_pts = np.array([
        [int(width * 0.42), horizon_y],
        [int(width * 0.58), horizon_y],
        [int(width * 0.90), height],
        [int(width * 0.10), height]
    ], dtype=np.int32)
    cv2.fillPoly(img, [road_pts], ground_color)

    # Lane markings (dashed center line)
    for i in range(5):
        y1 = int(horizon_y + (height - horizon_y) * (i / 5.0) * 0.8 + 15)
        y2 = int(horizon_y + (height - horizon_y) * ((i + 0.6) / 5.0) * 0.8 + 15)
        if y2 > height:
            break
        cv2.line(img, (int(width * 0.50), y1), (int(width * 0.50), y2), (255, 255, 255), 3)

    return img

def draw_sign_post(img, center_x, bottom_y, height=120):
    cv2.line(img, (center_x, bottom_y), (center_x, bottom_y + height), (120, 120, 120), 4)

def generate_samples(output_dir="sample_data"):
    os.makedirs(output_dir, exist_ok=True)
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    print("Generating curated traffic sign dataset and ground truth masks...")

    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + (np.pi / 8.0)

    # ==========================================
    # 1. STOP SIGN SCENE (Red Octagon)
    # ==========================================
    img1 = create_road_background()
    gt1 = np.zeros((480, 640), dtype=np.uint8)
    cx1, cy1, r1 = 480, 220, 45
    draw_sign_post(img1, cx1, cy1 + r1, height=130)

    oct_pts = np.array([[int(cx1 + r1 * np.cos(a)), int(cy1 + r1 * np.sin(a))] for a in angles], dtype=np.int32)
    cv2.fillPoly(img1, [oct_pts], (210, 20, 20)) # Red
    cv2.polylines(img1, [oct_pts], True, (255, 255, 255), 3) # White border
    cv2.putText(img1, "STOP", (cx1 - 32, cy1 + 8), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    cv2.fillPoly(gt1, [oct_pts], 255)

    cv2.imwrite(os.path.join(output_dir, "stop_sign_scene.png"), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(gt_dir, "stop_sign_scene_gt.png"), gt1)

    # ==========================================
    # 2. YIELD / PEDESTRIAN SCENE (Red Triangle)
    # ==========================================
    img2 = create_road_background()
    gt2 = np.zeros((480, 640), dtype=np.uint8)
    cx2, cy2, side = 470, 210, 85
    h2 = int(side * np.sqrt(3) / 2)
    draw_sign_post(img2, cx2, cy2 + int(h2 / 2), height=140)

    tri_pts = np.array([
        [cx2 - side // 2, cy2 - h2 // 2],
        [cx2 + side // 2, cy2 - h2 // 2],
        [cx2, cy2 + h2 // 2]
    ], dtype=np.int32)

    cv2.fillPoly(img2, [tri_pts], (220, 25, 25)) # Red border
    inner_side = int(side * 0.65)
    inner_h = int(inner_side * np.sqrt(3) / 2)
    inner_tri = np.array([
        [cx2 - inner_side // 2, cy2 - h2 // 2 + 10],
        [cx2 + inner_side // 2, cy2 - h2 // 2 + 10],
        [cx2, cy2 - h2 // 2 + 10 + inner_h]
    ], dtype=np.int32)
    cv2.fillPoly(img2, [inner_tri], (250, 250, 250)) # White center
    cv2.putText(img2, "YIELD", (cx2 - 24, cy2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 20, 20), 1)
    cv2.fillPoly(gt2, [tri_pts], 255)

    cv2.imwrite(os.path.join(output_dir, "yield_sign_scene.png"), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(gt_dir, "yield_sign_scene_gt.png"), gt2)

    # ==========================================
    # 3. SPEED LIMIT 40 SCENE (Circular Red Border)
    # ==========================================
    img3 = create_road_background()
    gt3 = np.zeros((480, 640), dtype=np.uint8)
    cx3, cy3, r3 = 490, 200, 44
    draw_sign_post(img3, cx3, cy3 + r3, height=140)

    cv2.circle(img3, (cx3, cy3), r3, (215, 20, 20), -1) # Red outer circle
    cv2.circle(img3, (cx3, cy3), int(r3 * 0.74), (255, 255, 255), -1) # White inner circle
    cv2.putText(img3, "40", (cx3 - 24, cy3 + 12), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2)
    cv2.circle(gt3, (cx3, cy3), r3, 255, -1)

    cv2.imwrite(os.path.join(output_dir, "speed_limit_scene.png"), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(gt_dir, "speed_limit_scene_gt.png"), gt3)

    # ==========================================
    # 4. BLUE MANDATORY / DIRECTION SIGN SCENE
    # ==========================================
    img4 = create_road_background()
    gt4 = np.zeros((480, 640), dtype=np.uint8)
    cx4, cy4, r4 = 150, 210, 42
    draw_sign_post(img4, cx4, cy4 + r4, height=130)

    cv2.circle(img4, (cx4, cy4), r4, (20, 90, 220), -1) # Blue circle
    cv2.circle(img4, (cx4, cy4), r4, (255, 255, 255), 2)
    # White forward arrow
    arrow_pts = np.array([
        [cx4, cy4 - 24],
        [cx4 + 16, cy4 - 8],
        [cx4 + 6, cy4 - 8],
        [cx4 + 6, cy4 + 20],
        [cx4 - 6, cy4 + 20],
        [cx4 - 6, cy4 - 8],
        [cx4 - 16, cy4 - 8]
    ], dtype=np.int32)
    cv2.fillPoly(img4, [arrow_pts], (255, 255, 255))
    cv2.circle(gt4, (cx4, cy4), r4, 255, -1)

    cv2.imwrite(os.path.join(output_dir, "mandatory_blue_scene.png"), cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(gt_dir, "mandatory_blue_scene_gt.png"), gt4)

    # ==========================================
    # 5. PEDESTRIAN CROSSING SIGN (Red Triangular Border)
    # ==========================================
    img5 = create_road_background()
    gt5 = np.zeros((480, 640), dtype=np.uint8)
    cx5, cy5, side5 = 480, 200, 86
    h5 = int(side5 * np.sqrt(3) / 2)
    draw_sign_post(img5, cx5, cy5 + int(h5 / 2), height=140)

    # Upright equilateral triangle
    ped_pts = np.array([
        [cx5, cy5 - int(h5 * 0.6)],
        [cx5 - side5 // 2, cy5 + int(h5 * 0.4)],
        [cx5 + side5 // 2, cy5 + int(h5 * 0.4)]
    ], dtype=np.int32)
    cv2.fillPoly(img5, [ped_pts], (220, 25, 25)) # Red border
    
    in_side5 = int(side5 * 0.68)
    in_h5 = int(in_side5 * np.sqrt(3) / 2)
    in_ped = np.array([
        [cx5, cy5 - int(h5 * 0.6) + 12],
        [cx5 - in_side5 // 2, cy5 - int(h5 * 0.6) + 12 + in_h5],
        [cx5 + in_side5 // 2, cy5 - int(h5 * 0.6) + 12 + in_h5]
    ], dtype=np.int32)
    cv2.fillPoly(img5, [in_ped], (255, 255, 255)) # White center

    # Pedestrian silhouette
    cv2.circle(img5, (cx5, cy5 - 6), 5, (20, 20, 20), -1)
    cv2.line(img5, (cx5, cy5 - 1), (cx5, cy5 + 14), (20, 20, 20), 3)
    cv2.line(img5, (cx5 - 8, cy5 + 6), (cx5 + 8, cy5 + 3), (20, 20, 20), 2)
    cv2.line(img5, (cx5, cy5 + 14), (cx5 - 7, cy5 + 26), (20, 20, 20), 2)
    cv2.line(img5, (cx5, cy5 + 14), (cx5 + 7, cy5 + 26), (20, 20, 20), 2)
    cv2.fillPoly(gt5, [ped_pts], 255)

    cv2.imwrite(os.path.join(output_dir, "warning_pedestrian_scene.png"), cv2.cvtColor(img5, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(gt_dir, "warning_pedestrian_scene_gt.png"), gt5)

    # ==========================================
    # 6. COMPREHENSIVE ROAD SCENE (ALL 4 PROMINENT TARGET SIGNS + CLUTTER)
    # 1. Speed Limit 40 (Red Circle)
    # 2. Pedestrian Crossing (Red Triangle)
    # 3. STOP (Red Octagon)
    # 4. Direction Board (Blue Rectangle)
    # Plus: Background tree leaves, road patches to test false positive suppression!
    # ==========================================
    img6 = create_road_background(width=720, height=480)
    gt6 = np.zeros((480, 720), dtype=np.uint8)

    # Sign 1: STOP Sign (Top Right, x=580, y=140, r=36)
    cx_s, cy_s, r_s = 580, 140, 36
    draw_sign_post(img6, cx_s, cy_s + r_s, height=130)
    oct_pts_c = np.array([[int(cx_s + r_s * np.cos(a)), int(cy_s + r_s * np.sin(a))] for a in angles], dtype=np.int32)
    cv2.fillPoly(img6, [oct_pts_c], (210, 20, 20))
    cv2.polylines(img6, [oct_pts_c], True, (255, 255, 255), 2)
    cv2.putText(img6, "STOP", (cx_s - 23, cy_s + 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
    cv2.fillPoly(gt6, [oct_pts_c], 255)

    # Sign 2: Speed Limit 40 (Mid Right, x=460, y=180, r=36)
    cx_sp, cy_sp, r_sp = 460, 180, 36
    draw_sign_post(img6, cx_sp, cy_sp + r_sp, height=130)
    cv2.circle(img6, (cx_sp, cy_sp), r_sp, (215, 20, 20), -1)
    cv2.circle(img6, (cx_sp, cy_sp), int(r_sp * 0.74), (255, 255, 255), -1)
    cv2.putText(img6, "40", (cx_sp - 19, cy_sp + 10), cv2.FONT_HERSHEY_DUPLEX, 0.82, (20, 20, 20), 2)
    cv2.circle(gt6, (cx_sp, cy_sp), r_sp, 255, -1)

    # Sign 3: Pedestrian Crossing Triangle (Left, x=220, y=180, side=76)
    cx_p, cy_p, side_p = 220, 180, 76
    h_p = int(side_p * np.sqrt(3) / 2)
    draw_sign_post(img6, cx_p, cy_p + int(h_p * 0.4), height=130)
    tri_ped = np.array([
        [cx_p, cy_p - int(h_p * 0.6)],
        [cx_p - side_p // 2, cy_p + int(h_p * 0.4)],
        [cx_p + side_p // 2, cy_p + int(h_p * 0.4)]
    ], dtype=np.int32)
    cv2.fillPoly(img6, [tri_ped], (220, 25, 25))
    in_side_p = int(side_p * 0.68)
    in_h_p = int(in_side_p * np.sqrt(3) / 2)
    in_tri_p = np.array([
        [cx_p, cy_p - int(h_p * 0.6) + 10],
        [cx_p - in_side_p // 2, cy_p - int(h_p * 0.6) + 10 + in_h_p],
        [cx_p + in_side_p // 2, cy_p - int(h_p * 0.6) + 10 + in_h_p]
    ], dtype=np.int32)
    cv2.fillPoly(img6, [in_tri_p], (255, 255, 255))
    cv2.circle(img6, (cx_p, cy_p - 5), 4, (20, 20, 20), -1)
    cv2.line(img6, (cx_p, cy_p - 1), (cx_p, cy_p + 12), (20, 20, 20), 2)
    cv2.line(img6, (cx_p - 7, cy_p + 5), (cx_p + 7, cy_p + 3), (20, 20, 20), 2)
    cv2.line(img6, (cx_p, cy_p + 12), (cx_p - 6, cy_p + 22), (20, 20, 20), 2)
    cv2.line(img6, (cx_p, cy_p + 12), (cx_p + 6, cy_p + 22), (20, 20, 20), 2)
    cv2.fillPoly(gt6, [tri_ped], 255)

    # Sign 4: Direction Board (Far Left, x=90, y=160, w=90, h=55)
    bx, by, bw, bh = 45, 130, 90, 58
    draw_sign_post(img6, bx + bw // 2, by + bh, height=130)
    cv2.rectangle(img6, (bx, by), (bx + bw, by + bh), (20, 90, 220), -1) # Blue
    cv2.rectangle(img6, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
    # Direction Arrow
    arr_c_x, arr_c_y = bx + 25, by + bh // 2
    arr_pts_dir = np.array([
        [arr_c_x, arr_c_y - 14],
        [arr_c_x + 10, arr_c_y - 4],
        [arr_c_x + 4, arr_c_y - 4],
        [arr_c_x + 4, arr_c_y + 14],
        [arr_c_x - 4, arr_c_y + 14],
        [arr_c_x - 4, arr_c_y - 4],
        [arr_c_x - 10, arr_c_y - 4]
    ], dtype=np.int32)
    cv2.fillPoly(img6, [arr_pts_dir], (255, 255, 255))
    cv2.putText(img6, "CITY", (bx + 42, by + 34), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1)
    gt6[by:by+bh, bx:bx+bw] = 255

    # ADD REALISTIC BACKGROUND CLUTTER (Small leaves, red/yellow flowers, pebbles, road lines)
    # These would cause 15 false positives in a naive algorithm!
    np.random.seed(42)
    for _ in range(25):
        # Small red/yellow flowers/leaves along the roadside
        fx = np.random.randint(10, 710)
        fy = np.random.randint(230, 470)
        fcolor = (210, 30, 30) if np.random.rand() > 0.5 else (240, 200, 20)
        cv2.circle(img6, (fx, fy), np.random.randint(3, 7), fcolor, -1)
    
    cv2.imwrite(os.path.join(output_dir, "comprehensive_test_scene.png"), cv2.cvtColor(img6, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(gt_dir, "comprehensive_test_scene_gt.png"), gt6)

    # Also keep multi_sign_scene
    cv2.imwrite(os.path.join(output_dir, "multi_sign_scene.png"), cv2.cvtColor(img6, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(gt_dir, "multi_sign_scene_gt.png"), gt6)

    print("Successfully generated all sample test scenes including comprehensive scene with exact 4 targets!")

if __name__ == "__main__":
    generate_samples()
