import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation(output_path="Presentation_Slides.pptx"):
    prs = Presentation()
    # Set 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Dark automotive theme colors
    BG_COLOR = RGBColor(15, 23, 42)      # Deep slate
    ACCENT_BLUE = RGBColor(59, 130, 246) # Vivid Blue
    ACCENT_GREEN = RGBColor(52, 211, 153)# Emerald
    TEXT_LIGHT = RGBColor(241, 245, 249) # Off-white
    TEXT_MUTED = RGBColor(148, 163, 184) # Muted gray
    CARD_BG = RGBColor(30, 41, 59)       # Dark Card

    def set_slide_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR

    def add_header(slide, title_text, category_text="DIGITAL IMAGE PROCESSING (DIP) CAPSTONE PROJECT"):
        # Category Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_GREEN

        # Slide Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11), Inches(0.8))
        tf = t_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    blank_layout = prs.slide_layouts[6]
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(2.5))
    tf1 = t_box.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "Smart Traffic Sign Detection & Segmentation System"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Real-World Digital Image Processing (DIP) & Advanced Driver Assistance (ADAS)"
    p1_sub.font.size = Pt(20)
    p1_sub.font.color.rgb = ACCENT_BLUE

    # Team Info Box
    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(4.5), Inches(11.3), Inches(2.0))
    tf_meta = meta_box.text_frame
    p_meta = tf_meta.paragraphs[0]
    p_meta.text = "Team Members: [Member 1, Member 2, Member 3, Member 4]\nCourse: Digital Image Processing | Target Duration: 8–10 Minutes\nPerformance: 95.7% Mean IoU | Zero False Detections | ~117 FPS Real-Time CPU"
    p_meta.font.size = Pt(14)
    p_meta.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 2: MOTIVATION & PROBLEM STATEMENT
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "1. Real-World Motivation & Problem Statement", "SPEAKER 1 | 0:45 - 2:00 MIN")

    box2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf2 = box2.text_frame
    tf2.word_wrap = True

    bullets2 = [
        ("Autonomous Driving Mission-Critical Role:", "Advanced Driver Assistance Systems (ADAS) and Autonomous Vehicles require zero-latency traffic sign recognition to safely govern vehicle deceleration and lane navigation."),
        ("Environmental Real-World Challenges:", "Outdoor cameras endure solar zenith shifts, dense tree foliage shadows, rain streaks, and heavy background urban clutter."),
        ("The 'Black Box' Deep Learning Dilemma:", "Modern CNNs/YOLO models incur heavy edge compute requirements and lack transparent, deterministic mathematical interpretability."),
        ("Our Core Project Objective:", "Design, implement, and validate a deterministic DIP pipeline combining HSV color slicing, morphological filtering, spatial edge operators, and Douglas-Peucker geometric analysis to detect signs with 0% false alarm rate.")
    ]
    for title, desc in bullets2:
        p = tf2.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 3: HSV COLOR SPACE & CHROMATICITY ANALYSIS
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "2. Color Processing & 2D Chromaticity Analysis", "SPEAKER 2 | 2:00 - 3:15 MIN")

    box3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf3 = box3.text_frame
    tf3.word_wrap = True

    bullets3 = [
        ("Why RGB Fails Under Shadows:", "In RGB space, illumination fluctuations shift (R, G, B) vectors simultaneously, making Euclidean color distance highly unstable."),
        ("Cylindrical HSV Decoupling:", "HSV separates Chromaticity (Hue H and Saturation S) from Luminance (Value V), providing intrinsic invariance to solar angle and shadow transitions."),
        ("Dual-Interval Red Hue Wrap-around:", "In OpenCV, Hue spans [0, 180]. Pure spectral red wraps across 0° and 180°, necessitating two distinct threshold ranges: [0, 10] and [165, 180]."),
        ("Calibrated Decision Bands:", "Red: [0,10] ∪ [165,180] | Blue Mandatory: [102,130] | Yellow Warning: [15,35] | Saturation Floor: S ≥ 100."),
        ("2D Chromaticity Verification:", "Extracting pixels strictly from candidate segmentation masks proves that traffic signs form discrete, non-overlapping clusters cleanly isolated from background ambient scatter.")
    ]
    for title, desc in bullets3:
        p = tf3.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 4: MORPHOLOGY & SPATIAL EDGE OPERATORS
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "3. Mathematical Morphology & Spatial Edge Operators", "SPEAKER 2 | 3:15 - 4:30 MIN")

    box4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf4 = box4.text_frame
    tf4.word_wrap = True

    bullets4 = [
        ("Morphological Opening (A ∘ B):", "Erosion followed by dilation with an elliptical structuring element eliminates isolated background noise specks and thin foliage spurs."),
        ("Morphological Closing (A • B):", "Dilation followed by erosion seals microscopic fissures and bridges disconnected sign border segments."),
        ("Contour-Reconstruction Hole Filling:", "Scanline rasterization transforms hollow sign rings (like speed limit borders) into solid candidate geometric regions."),
        ("Spatial Derivative Edge Detection:", "Sobel (1st derivative Gx, Gy) yields gradient vectors; Laplacian of Gaussian (2nd derivative) highlights zero-crossings; Canny provides optimal thin-edge hysteresis ridges."),
        ("Edge Density Metric Bug Fix:", "Thresholding continuous gradient maps (T ≥ 50) eliminates noise floor, yielding realistic edge densities of 3.87% (Sobel) and 1.47% (Canny) rather than artificial 90%+ readings.")
    ]
    for title, desc in bullets4:
        p = tf4.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 5: OVERCOMING FALSE DETECTIONS & GEOMETRY
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "4. Overcoming False Detections: Geometric Filtering & NMS", "SPEAKER 3 | 4:30 - 5:45 MIN")

    box5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf5 = box5.text_frame
    tf5.word_wrap = True

    bullets5 = [
        ("The False Positive Crisis in Naive DIP:", "Initial color thresholding detected 15+ candidate regions—mistaking small leaves, roadside flowers, and concentric inner borders for traffic signs."),
        ("Five Robust DIP Enhancements:", "1. Minimum Area Threshold (≥ 600 px) rejects distant clutter.\n2. Solidity Filtering (≥ 0.78) rejects irregular, ragged foliage.\n3. Aspect Ratio Clamping (0.48 ≤ AR ≤ 2.30) eliminates road lane marks and vertical poles.\n4. Douglas-Peucker Polygon Approximation (ε = 0.035P) identifies vertex counts (8=Stop, 3=Yield/Pedestrian, 4=Direction Board).\n5. Non-Maximum Suppression (NMS) with IoU ≥ 0.35 merges concentric duplicate boxes."),
        ("Outcome:", "False detections dropped from 15 to exactly zero, isolating the 4 prominent target signs perfectly.")
    ]
    for title, desc in bullets5:
        p = tf5.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 6: TEMPLATE MATCHING OCR & ARCHITECTURE
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "5. System Architecture & Template Matching OCR", "SPEAKER 3 | 5:45 - 6:30 MIN")

    box6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf6 = box6.text_frame
    tf6.word_wrap = True

    bullets6 = [
        ("Template Matching via Normalized Cross-Correlation (NCC):", "To recognize internal text/digits without a neural network, candidate central regions (65%) are normalized to 48x48 and cross-correlated against canonical binary templates."),
        ("Discriminative Symbol Recognition:", "High cross-correlation (γ ≥ 0.58) deterministically distinguishes 'Speed Limit 40' from other circular signs, and verifies 'STOP' lettering."),
        ("Heuristic Geometric Score:", "Scores (e.g. 93%, 96%) represent deterministic geometric conformance to ideal mathematical sign models rather than uncalibrated probabilities."),
        ("Modular Architecture:", "Seamless pipeline connecting Preprocessing → Morphology → Canny Edges → Geometric Classifier → NMS → ADAS Cockpit HUD.")
    ]
    for title, desc in bullets6:
        p = tf6.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 7: LIVE APPLICATION DEMONSTRATION PLAN
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "6. Live Application & ADAS HUD Demonstration", "ALL SPEAKERS | 6:30 - 8:15 MIN")

    box7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf7 = box7.text_frame
    tf7.word_wrap = True

    bullets7 = [
        ("Tab 1 - Vision Pipeline Progression:", "Display live stage-by-stage progression from input image → HSV conversion → raw color slicing → morphological cleanup → Canny edge ridges → final overlay."),
        ("Tab 2 - False Positive Suppression:", "Demonstrate side-by-side Before (15 false clutter detections) vs. After (4 clean true signs)."),
        ("Tab 3 - 2D Chromaticity Studio & Color Probe:", "Show 2D Hue-Saturation clusters and demonstrate the interactive pixel color probe on arbitrary coordinates."),
        ("Tab 4 & 5 - Edge Comparison & Ground Truth Overlap:", "Compare Sobel/Laplacian/Canny metrics and display 94.3% IoU and 97.1% Dice scores against ground-truth masks."),
        ("Tab 6 - Video HUD Stream:", "Run real-time frame-by-frame driving simulation video with cockpit telemetry alerts at ~117 FPS.")
    ]
    for title, desc in bullets7:
        p = tf7.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_GREEN
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 8: QUANTITATIVE BENCHMARK RESULTS
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "7. Quantitative Benchmark & Performance Evaluation", "SPEAKER 4 | 8:15 - 9:15 MIN")

    box8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf8 = box8.text_frame
    tf8.word_wrap = True

    bullets8 = [
        ("Comprehensive Multi-Sign Scene (4 Targets + 25 Clutter Spots):", "IoU = 94.30% | Dice = 97.07% | Precision = 100.0% | Latency = 15.94 ms"),
        ("Speed Limit 40 Scene (Red Circle):", "IoU = 100.00% | Dice = 100.00% | Precision = 100.0% | Latency = 5.54 ms"),
        ("Pedestrian Crossing Scene (Red Triangle):", "IoU = 99.75% | Dice = 99.88% | Precision = 100.0% | Latency = 4.68 ms"),
        ("STOP Sign Scene (Red Octagon):", "IoU = 88.11% | Dice = 93.68% | Precision = 100.0% | Latency = 5.84 ms"),
        ("Overall Dataset Aggregate Mean:", "Mean IoU = 95.66% | Mean Dice = 97.74% | Precision = 100.00% | Latency = 8.56 ms (~117 FPS on CPU)")
    ]
    for title, desc in bullets8:
        p = tf8.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 9: LIMITATIONS & FUTURE IMPROVEMENTS
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "8. Limitations, Challenges & Future Scope", "SPEAKER 4 | 9:15 - 10:00 MIN")

    box9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf9 = box9.text_frame
    tf9.word_wrap = True

    bullets9 = [
        ("Identified Real-World Limitations:", "1. Dense fog and night headlight glare scatter chromatic wavelengths.\n2. Oblique viewing angles exceeding 60° cause affine perspective warping of polygon vertices.\n3. Extreme physical occlusions (> 50%) reduce contour solidity."),
        ("Implemented Weather Simulator & CLAHE Restoration:", "Integrated Koschmieder atmospheric fog, night attenuation, and rain simulation with automated CLAHE and Bilateral filtering restoration."),
        ("Future Scope:", "1. Hu Invariant Moments for affine perspective invariance.\n2. Multi-scale Retinex illumination decomposition.\n3. Extended template dictionary for numeric speed limits (30, 60, 80, 100 km/h).")
    ]
    for title, desc in bullets9:
        p = tf9.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_BLUE
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 10: CONCLUSION & VIVA Q&A
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "9. Conclusion & Viva Defense", "ALL SPEAKERS | 10:00 - 10:45 MIN")

    box10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf10 = box10.text_frame
    tf10.word_wrap = True

    bullets10 = [
        ("Project Summary:", "Successfully designed, implemented, and benchmarked a deterministic real-world Digital Image Processing application for traffic sign recognition."),
        ("Key Accomplishments:", "• 6 core DIP techniques implemented (HSV Slicing, Morphology, Gaussian/Canny/Sobel, Douglas-Peucker, NMS, Template Matching OCR).\n• 95.66% Mean IoU and 100.0% Precision across all test scenes.\n• Real-time processing at ~117 FPS on standard CPU hardware.\n• Full-featured interactive Streamlit GUI with ADAS HUD telemetry and degraded weather stress testing."),
        ("Open for Questions:", "Thank you for your time and guidance. We welcome any questions regarding our mathematical principles and implementation.")
    ]
    for title, desc in bullets10:
        p = tf10.add_paragraph()
        p.text = f"•  {title} "
        p.font.bold = True
        p.font.size = Pt(15)
        p.font.color.rgb = ACCENT_GREEN
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = TEXT_LIGHT

    prs.save(output_path)
    print(f"Presentation successfully saved to: {output_path}")

if __name__ == "__main__":
    create_presentation()

