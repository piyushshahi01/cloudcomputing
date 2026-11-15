import cv2
import numpy as np

def simulate_fog(image_rgb: np.ndarray, fog_intensity: float = 0.5) -> np.ndarray:
    """
    Simulates atmospheric haze/fog using Koschmieder's atmospheric scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))
    where J(x) is true scene radiance, t(x) is transmission, A is ambient airlight.
    """
    fog_intensity = np.clip(fog_intensity, 0.0, 1.0)
    if fog_intensity == 0:
        return image_rgb.copy()

    h, w, _ = image_rgb.shape
    airlight = np.array([220, 225, 230], dtype=np.float32) # Daylight sky airlight
    
    # Transmission map decreases with depth (simulated vertical perspective)
    y_coords = np.linspace(0.4, 0.9, h)[:, None]
    transmission = np.clip(1.0 - fog_intensity * y_coords, 0.15, 1.0)
    transmission = np.repeat(transmission, w, axis=1)[:, :, None]

    foggy = image_rgb.astype(np.float32) * transmission + airlight * (1.0 - transmission)
    return np.clip(foggy, 0, 255).astype(np.uint8)

def simulate_night(image_rgb: np.ndarray, darkness: float = 0.6) -> np.ndarray:
    """
    Simulates low-light / night conditions by non-linear intensity attenuation.
    """
    darkness = np.clip(darkness, 0.0, 0.95)
    if darkness == 0:
        return image_rgb.copy()

    scale = 1.0 - darkness
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * scale # Attenuate Value channel
    hsv[:, :, 1] = hsv[:, :, 1] * (0.8 + 0.2 * scale) # Slight desaturation
    restored_rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return restored_rgb

def simulate_rain(image_rgb: np.ndarray, rain_amount: float = 0.5) -> np.ndarray:
    """
    Simulates directional rainfall streaks using linear motion kernel convolution.
    """
    if rain_amount <= 0:
        return image_rgb.copy()

    h, w, _ = image_rgb.shape
    num_streaks = int(500 * rain_amount)
    rain_layer = np.zeros((h, w), dtype=np.uint8)

    # Random streak points
    xs = np.random.randint(0, w, num_streaks)
    ys = np.random.randint(0, h, num_streaks)
    streak_len = int(12 + 18 * rain_amount)

    for x, y in zip(xs, ys):
        cv2.line(rain_layer, (x, y), (x + int(streak_len * 0.3), y + streak_len), 200, 1)

    # Blur streaks slightly
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    
    output = image_rgb.copy()
    for c in range(3):
        output[:, :, c] = np.clip(output[:, :, c] + rain_layer * 0.5, 0, 255)
    return output

def simulate_sensor_noise(image_rgb: np.ndarray, noise_sigma: float = 20.0) -> np.ndarray:
    """
    Simulates additive Gaussian sensor thermal noise: I' = I + N(0, sigma^2)
    """
    if noise_sigma <= 0:
        return image_rgb.copy()
    noise = np.random.normal(0, noise_sigma, image_rgb.shape).astype(np.float32)
    noisy = image_rgb.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def restore_degraded_image(
    image_rgb: np.ndarray,
    apply_clahe: bool = True,
    apply_bilateral: bool = True,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Classical DIP Restoration Pipeline:
    1. Gamma Correction for non-linear dark expansion: s = c * r^gamma
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB L-channel
    3. Bilateral Filtering for edge-preserving noise suppression.
    """
    processed = image_rgb.copy()

    # Step 1: Gamma correction
    if abs(gamma - 1.0) > 0.05:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        processed = cv2.LUT(processed, table)

    # Step 2: CLAHE on L-channel in CIE LAB color space
    if apply_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Step 3: Bilateral filter (smooths noise while preserving crisp boundary transitions)
    if apply_bilateral:
        processed = cv2.bilateralFilter(processed, d=7, sigmaColor=50, sigmaSpace=50)

    return processed

