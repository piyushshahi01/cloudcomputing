import os
import cv2
import numpy as np
from generate_samples import create_road_background, draw_sign_post

def generate_sample_video(output_path="sample_data/traffic_drive.mp4", num_frames=90, fps=25):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    width, height = 640, 480
    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + (np.pi / 8.0)

    for f in range(num_frames):
        # Progressively approach the sign (zoom in)
        progress = f / float(num_frames)
        radius = int(25 + 40 * progress) # Grows from 25 to 65 px
        cx = int(450 + 50 * progress)
        cy = int(220 + 20 * progress)

        frame = create_road_background(width, height)
        draw_sign_post(frame, cx, cy + radius, height=int(100 + 50 * progress))

        # Draw approaching Stop Sign
        oct_pts = np.array([[int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))] for a in angles], dtype=np.int32)
        cv2.fillPoly(frame, [oct_pts], (210, 20, 20))
        cv2.polylines(frame, [oct_pts], True, (255, 255, 255), max(1, int(2 + 2 * progress)))
        
        font_scale = 0.4 + 0.5 * progress
        cv2.putText(frame, "STOP", (int(cx - 20 * (radius / 30.0)), int(cy + 6 * (radius / 30.0))), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), max(1, int(font_scale * 2)))

        # Convert RGB to BGR for OpenCV VideoWriter
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Sample video successfully saved to {output_path}")

if __name__ == "__main__":
    generate_sample_video()

