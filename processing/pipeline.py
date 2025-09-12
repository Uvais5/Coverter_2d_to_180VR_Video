import os
import cv2
import numpy as np
import torch
from .depth import DepthEstimator
from .stereo import synthesize_stereo, stack_top_bottom
from .project180 import barrel_wrap

class VR180Pipeline:
    def __init__(self, workdir: str, device: str = None):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VR180Pipeline] Using device: {self.device}")

    def run(self, input_video: str) -> str:
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(self.workdir, "vr180_tb.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height * 2))

        depth_estimator = DepthEstimator(device=self.device)

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize only for depth prediction (speed), keep original for stereo
            small_frame = cv2.resize(frame, (512, 512))
            d, quality = depth_estimator.predict(small_frame)
            print(f"[Frame {i}] Depth quality: {quality:.4f}")

            # Upscale depth map to original frame size
            d = cv2.resize(d, (frame.shape[1], frame.shape[0]))

            # Generate stereo and stack
            L, R = synthesize_stereo(frame, d, ipd_px=10)
            Lw = barrel_wrap(L, strength=0.6)
            Rw = barrel_wrap(R, strength=0.6)
            stacked = stack_top_bottom(Lw, Rw)

            writer.write(stacked)
            i += 1

        cap.release()
        writer.release()
        print(f"[VR180Pipeline] VR video saved to: {out_path}")
        return out_path
