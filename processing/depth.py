import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        print(f"[DepthEstimator] Using device: {self.device}")
        
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
        self.model = self.model.float()  # force float32
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray):
        """
        Input: BGR uint8 HxWx3
        Output: depth map normalized [0,1], quality estimate
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device).float()  # force float32
        pred = self.model(inp)

        # Resize to original resolution
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalize depth
        depth_min, depth_max = depth.min(), depth.max()
        d_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        # Optional quality score (variance of depth map)
        quality = np.var(d_norm)

        return d_norm, quality
