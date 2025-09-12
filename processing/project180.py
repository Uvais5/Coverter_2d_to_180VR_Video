import cv2
import numpy as np


# MVP: simulate a 180° wrap by applying barrel distortion (fisheye-like) and letterboxing.
# For a true 180 equirectangular projection, you'd map rays on a hemisphere. This simplified
# version gives a convincing half-dome wrap for demo purposes.


def barrel_wrap(bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    H, W = bgr.shape[:2]
    cx, cy = W/2, H/2
    xs = (np.linspace(0, W-1, W) - cx) / cx
    ys = (np.linspace(0, H-1, H) - cy) / cy
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv**2 + yv**2)
    factor = 1 + strength * (r**2)
    map_x = (xv / factor) * cx + cx
    map_y = (yv / factor) * cy + cy
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    warped = cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped