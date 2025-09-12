import cv2
import numpy as np


# Simple horizontal parallax using depth (0=far, 1=near). Positive shift for near pixels.


def depth_to_disparity(depth: np.ndarray, max_px_shift: int = 16) -> np.ndarray:
# invert so near = larger shift
    disp = (1.0 - depth) * max_px_shift
    return disp.astype(np.float32)


def synthesize_stereo(bgr: np.ndarray, depth: np.ndarray, ipd_px: int = 8) -> tuple[np.ndarray, np.ndarray]:
    H, W, _ = bgr.shape
    disp = depth_to_disparity(depth, max_px_shift=ipd_px)


    # Left eye: shift content to the right for near pixels, Right eye: opposite
    map_x = np.tile(np.arange(W, dtype=np.float32), (H,1))
    map_y = np.tile(np.arange(H, dtype=np.float32)[:,None], (1,W))


    left_mapx = map_x + disp
    right_mapx = map_x - disp


    left = cv2.remap(bgr, left_mapx, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    right = cv2.remap(bgr, right_mapx, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return left, right


def stack_top_bottom(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.vstack([left, right])