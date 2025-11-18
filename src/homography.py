import os
import cv2
import numpy as np
from config import CALIB_DIR

def img_to_ground(x_px, y_px, H):
    p = np.array([x_px, y_px, 1.0], dtype=float)
    q = H @ p
    X = q[0] / q[2]; Y = q[1] / q[2]
    return float(X), float(Y)

def load_homography(run_name, cur_W, cur_H):
    path_npz = os.path.join(CALIB_DIR, f"H_{run_name}.npz")
    if not os.path.exists(path_npz):
        return None

    data = np.load(path_npz)
    H = data["H"]
    if H.shape != (3, 3):
        raise ValueError("Loaded homography is not 3x3.")
    W0 = int(data["calib_width"])
    H0 = int(data["calib_height"])

    if (W0 != cur_W) or (H0 != cur_H):
        sx = W0 / float(cur_W)
        sy = H0 / float(cur_H)
        S = np.array([[sx, 0, 0],
                      [0,  sy, 0],
                      [0,  0,  1]], dtype=float)
        H = H @ S
        print(f"[homography] Adjusted: calib=({W0},{H0}) -> current=({cur_W},{cur_H})")
    else:
        print("[homography] Resolution matches calibration.")

    return H

def save_homography(run_name, H, W, Himg):
    os.makedirs(CALIB_DIR, exist_ok=True)
    path_npz = os.path.join(CALIB_DIR, f"H_{run_name}.npz")
    np.savez(path_npz, H=H, calib_width=int(W), calib_height=int(Himg))
    print(f"[homography] Saved to {path_npz}")

def solve_H_interactively(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to load {img_path}")

    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("frame", img)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", img)
    print("Left-click 4 ground points (A,B,C,D); press any key when done.")
    cv2.setMouseCallback("frame", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 4:
        raise ValueError("Need at least 4 points for homography.")
    img_points = np.array(points[:4], dtype=np.float32)

    real_coords = np.array([ 
        [3.5, 0.0],   #choose right side first
        [3.5, 8.0],
        [0.0,  0.0],
        [0.0,  8.0]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_points, real_coords, cv2.RANSAC, 2.0)
    if H is None or H.shape != (3, 3):
        raise ValueError("Homography solve failed.")
    print("Homography:\n", H)
    return H
