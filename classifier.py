import os
import glob
import re
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# -----------------------------
# CONFIG
# -----------------------------
BASE_RUN_DIR = "./runs/detect"   # where your YOLO runs live
VIDEO_CONFIG = {
    # run_name: (fps, width, height)
    "crash_ver2": (30.0, 1920, 1080),
    # add more videos here
}

# accident intervals in seconds for each video/run
ACCIDENT_INTERVALS = {
    "crash_ver2": [
        (0.0, 5.0),
    ],
    # "normal_drive": [],
}

# window settings
WIN_SEC  = 2.0   # window length in seconds
STEP_SEC = 1.0   # stride between windows in seconds

# -----------------------------
# HOMOGRAPHY (IMPORT YOUR H)
# -----------------------------
# Here you should load the same H you saved in your speed script
# For example, if you saved it as ./calib/H_crash_ver2.npz:
def load_homography(run_name):
    calib_dir = "./calib"
    H_file = os.path.join(calib_dir, f"H_{run_name}.npz")
    dat = np.load(H_file)
    H = dat["H"]
    return H

def img_to_ground(x_px, y_px, H):
    p = np.array([x_px, y_px, 1.0], dtype=float)
    q = H @ p
    X = q[0] / q[2]; Y = q[1] / q[2]
    return float(X), float(Y)

def natural_key(path):
    b = os.path.basename(path)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", b)]


# -----------------------------
# PARSE ONE RUN INTO FEATURES + LABELS
# -----------------------------
def build_features_for_run(run_name):
    """
    For a single run (e.g., 'crash_ver2'), build:
      X_run: feature matrix (num_windows, num_features)
      y_run: labels (0/1) per window
    """
    print(f"\n=== Building dataset for run: {run_name} ===")

    fps, W, Himg = VIDEO_CONFIG[run_name]
    WIN  = int(WIN_SEC  * fps)
    STEP = int(STEP_SEC * fps)

    H_mat = load_homography(run_name)

    labels_dir = os.path.join(BASE_RUN_DIR, run_name, "labels")
    txt_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")), key=natural_key)
    if not txt_files:
        raise FileNotFoundError(f"No txt files in {labels_dir}")

    # 1) Read all frames: list of detections
    frames = []  # list of list[det]
    for fi, pth in enumerate(txt_files):
        dets = []
        with open(pth, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                cls, cx, cy, w, h, tid = parts
                cls = int(cls)
                cx, cy, w, h = map(float, (cx, cy, w, h))
                tid = int(float(tid))
                dets.append({
                    "cls": cls,
                    "cx": cx, "cy": cy,
                    "w":  w,  "h":  h,
                    "track_id": tid,
                })
        frames.append(dets)

    n_frames = len(frames)
    print(f" Total frames: {n_frames}")

    # 2) Build per-track history to get speeds (m/s)
    track_history = defaultdict(list)  # tid -> list of (t, X, Y, v_mps)

    # We can also store speed back into frames[fi]
    for fi, dets in enumerate(frames):
        t = fi / fps
        for d in dets:
            tid = d["track_id"]
            cx_px = d["cx"] * W
            cy_px = d["cy"] * Himg
            h_px  = d["h"]  * Himg

            # bottom center
            x_bc = cx_px
            y_bc = cy_px + 0.5 * h_px

            X, Y = img_to_ground(x_bc, y_bc, H_mat)

            if not track_history[tid]:
                v_mps = 0.0
            else:
                t0, X0, Y0, _ = track_history[tid][-1]
                dt = t - t0
                if dt > 1e-6:
                    dist = float(np.hypot(X - X0, Y - Y0))
                    v_mps = dist / dt
                else:
                    v_mps = 0.0

            track_history[tid].append((t, X, Y, v_mps))
            # store speed back into detection
            d["X"] = X
            d["Y"] = Y
            d["speed_mps"] = v_mps

    # 3) Function to compute features from frames[start:end]
    def compute_window_features(start, end):
        # features we’ll use:
        # - max speed
        # - mean speed
        # - std speed
        # - number of "slow" vehicles (< 5 km/h)
        # - total number of detections
        speeds = []
        num_slow = 0
        count_det = 0

        for fi in range(start, end):
            if fi < 0 or fi >= n_frames:
                continue
            dets = frames[fi]
            for d in dets:
                v = d.get("speed_mps", 0.0)
                speeds.append(v)
                count_det += 1
                if v < (5 / 3.6):  # < 5 km/h
                    num_slow += 1

        if not speeds:
            return None

        speeds = np.array(speeds)
        feat_vec = np.array([
            speeds.max(),
            speeds.mean(),
            speeds.std(),
            num_slow,
            count_det,
        ], dtype=np.float32)
        return feat_vec

    # 4) Build X_run, y_run by sliding a window
    X_run = []
    y_run = []
    window_times = []  # (t_start, t_end) for debug

    accident_intervals = ACCIDENT_INTERVALS.get(run_name, [])

    def window_label(t_start, t_end):
        # 1 if window overlaps any accident interval
        for (a_start, a_end) in accident_intervals:
            if not (t_end < a_start or t_start > a_end):
                return 1
        return 0

    for start in range(0, n_frames - WIN, STEP):
        end = start + WIN
        t_start = start / fps
        t_end   = end   / fps
        feat_vec = compute_window_features(start, end)
        if feat_vec is None:
            continue
        label = window_label(t_start, t_end)

        X_run.append(feat_vec)
        y_run.append(label)
        window_times.append((t_start, t_end))

    X_run = np.vstack(X_run) if X_run else np.zeros((0, 5), dtype=np.float32)
    y_run = np.array(y_run, dtype=np.int64)
    print(f" Extracted {X_run.shape[0]} windows, {y_run.sum()} accidents")

    return X_run, y_run, window_times


# -----------------------------
# BUILD DATASET FOR ALL RUNS
# -----------------------------
all_X = []
all_y = []

for run_name in VIDEO_CONFIG.keys():
    X_run, y_run, _ = build_features_for_run(run_name)
    if X_run.size == 0:
        continue
    all_X.append(X_run)
    all_y.append(y_run)

X = np.vstack(all_X)
y = np.concatenate(all_y)

print("\n=== Final dataset ===")
print("X shape:", X.shape)
print("y distribution:", np.bincount(y))