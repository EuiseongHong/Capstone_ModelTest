# src/main_realtime.py
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

from config import (
    VIDEOS_DIR, RUNS_DIR, YOLO_WEIGHTS,
    USE_PRECALC_H, SAVE_AFTER_SOLVE,
    ACCIDENT_THRESH, MIN_ALERT_GAP_SEC,
    ENABLE_ACCIDENT_LOG,
)
from homography import load_homography, save_homography, solve_H_interactively, img_to_ground
from speed_tracker import SpeedTracker
from accident_features import build_accident_features_v1
from accident_model import load_accident_model

def main():
    name     = "crash_ver2"
    raw_path = os.path.join(VIDEOS_DIR, f"{name}.mov")
    run_dir  = os.path.join(RUNS_DIR, "detect", name)
    os.makedirs(run_dir, exist_ok=True)

    cap = cv2.VideoCapture(raw_path)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Himg = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or fps <= 0:
        raise RuntimeError("Could not read FPS.")

    print(f"FPS={fps}, W={W}, H={Himg}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Failed to read frame 0.")
    frame_path = os.path.join(run_dir, "frame_1.jpg")
    cv2.imwrite(frame_path, frame0)
    cap.release()

    # ---- homography ----
    H = None
    if USE_PRECALC_H:
        H = load_homography(name, W, Himg)
    if H is None:
        H = solve_H_interactively(frame_path)
        if SAVE_AFTER_SOLVE:
            save_homography(name, H, W, Himg)

    # ---- YOLO ----
    model = YOLO(YOLO_WEIGHTS)

    results_gen = model.track(
        source=raw_path,
        tracker="bytetrack.yaml",
        conf=0.4,
        persist=True,
        save=False,
        save_txt=False,
        show=False,
        stream=True,
        name=name,
        imgsz=640,
    )

    # ---- speed + accident modules ----
    tracker = SpeedTracker()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accident_model = load_accident_model(device)
    last_alert_time = defaultdict(lambda: -1.0)

    ENABLE_DISPLAY = False

    for fi, result in enumerate(results_gen):
        t = fi / fps
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            if ENABLE_DISPLAY:
                cv2.imshow("Speed View", result.orig_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        ids   = boxes.id.cpu().numpy().astype(int)
        xywhn = boxes.xywhn.cpu().numpy()

        for tid, (cx, cy, w, h) in zip(ids, xywhn):
            cx_px = cx * W
            cy_px = cy * Himg
            h_px  = h  * Himg

            x_bc = cx_px
            y_bc = cy_px + 0.5 * h_px

            X, Y = img_to_ground(x_bc, y_bc, H)

            v_kmh = tracker.update_track(tid, t, X, Y, fi, fps)
            if v_kmh is None:
                continue

            # ---- accident DNN ----
            features = build_accident_features_v1(tid, v_kmh, tracker)
            with torch.no_grad():
                x = torch.from_numpy(features).to(device).unsqueeze(0)
                logit = accident_model(x)
                prob = torch.sigmoid(logit)[0, 0].item()

            if prob >= ACCIDENT_THRESH and t - last_alert_time[tid] >= MIN_ALERT_GAP_SEC:
                last_alert_time[tid] = t
                if ENABLE_ACCIDENT_LOG:
                    print(
                        f"[ACCIDENT] t={t:.2f}s | track={tid} | "
                        f"prob={prob:.2f} | speed={v_kmh:.1f} km/h"
                    )

        if ENABLE_DISPLAY:
            frame = result.orig_img
            cv2.imshow("Speed View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
