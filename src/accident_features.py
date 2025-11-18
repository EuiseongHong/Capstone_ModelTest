# src/accident_features.py
import numpy as np

def build_accident_features_v1(tid, v_kmh, speed_tracker):
    speeds_mps = speed_tracker.per_track_inst_all[tid]
    if len(speeds_mps) >= 2:
        diffs = np.diff(speeds_mps)
        max_dec = float(np.min(diffs))  # negative = strongest decel
    else:
        max_dec = 0.0

    v_mps = v_kmh / 3.6
    mean_speed = float(np.mean(speeds_mps)) if speeds_mps else v_mps
    min_speed = float(np.min(speeds_mps)) if speeds_mps else v_mps

    duration = speed_tracker.track_tsum[tid]
    distance = speed_tracker.track_dsum[tid]

    feat = np.array([
        v_kmh,
        mean_speed * 3.6,
        min_speed * 3.6,
        max_dec,
        duration,
        distance,
    ], dtype=np.float32)
    return feat
