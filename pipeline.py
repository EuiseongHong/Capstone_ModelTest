"""
Video → YOLO + ByteTrack → per-track positions (X,Y over time) → speed over time
                                                        ↓
                                           Simple DNN classifier (MLP)
                                                        ↓
                                          P(accident | track, last few sec)
"""
import numpy as np


def build_accident_features_v1(tid, v_kmh, per_track_inst_all, track_tsum, track_dsum):
    speeds_mps = per_track_inst_all[tid]  #list of v_mps
    if len(speeds_mps) >= 2:
        diffs = np.diff(speeds_mps)
        max_dec = float(np.min(diffs))  #negative = deceleration
    else:
        max_dec = 0.0

    v_mps = v_kmh / 3.6
    mean_speed = float(np.mean(speeds_mps)) if speeds_mps else v_mps
    min_speed = float(np.min(speeds_mps)) if speeds_mps else v_mps

    duration = track_tsum[tid]   #seconds
    distance = track_dsum[tid]   #meters

    feat = np.array([
        v_kmh,
        mean_speed * 3.6,  #convert to km/h
        min_speed * 3.6,
        max_dec,           # m/s^2-ish (negative)
        duration,
        distance,
    ], dtype=np.float32)
    return feat