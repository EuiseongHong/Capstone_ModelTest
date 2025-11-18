# src/speed_tracker.py
import numpy as np
from collections import defaultdict
from config import MAX_STEP_M, N_FRAMES_SPEED

class SpeedTracker:
    def __init__(self):
        self.track_history      = defaultdict(list)  # tid -> [(t, X, Y)]
        self.track_dsum         = defaultdict(float)
        self.track_tsum         = defaultdict(float)
        self.per_track_inst_all = defaultdict(list)

    def update_track(self, tid, t, X, Y, fi, fps):
        """
        Called once per object per frame after homography.
        Returns instantaneous speed (km/h) or None if not updated.
        """
        self.track_history[tid].append((t, X, Y))
        seq = self.track_history[tid]

        if len(seq) >= N_FRAMES_SPEED + 1 and (fi % N_FRAMES_SPEED == 0):
            (t0, X0, Y0) = seq[-(N_FRAMES_SPEED + 1)]
            (t1, X1, Y1) = seq[-1]
            dt = t1 - t0
            if dt > 1e-6:
                dist = float(np.hypot(X1 - X0, Y1 - Y0))
                if dist <= MAX_STEP_M * N_FRAMES_SPEED:
                    v_mps = dist / dt
                    v_kmh = v_mps * 3.6

                    self.track_dsum[tid]         += dist
                    self.track_tsum[tid]         += dt
                    self.per_track_inst_all[tid].append(v_mps)

                    return v_kmh
        return None
