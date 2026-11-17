"""
UI-independent core of the stereo extrinsics + rectification tool (step 2).

* Intrinsics come from step 1 (``01_camera_intrincis``): ``intrinsics_<serial>.json``.
* Detections reuse step 1's blinking chessboard detector (``calib_core``).
* Corner ordering is made physically consistent between the two cameras with the square colours
  (``canonicalize_corners``); otherwise OpenCV may number the corners of one camera in the opposite direction, which
  silently corrupts the stereo calibration.
* ``calibrate_stereo`` estimates R, T (pose of the right camera w.r.t. the left one: X_right = R X_left + T), E, F with
  per-pair outlier rejection, then computes the rectification (R1, R2, P1, P2, Q) and its quality (vertical disparity
  of the rectified corners).
* ``compute_maps`` gives the cv2.remap maps for frames; ``compute_event_lut`` gives, for every raw pixel, its rectified
  coordinates (to rectify events one by one).
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

STEP1_DIR = Path(__file__).resolve().parent.parent / "01_camera_intrincis"
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))
import calib_core as cc  # noqa: E402  (step 1 module: detector, drawing helpers, projection)

PINHOLE, FISHEYE = cc.PINHOLE, cc.FISHEYE
PAIRS_FILE = "recorded_pairs.json"
IMAGES_DIR = "pattern_images"


# ----------------------------------------------------------------------------------------------------------------------
# Intrinsics (output of step 1)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Intrinsics:
    serial: str
    model: str
    image_size: Tuple[int, int]  # (width, height)
    K: np.ndarray
    d: np.ndarray
    rms: float
    path: str
    pattern: dict = field(default_factory=dict)

    def short(self) -> str:
        return (f"{self.model}, fx={self.K[0, 0]:.1f} fy={self.K[1, 1]:.1f} cx={self.K[0, 2]:.1f} cy={self.K[1, 2]:.1f}, "
                f"RMS {self.rms:.3f}px")


def load_intrinsics(path) -> Intrinsics:
    data = json.loads(Path(path).read_text())
    model = data.get("camera_model", PINHOLE)
    d = np.asarray(data["distortion_coefficients"], dtype=np.float64)
    d = d.reshape(-1, 1) if model == FISHEYE else d.reshape(1, -1)
    return Intrinsics(serial=str(data.get("camera_serial", "")), model=model,
                      image_size=(int(data["image_size"]["width"]), int(data["image_size"]["height"])),
                      K=np.asarray(data["camera_matrix"], dtype=np.float64), d=d,
                      rms=float(data.get("overall_rms_reprojection_error", float("nan"))), path=str(path),
                      pattern=data.get("pattern", {}))


def find_intrinsics_file(directory, serial: str) -> Optional[Path]:
    """Latest intrinsics of a camera in the step 1 output folder."""
    directory = Path(directory)
    direct = directory / f"intrinsics_{serial}.json"
    if direct.exists():
        return direct
    candidates = sorted(directory.glob(f"calib_{serial}_*/intrinsics_{serial}.json"))
    return candidates[-1] if candidates else None


# ----------------------------------------------------------------------------------------------------------------------
# Detections
# ----------------------------------------------------------------------------------------------------------------------
def _square_value(frame: np.ndarray, g: np.ndarray, r: int, c: int) -> float:
    """Median intensity around the centre of the square bounded by inner corners (r, c) and (r + 1, c + 1)."""
    centre = g[r:r + 2, c:c + 2].reshape(-1, 2).mean(axis=0)
    x, y = int(round(centre[0])), int(round(centre[1]))
    h, w = frame.shape[:2]
    patch = frame[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)]
    return float(np.median(patch)) if patch.size else 128.0


def canonicalize_corners(corners: np.ndarray, frame: np.ndarray, cols: int, rows: int) -> Tuple[np.ndarray, bool]:
    """Orders the corners so that corner 0 is the same physical corner in every view and in both cameras.

    In the blinking pattern (see chessboard_pattern.html) the first inner square is a blinking square, i.e. black (0)
    in the detection frame, and, when cols + rows is odd, the last inner square is white. If the colours are the other
    way round, OpenCV returned the corners rotated by 180 deg and the order is reversed. Returns (corners, resolved).
    """
    if (cols + rows) % 2 == 0 or cols < 2 or rows < 2:
        return corners, False  # Symmetric pattern: the 180 deg ambiguity cannot be resolved
    g = corners.reshape(rows, cols, 2)
    first = _square_value(frame, g, 0, 0)
    last = _square_value(frame, g, rows - 2, cols - 2)
    if first < 128 <= last:
        return corners, True
    if last < 128 <= first:
        return corners[::-1].copy(), True
    return corners, False


@dataclass
class Detection:
    """Latest detection of one camera (used for pairing)."""
    wall_time: float
    corners: Optional[np.ndarray]
    frame: Optional[np.ndarray]
    stable: bool = False


@dataclass
class StereoPair:
    left: np.ndarray  # (N, 2) float32, canonical order
    right: np.ndarray
    frame_left: Optional[np.ndarray] = None  # uint8 binary detection frames
    frame_right: Optional[np.ndarray] = None
    wall_time: float = 0.0


class StereoPairCollector:
    """Decides whether a simultaneous detection in both cameras becomes a stereo pair."""

    def __init__(self, skip_time_s: float = 2.0, min_shift_px: float = 10.0, sync_window_s: float = 0.3,
                 stability_px: float = 1.5):
        self.skip_time_s = skip_time_s
        self.min_shift_px = min_shift_px
        self.sync_window_s = sync_window_s  # Max wall-clock difference between the two detections
        self.stability_px = stability_px  # Max corner motion between two consecutive detections of one camera
        self._last_time: Optional[float] = None

    def reset_timer(self) -> None:
        self._last_time = None

    def is_stable(self, prev: Optional[Detection], corners: Optional[np.ndarray], wall_time: float) -> bool:
        """The board is static in a camera if two consecutive detections (< 0.5 s apart) barely moved."""
        if corners is None or prev is None or prev.corners is None or wall_time - prev.wall_time > 0.5:
            return False
        return cc.mean_corner_shift(corners, prev.corners) < self.stability_px

    def accept(self, left: Detection, right: Detection, existing: Sequence[StereoPair]) -> Tuple[bool, str]:
        if left.corners is None or right.corners is None:
            return False, "chessboard not seen by both cameras"
        if abs(left.wall_time - right.wall_time) > self.sync_window_s:
            return False, "detections not simultaneous"
        if not (left.stable and right.stable):
            return False, "waiting for a stable view in both cameras"
        now = max(left.wall_time, right.wall_time)
        if self._last_time is not None and now - self._last_time < self.skip_time_s:
            return False, "waiting (skip time)"
        if self.min_shift_px > 0 and existing:
            shift = min(0.5 * (cc.mean_corner_shift(left.corners, p.left) + cc.mean_corner_shift(right.corners, p.right))
                        for p in existing)
            if shift < self.min_shift_px:
                return False, "pose already captured - move the board/rig"
        self._last_time = now
        return True, "captured"


# ----------------------------------------------------------------------------------------------------------------------
# Stereo calibration and rectification
# ----------------------------------------------------------------------------------------------------------------------
def _project(obj, rvec, tvec, K, d, model) -> np.ndarray:
    return cc._project(obj, rvec, tvec, K, d, model)


def rectify_points(pts: np.ndarray, K, d, R, P, model: str) -> np.ndarray:
    """Raw pixel coordinates -> rectified pixel coordinates."""
    src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if model == FISHEYE:
        out = cv2.fisheye.undistortPoints(src, K, d, R=R, P=P)
    else:
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        out = cv2.undistortPointsIter(src, K, d, R, P, crit)
    return out.reshape(-1, 2)


@dataclass
class StereoResult:
    model: str
    image_size: Tuple[int, int]
    serial_left: str
    serial_right: str
    K1: np.ndarray
    d1: np.ndarray
    K2: np.ndarray
    d2: np.ndarray
    R: np.ndarray  # X_right = R @ X_left + T
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    rms: float
    selected: np.ndarray  # bool, one per pair
    per_pair_errors: np.ndarray  # RMS reprojection error (px), one per *selected* pair
    rvecs: List[np.ndarray]  # Board pose in the left camera, one per *selected* pair
    tvecs: List[np.ndarray]
    R1: np.ndarray = None
    R2: np.ndarray = None
    P1: np.ndarray = None
    P2: np.ndarray = None
    Q: np.ndarray = None
    roi1: Tuple[int, int, int, int] = (0, 0, 0, 0)
    roi2: Tuple[int, int, int, int] = (0, 0, 0, 0)
    alpha: float = 0.0
    rect_errors: np.ndarray = None  # RMS vertical disparity (px) of the rectified corners, one per pair (all pairs)
    intrinsics_refined: bool = False
    pattern: dict = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def n_pairs(self) -> int:
        return int(self.selected.size)

    @property
    def n_kept(self) -> int:
        return int(self.selected.sum())

    @property
    def horizontal(self) -> bool:
        """True for a horizontal rig (rectified images side by side, epipolar lines horizontal)."""
        return bool(abs(self.P2[0, 3]) >= abs(self.P2[1, 3]))

    def rect_stats(self) -> dict:
        e = self.rect_errors[self.selected] if self.rect_errors is not None else np.zeros(0)
        if e.size == 0:
            return {}
        return {"mean_rms_px": float(e.mean()), "max_rms_px": float(e.max()), "median_rms_px": float(np.median(e))}

    def pair_error(self, i: int) -> Optional[float]:
        if not self.selected[i]:
            return None
        return float(self.per_pair_errors[int(self.selected[:i].sum())])

    def rotation_deg(self) -> float:
        return float(np.degrees(np.linalg.norm(cv2.Rodrigues(self.R)[0])))

    def warnings(self) -> List[str]:
        out = []
        T = self.T.ravel()
        if self.horizontal and T[0] > 0:
            out.append("T_x > 0: the 'left' camera is physically on the RIGHT. Swap the left/right cameras "
                       "(disparities would otherwise be negative).")
        if not self.horizontal and T[1] > 0:
            out.append("Vertical rig with T_y > 0: the 'left' camera is BELOW the 'right' one.")
        stats = self.rect_stats()
        if stats and stats["mean_rms_px"] > 0.5:
            out.append(f"Large rectification error ({stats['mean_rms_px']:.2f} px): check the intrinsics and capture "
                       "more pairs covering the whole field of view at various distances.")
        return out

    def summary(self) -> str:
        T = self.T.ravel()
        rv = np.degrees(cv2.Rodrigues(self.R)[0].ravel())
        stats = self.rect_stats()
        axis = "horizontal" if self.horizontal else "VERTICAL"
        lines = [
            f"Left: {self.serial_left}   Right: {self.serial_right}",
            f"Model: {self.model}   image {self.image_size[0]}x{self.image_size[1]}"
            + ("   (intrinsics refined)" if self.intrinsics_refined else "   (intrinsics fixed)"),
            f"Pairs kept: {self.n_kept}/{self.n_pairs}",
            f"Stereo RMS reprojection error: {self.rms:.4f} px",
            "",
            f"Baseline |T| = {self.baseline * 1000:.2f} mm   ({axis} rig)",
            f"T = [{T[0] * 1000:.2f}, {T[1] * 1000:.2f}, {T[2] * 1000:.2f}] mm",
            f"R (Rodrigues) = [{rv[0]:.3f}, {rv[1]:.3f}, {rv[2]:.3f}] deg   (|angle| {self.rotation_deg():.3f} deg)",
            "",
            f"Rectification (alpha = {self.alpha:.2f}):",
            f"  f = {self.P1[0, 0]:.2f} px   cx1 = {self.P1[0, 2]:.2f}  cx2 = {self.P2[0, 2]:.2f}  cy = {self.P1[1, 2]:.2f}",
        ]
        if stats:
            lines.append(f"  Rectified {'vertical' if self.horizontal else 'horizontal'} disparity error: "
                         f"mean {stats['mean_rms_px']:.3f} px, median {stats['median_rms_px']:.3f}, "
                         f"max {stats['max_rms_px']:.3f}")
        for w in self.warnings():
            lines.append(f"WARNING: {w}")
        lines += ["", "Per-pair errors (reprojection | rectified epipolar):"]
        for i in range(self.n_pairs):
            e = self.pair_error(i)
            r = self.rect_errors[i] if self.rect_errors is not None else float("nan")
            lines.append(f"  pair {i + 1:3d}: " + ("REJECTED (outlier)" if e is None else f"{e:.3f} px")
                         + f" | {r:.3f} px")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def m(a):
            return np.asarray(a, dtype=np.float64).tolist()

        return {
            "camera_model": self.model,
            "image_size": {"width": int(self.image_size[0]), "height": int(self.image_size[1])},
            "left": {"serial": self.serial_left, "camera_matrix": m(self.K1),
                     "distortion_coefficients": np.asarray(self.d1).ravel().tolist(),
                     "rectification_rotation": m(self.R1), "projection_matrix": m(self.P1),
                     "valid_roi_xywh": [int(v) for v in self.roi1]},
            "right": {"serial": self.serial_right, "camera_matrix": m(self.K2),
                      "distortion_coefficients": np.asarray(self.d2).ravel().tolist(),
                      "rectification_rotation": m(self.R2), "projection_matrix": m(self.P2),
                      "valid_roi_xywh": [int(v) for v in self.roi2]},
            "R": m(self.R), "T": np.asarray(self.T).ravel().tolist(), "E": m(self.E), "F": m(self.F), "Q": m(self.Q),
            "convention": "X_right = R @ X_left + T (meters); rectified pixel = P @ R_rect @ undistorted ray",
            "baseline_m": self.baseline,
            "rig_orientation": "horizontal" if self.horizontal else "vertical",
            "alpha": float(self.alpha),
            "stereo_rms_reprojection_error": float(self.rms),
            "n_pairs": self.n_pairs,
            "n_kept_pairs": self.n_kept,
            "selected_pairs": [int(b) for b in self.selected],
            "per_pair_rms_reprojection_errors": [float(e) for e in self.per_pair_errors],
            "rectification_errors_px": [float(e) for e in self.rect_errors],
            "rectification_error_stats": self.rect_stats(),
            "intrinsics_refined": bool(self.intrinsics_refined),
            "pattern": self.pattern,
        }


def _stereo_calibrate_once(obj, ptsL, ptsR, K1, d1, K2, d2, size, model, fix_intrinsics):
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    n = len(ptsL)
    if model == FISHEYE:
        flags = (cv2.fisheye.CALIB_FIX_INTRINSIC if fix_intrinsics else
                 cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_FIX_SKEW)
        objs = [obj.reshape(1, -1, 3).astype(np.float64) for _ in range(n)]
        ipl = [p.reshape(1, -1, 2).astype(np.float64) for p in ptsL]
        ipr = [p.reshape(1, -1, 2).astype(np.float64) for p in ptsR]
        out = cv2.fisheye.stereoCalibrate(objs, ipl, ipr, K1.copy(), d1.copy(), K2.copy(), d2.copy(), size,
                                          flags=flags, criteria=crit)
        rms, K1n, d1n, K2n, d2n, R, T = out[:7]
        rvecs, tvecs = (list(out[7]), list(out[8])) if len(out) >= 9 else ([], [])
        T = np.asarray(T, np.float64).reshape(3, 1)
        tx = np.array([[0, -T[2, 0], T[1, 0]], [T[2, 0], 0, -T[0, 0]], [-T[1, 0], T[0, 0], 0]])
        E = tx @ R
        F = np.linalg.inv(K2n).T @ E @ np.linalg.inv(K1n)  # In the (fisheye-)undistorted pixel space
    else:
        flags = (cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else
                 cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO)
        objs = [obj.astype(np.float32) for _ in range(n)]
        ipl = [p.reshape(-1, 1, 2).astype(np.float32) for p in ptsL]
        ipr = [p.reshape(-1, 1, 2).astype(np.float32) for p in ptsR]
        out = cv2.stereoCalibrateExtended(objs, ipl, ipr, K1.copy(), d1.copy(), K2.copy(), d2.copy(), size,
                                          np.eye(3), np.zeros((3, 1)), flags=flags, criteria=crit)
        rms, K1n, d1n, K2n, d2n, R, T, E, F, rvecs, tvecs, _ = out
        rvecs, tvecs = list(rvecs), list(tvecs)
    if not rvecs:  # Old OpenCV: board poses from the left camera only
        for p in ptsL:
            ok, rv, tv = _left_pose(obj, p, K1n, d1n, model)
            rvecs.append(rv)
            tvecs.append(tv)
    errs = []
    for p_l, p_r, rv, tv in zip(ptsL, ptsR, rvecs, tvecs):
        Rl, _ = cv2.Rodrigues(rv)
        rv_r, _ = cv2.Rodrigues(R @ Rl)
        tv_r = R @ np.asarray(tv).reshape(3, 1) + T.reshape(3, 1)
        el = _project(obj, rv, tv, K1n, d1n, model) - p_l.reshape(-1, 2)
        er = _project(obj, rv_r, tv_r, K2n, d2n, model) - p_r.reshape(-1, 2)
        errs.append(math.sqrt(np.mean(np.concatenate([np.sum(el ** 2, 1), np.sum(er ** 2, 1)]))))
    return float(rms), K1n, d1n, K2n, d2n, R, T.reshape(3, 1), E, F, rvecs, tvecs, np.array(errs)


def _left_pose(obj, pts, K, d, model):
    if model == FISHEYE:
        norm = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, d)
        return cv2.solvePnP(obj.astype(np.float64), norm, np.eye(3), None)
    return cv2.solvePnP(obj.astype(np.float64), pts.astype(np.float64), K, d)


def compute_rectification(res: StereoResult, alpha: float = 0.0) -> None:
    """Fills R1, R2, P1, P2, Q, ROIs and the rectification errors of a StereoResult (in place).

    alpha (pinhole) / balance (fisheye) in [0, 1]: 0 = only valid pixels (zoomed in), 1 = keep every source pixel.
    """
    size = tuple(res.image_size)
    if res.model == FISHEYE:
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(res.K1, res.d1, res.K2, res.d2, size, res.R, res.T,
                                                      flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=size,
                                                      balance=float(alpha), fov_scale=1.0)
        roi1 = roi2 = (0, 0, size[0], size[1])
    else:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(res.K1, res.d1, res.K2, res.d2, size, res.R, res.T,
                                                          flags=cv2.CALIB_ZERO_DISPARITY, alpha=float(alpha),
                                                          newImageSize=size)
    res.R1, res.R2, res.P1, res.P2, res.Q = R1, R2, P1, P2, Q
    res.roi1, res.roi2 = tuple(int(v) for v in roi1), tuple(int(v) for v in roi2)
    res.alpha = float(alpha)


def rectification_errors(res: StereoResult, ptsL: Sequence[np.ndarray], ptsR: Sequence[np.ndarray]) -> np.ndarray:
    """RMS of the epipolar misalignment of the rectified corners (y difference for a horizontal rig), per pair."""
    axis = 1 if res.horizontal else 0
    errs = []
    for p_l, p_r in zip(ptsL, ptsR):
        rl = rectify_points(p_l, res.K1, res.d1, res.R1, res.P1, res.model)
        rr = rectify_points(p_r, res.K2, res.d2, res.R2, res.P2, res.model)
        errs.append(math.sqrt(np.mean((rl[:, axis] - rr[:, axis]) ** 2)))
    return np.array(errs)


def calibrate_stereo(ptsL: Sequence[np.ndarray], ptsR: Sequence[np.ndarray], intr_l: Intrinsics, intr_r: Intrinsics,
                     cols: int, rows: int, square_w: float, square_h: float, outlier_ths: float = 2.0,
                     fix_intrinsics: bool = True, alpha: float = 0.0,
                     log: Callable[[str], None] = print) -> StereoResult:
    t0 = time.time()
    if len(ptsL) != len(ptsR) or len(ptsL) < 3:
        raise ValueError(f"At least 3 stereo pairs are required (got {len(ptsL)}).")
    if intr_l.model != intr_r.model:
        raise ValueError(f"Both cameras must use the same camera model (left: {intr_l.model}, right: {intr_r.model}).")
    if tuple(intr_l.image_size) != tuple(intr_r.image_size):
        raise ValueError(f"Image sizes differ (left {intr_l.image_size}, right {intr_r.image_size}).")
    model, size = intr_l.model, tuple(intr_l.image_size)
    obj = cc.make_object_points(cols, rows, square_w, square_h)
    ptsL = [np.asarray(p, np.float32).reshape(-1, 2) for p in ptsL]
    ptsR = [np.asarray(p, np.float32).reshape(-1, 2) for p in ptsR]

    log(f"Stereo calibration ({model}, intrinsics {'fixed' if fix_intrinsics else 'refined'}) with {len(ptsL)} pairs...")
    sel = np.ones(len(ptsL), dtype=bool)
    out = _stereo_calibrate_once(obj, ptsL, ptsR, intr_l.K, intr_l.d, intr_r.K, intr_r.d, size, model, fix_intrinsics)
    log(f"First pass: RMS = {out[0]:.4f} px")
    if outlier_ths >= 0 and len(ptsL) >= 4:
        errs = out[-1]
        keep = errs <= errs.mean() + outlier_ths * errs.std()
        if not keep.all() and keep.sum() >= 3:
            sel = keep
            out = _stereo_calibrate_once(obj, [p for p, k in zip(ptsL, keep) if k],
                                         [p for p, k in zip(ptsR, keep) if k], intr_l.K, intr_l.d, intr_r.K, intr_r.d,
                                         size, model, fix_intrinsics)
            log(f"Removed {int((~keep).sum())} outlier pair(s): RMS = {out[0]:.4f} px")
    rms, K1, d1, K2, d2, R, T, E, F, rvecs, tvecs, errs = out
    res = StereoResult(model=model, image_size=size, serial_left=intr_l.serial, serial_right=intr_r.serial,
                       K1=K1, d1=d1, K2=K2, d2=d2, R=R, T=T, E=E, F=F, rms=rms, selected=sel, per_pair_errors=errs,
                       rvecs=rvecs, tvecs=tvecs, intrinsics_refined=not fix_intrinsics,
                       pattern={"n_cols": cols, "n_rows": rows, "square_width": float(square_w),
                                "square_height": float(square_h)})
    compute_rectification(res, alpha)
    res.rect_errors = rectification_errors(res, ptsL, ptsR)
    res.duration_s = time.time() - t0
    stats = res.rect_stats()
    log(f"Baseline {res.baseline * 1000:.2f} mm, rectified epipolar error {stats.get('mean_rms_px', float('nan')):.3f} px "
        f"(mean over kept pairs)")
    return res


def compute_maps(res: StereoResult, side: str, m1type=cv2.CV_32FC1) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps (rectified image <- raw image) of one camera ('left' or 'right')."""
    K, d, R, P = (res.K1, res.d1, res.R1, res.P1) if side == "left" else (res.K2, res.d2, res.R2, res.P2)
    size = tuple(res.image_size)
    if res.model == FISHEYE:
        return cv2.fisheye.initUndistortRectifyMap(K, d, R, P, size, m1type)
    return cv2.initUndistortRectifyMap(K, d, R, P, size, m1type)


def compute_event_lut(res: StereoResult, side: str) -> np.ndarray:
    """(H, W, 2) float32: rectified (x, y) of every raw pixel, to rectify events: lut[y, x]."""
    K, d, R, P = (res.K1, res.d1, res.R1, res.P1) if side == "left" else (res.K2, res.d2, res.R2, res.P2)
    w, h = res.image_size
    grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)), -1).reshape(-1, 2)
    return rectify_points(grid, K, d, R, P, res.model).reshape(h, w, 2).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------------------------------
def draw_epipolar_lines(img: np.ndarray, horizontal: bool = True, step: int = 40, color=(0, 200, 0)) -> None:
    h, w = img.shape[:2]
    if horizontal:
        for y in range(step // 2, h, step):
            cv2.line(img, (0, y), (w - 1, y), color, 1)
    else:
        for x in range(step // 2, w, step):
            cv2.line(img, (x, 0), (x, h - 1), color, 1)


def to_bgr(frame: Optional[np.ndarray], size: Tuple[int, int]) -> np.ndarray:
    if frame is None:
        return np.full((size[1], size[0], 3), 255, np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()


def draw_pair(pair: StereoPair, cols: int, rows: int, size: Tuple[int, int], res: Optional[StereoResult] = None,
              maps=None, index: int = 0) -> np.ndarray:
    """Side-by-side image of a stereo pair: raw with corners, or rectified with epipolar lines if ``res`` is given."""
    imgs = []
    for side, frame, pts in (("left", pair.frame_left, pair.left), ("right", pair.frame_right, pair.right)):
        img = to_bgr(frame, size)
        if res is not None and maps is not None:
            mx, my = maps[side]
            img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 60, 60))
            K, d, R, P = (res.K1, res.d1, res.R1, res.P1) if side == "left" else (res.K2, res.d2, res.R2, res.P2)
            pts = rectify_points(pts, K, d, R, P, res.model).astype(np.float32)
            draw_epipolar_lines(img, res.horizontal)
        cv2.drawChessboardCorners(img, (cols, rows), pts.reshape(-1, 1, 2).astype(np.float32), True)
        cv2.circle(img, tuple(np.round(pts[0]).astype(int)), 9, (255, 0, 255), 3)  # Corner 0 (must match)
        cc.put_label(img, side.upper(), (10, img.shape[0] - 50))
        imgs.append(img)
    out = np.hstack(imgs)
    if res is not None:
        e = res.pair_error(index)
        txt = (f"Pair {index + 1}: " + ("REJECTED (outlier)" if e is None else f"reprojection {e:.3f} px")
               + f" | epipolar error {res.rect_errors[index]:.3f} px")
        cc.put_label(out, txt, (10, 10))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------------------------------
def save_pairs(directory, pairs: Sequence[StereoPair], image_size, cols, rows, square_w, square_h,
               serial_left: str, serial_right: str) -> Path:
    directory = Path(directory)
    img_dir = directory / IMAGES_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):
        old.unlink()
    for i, p in enumerate(pairs):
        if p.frame_left is not None:
            cv2.imwrite(str(img_dir / f"left_{i + 1}.png"), p.frame_left)
        if p.frame_right is not None:
            cv2.imwrite(str(img_dir / f"right_{i + 1}.png"), p.frame_right)
    data = {
        "serial_left": serial_left, "serial_right": serial_right,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "pattern": {"n_cols": cols, "n_rows": rows, "square_width": float(square_w), "square_height": float(square_h)},
        "pairs": [{"wall_time": float(p.wall_time), "left": np.asarray(p.left).round(4).tolist(),
                   "right": np.asarray(p.right).round(4).tolist()} for p in pairs],
    }
    path = directory / PAIRS_FILE
    path.write_text(json.dumps(data, indent=1))
    return path


def load_pairs(directory):
    """Returns (image_size, pattern, pairs, serial_left, serial_right)."""
    directory = Path(directory)
    data = json.loads((directory / PAIRS_FILE).read_text())
    size = (int(data["image_size"]["width"]), int(data["image_size"]["height"]))
    pattern = data["pattern"]
    n = int(pattern["n_cols"]) * int(pattern["n_rows"])
    pairs = []
    for i, p in enumerate(data["pairs"]):
        left = np.asarray(p["left"], np.float32).reshape(-1, 2)
        right = np.asarray(p["right"], np.float32).reshape(-1, 2)
        if len(left) != n or len(right) != n:
            raise ValueError(f"Pair {i + 1} does not have {n} corners per camera")
        fl, fr = directory / IMAGES_DIR / f"left_{i + 1}.png", directory / IMAGES_DIR / f"right_{i + 1}.png"
        pairs.append(StereoPair(left, right,
                                cv2.imread(str(fl), cv2.IMREAD_GRAYSCALE) if fl.exists() else None,
                                cv2.imread(str(fr), cv2.IMREAD_GRAYSCALE) if fr.exists() else None,
                                float(p.get("wall_time", 0.0))))
    return size, pattern, pairs, str(data.get("serial_left", "")), str(data.get("serial_right", ""))


def save_stereo(directory, res: StereoResult, name: str, metadata: Optional[dict] = None) -> Tuple[Path, Path]:
    """Writes <name>.json (plain JSON) and <name>_opencv.yaml (cv2.FileStorage, same names as OpenCV's stereo sample)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = {"date": time.strftime("%Y-%m-%d %H:%M:%S")}
    data.update(metadata or {})
    data.update(res.to_dict())
    json_path = directory / f"{name}.json"
    json_path.write_text(json.dumps(data, indent=2))

    yaml_path = directory / f"{name}_opencv.yaml"
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_model", res.model)
    fs.write("serial_left", res.serial_left)
    fs.write("serial_right", res.serial_right)
    fs.write("image_width", int(res.image_size[0]))
    fs.write("image_height", int(res.image_size[1]))
    for key, val in (("M1", res.K1), ("D1", np.asarray(res.d1).reshape(1, -1)), ("M2", res.K2),
                     ("D2", np.asarray(res.d2).reshape(1, -1)), ("R", res.R), ("T", res.T), ("E", res.E),
                     ("F", res.F), ("R1", res.R1), ("R2", res.R2), ("P1", res.P1), ("P2", res.P2), ("Q", res.Q),
                     ("roi1", np.asarray(res.roi1, np.float64).reshape(1, 4)),
                     ("roi2", np.asarray(res.roi2, np.float64).reshape(1, 4))):
        fs.write(key, np.asarray(val, dtype=np.float64))
    fs.write("baseline_m", res.baseline)
    fs.write("stereo_rms_reprojection_error", float(res.rms))
    fs.release()
    return json_path, yaml_path


def load_stereo(path) -> StereoResult:
    """Reads a stereo_<left>_<right>.json written by save_stereo (used by step 3)."""
    data = json.loads(Path(path).read_text())
    left, right = data["left"], data["right"]
    model = data.get("camera_model", PINHOLE)

    def dist(v):
        a = np.asarray(v, dtype=np.float64)
        return a.reshape(-1, 1) if model == FISHEYE else a.reshape(1, -1)

    n = int(data.get("n_pairs", 0))
    res = StereoResult(
        model=model, image_size=(int(data["image_size"]["width"]), int(data["image_size"]["height"])),
        serial_left=str(left.get("serial", "")), serial_right=str(right.get("serial", "")),
        K1=np.asarray(left["camera_matrix"], np.float64), d1=dist(left["distortion_coefficients"]),
        K2=np.asarray(right["camera_matrix"], np.float64), d2=dist(right["distortion_coefficients"]),
        R=np.asarray(data["R"], np.float64), T=np.asarray(data["T"], np.float64).reshape(3, 1),
        E=np.asarray(data["E"], np.float64), F=np.asarray(data["F"], np.float64),
        rms=float(data.get("stereo_rms_reprojection_error", float("nan"))),
        selected=np.asarray(data.get("selected_pairs", [1] * n), dtype=bool),
        per_pair_errors=np.asarray(data.get("per_pair_rms_reprojection_errors", []), np.float64),
        rvecs=[], tvecs=[],
        R1=np.asarray(left["rectification_rotation"], np.float64), R2=np.asarray(right["rectification_rotation"], np.float64),
        P1=np.asarray(left["projection_matrix"], np.float64), P2=np.asarray(right["projection_matrix"], np.float64),
        Q=np.asarray(data["Q"], np.float64), roi1=tuple(left.get("valid_roi_xywh", (0, 0, 0, 0))),
        roi2=tuple(right.get("valid_roi_xywh", (0, 0, 0, 0))), alpha=float(data.get("alpha", 0.0)),
        rect_errors=np.asarray(data.get("rectification_errors_px", []), np.float64),
        intrinsics_refined=bool(data.get("intrinsics_refined", False)), pattern=data.get("pattern", {}))
    return res


def find_stereo_file(directory, serial_left: str, serial_right: str) -> Optional[Path]:
    """Latest stereo calibration of a camera pair in the step 2 output folder."""
    directory = Path(directory)
    name = f"stereo_{serial_left}_{serial_right}"
    if (directory / f"{name}.json").exists():
        return directory / f"{name}.json"
    candidates = sorted(directory.glob(f"{name}_*/{name}.json"))
    return candidates[-1] if candidates else None


def save_maps(path, res: StereoResult) -> Path:
    """Rectification look-up tables (compressed .npz):

    * map{1,2}_{x,y}: cv2.remap maps (rectified <- raw) to rectify frames: cv2.remap(img, map1_x, map1_y, ...)
    * event_lut{1,2}: (H, W, 2) rectified coordinates of each raw pixel, to rectify events: lut[y, x] -> (x', y')
    """
    path = Path(path)
    arrays = {}
    for i, side in ((1, "left"), (2, "right")):
        mx, my = compute_maps(res, side)
        arrays[f"map{i}_x"], arrays[f"map{i}_y"] = mx, my
        arrays[f"event_lut{i}"] = compute_event_lut(res, side)
    np.savez_compressed(path, image_size=np.array(res.image_size), **arrays)
    return path
