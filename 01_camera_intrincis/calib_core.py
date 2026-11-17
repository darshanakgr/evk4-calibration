"""
UI-independent core of the EVK4 intrinsic calibration tool.

This is a Python port of the logic found in the Metavision SDK C++ samples
``metavision_mono_calibration_recording`` (blinking chessboard detection) and
``metavision_mono_calibration`` (OpenCV calibration with outlier rejection and
optional fronto-parallel refinement).

Pipeline
--------
events --> (trail filter, done by the caller) --> BlinkingFrameGenerator
       --> binary "blinking" frame (blinking pixels = black squares)
       --> cv2.findChessboardCorners / cornerSubPix --> BlinkingWindow
       --> ViewCollector (skip time + minimum pose change) --> calibration views
       --> run_calibration() --> CalibrationResult
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PINHOLE = "pinhole"
FISHEYE = "fisheye"


# ----------------------------------------------------------------------------------------------------------------------
# Blinking frame generation
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class BlinkingConfig:
    """Parameters of the blinking frame generator (same meaning as Metavision's BlinkingFrameGeneratorAlgorithmConfig).

    The object is read at every evaluation, so it can be modified live from another thread.
    """
    accumulation_time_us: int = 100_000  # Window during which a pixel must receive both ON and OFF events
    min_blinking_pixels: int = 100  # Minimum number of blinking pixels to output a frame
    ratio_on: float = 0.15  # Max ratio (#pixels with only ON events) / (#blinking pixels)
    ratio_off: float = 0.15  # Max ratio (#pixels with only OFF events) / (#blinking pixels)
    median_blur_radius: int = 1  # Median blur applied on the blinking mask (0 = disabled)


class BlinkingFrameGenerator:
    """Accumulates events over fixed time windows and produces a binary frame of the pixels that blinked.

    A pixel is "blinking" if it received both positive and negative events during the window. A frame is produced
    only if enough pixels blink and if few pixels received events of a single polarity (which happens when the camera
    or the scene moves). This means a frame is produced only while the camera is static in front of the pattern.

    The output frame is inverted: blinking pixels are black (0) and everything else is white (255), so that a
    blinking chessboard looks like a regular printed chessboard.
    """

    def __init__(self, width: int, height: int, config: BlinkingConfig):
        self.width = width
        self.height = height
        self.config = config
        self._pos = np.zeros(width * height, dtype=bool)
        self._neg = np.zeros(width * height, dtype=bool)
        self._window_start: Optional[int] = None

    def reset(self) -> None:
        self._pos[:] = False
        self._neg[:] = False
        self._window_start = None

    def process_events(self, evs: np.ndarray, idx: Optional[np.ndarray] = None,
                       pol: Optional[np.ndarray] = None) -> List[Tuple[int, Optional[np.ndarray]]]:
        """Processes a time-sorted buffer of events (structured array with fields x, y, p, t).

        ``idx`` (flat pixel indices y * width + x) and ``pol`` (bool, p > 0) can be passed if already computed.
        Returns a list of ``(window_end_ts, frame_or_None)``, one entry per completed accumulation window.
        """
        out: List[Tuple[int, Optional[np.ndarray]]] = []
        n = evs.size
        if n == 0:
            return out
        if idx is None:
            idx = evs["y"].astype(np.int64) * self.width + evs["x"].astype(np.int64)
        if pol is None:
            pol = evs["p"] > 0
        t = evs["t"]
        if self._window_start is None:
            self._window_start = int(t[0])

        start = 0
        while start < n:
            acc = max(1000, int(self.config.accumulation_time_us))
            win_end = self._window_start + acc
            stop = start + int(np.searchsorted(t[start:], win_end, side="left"))
            if stop > start:
                self._accumulate(idx[start:stop], pol[start:stop])
            if stop >= n:
                break
            # The current window is complete
            out.append((win_end, self._evaluate()))
            self._pos[:] = False
            self._neg[:] = False
            self._window_start = win_end
            next_t = int(t[stop])
            if next_t >= self._window_start + acc:  # Gap in the stream: jump directly to the window containing next_t
                self._window_start = next_t - (next_t - self._window_start) % acc
            start = stop
        return out

    def _accumulate(self, idx: np.ndarray, pol: np.ndarray) -> None:
        self._pos[idx[pol]] = True
        self._neg[idx[~pol]] = True

    def _evaluate(self) -> Optional[np.ndarray]:
        cfg = self.config
        both = self._pos & self._neg
        n_both = int(np.count_nonzero(both))
        if n_both < cfg.min_blinking_pixels or n_both == 0:
            return None
        n_on_only = int(np.count_nonzero(self._pos)) - n_both
        n_off_only = int(np.count_nonzero(self._neg)) - n_both
        if n_on_only > cfg.ratio_on * n_both or n_off_only > cfg.ratio_off * n_both:
            return None
        frame = np.where(both, np.uint8(0), np.uint8(255)).reshape(self.height, self.width)
        if cfg.median_blur_radius > 0:
            frame = cv2.medianBlur(frame, 2 * int(cfg.median_blur_radius) + 1)
        return frame


# ----------------------------------------------------------------------------------------------------------------------
# Chessboard detection
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class BlinkingWindow:
    ts: int  # Timestamp (us) of the end of the accumulation window
    frame: Optional[np.ndarray]  # uint8 HxW binary frame, None if no blinking pattern was found
    corners: Optional[np.ndarray]  # (cols*rows, 2) float32 corner positions, None if no chessboard was found


def points_on_grid(pts: np.ndarray, cols: int, rows: int, cos_max_angle: float = math.cos(math.pi / 9)) -> bool:
    """Checks that consecutive corners along rows and columns are roughly aligned (tolerates radial distortion).

    Equivalent of Metavision::are_points_on_grid_radial_distortion: the angle between two consecutive segments
    of a row (or column) must be below acos(cos_max_angle).
    """
    g = np.asarray(pts, dtype=np.float64).reshape(rows, cols, 2)

    def aligned(segments: np.ndarray) -> bool:  # segments: (lines, k, 2)
        if segments.shape[1] < 2:
            return True
        a, b = segments[:, :-1], segments[:, 1:]
        na, nb = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2)
        if np.any(na < 1e-6) or np.any(nb < 1e-6):
            return False
        cos = np.sum(a * b, axis=2) / (na * nb)
        return bool(np.all(cos > cos_max_angle))

    return aligned(np.diff(g, axis=1)) and aligned(np.diff(g, axis=0).transpose(1, 0, 2))


def find_chessboard_corners(frame: np.ndarray, cols: int, rows: int) -> Optional[np.ndarray]:
    """Finds the inner corners of a chessboard in a binary (blinking) frame. Returns (cols*rows, 2) float32 or None."""
    flags = cv2.CALIB_CB_FILTER_QUADS | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(frame, (cols, rows), flags=flags)
    if not found or corners is None or len(corners) != cols * rows:
        return None
    pts = corners.reshape(-1, 2)
    # Sub-pixel refinement with a window that cannot straddle neighbouring corners
    g = pts.reshape(rows, cols, 2)
    d_min = min(np.linalg.norm(np.diff(g, axis=1), axis=2).min(), np.linalg.norm(np.diff(g, axis=0), axis=2).min())
    half_win = int(max(2, min(11, d_min * 0.4)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.001)
    corners = cv2.cornerSubPix(frame, corners, (half_win, half_win), (-1, -1), criteria)
    pts = corners.reshape(-1, 2).astype(np.float32)
    if not points_on_grid(pts, cols, rows):
        return None
    return pts


class BlinkingChessboardDetector:
    """Detects a blinking chessboard in a stream of (trail-filtered) events."""

    def __init__(self, width: int, height: int, cols: int, rows: int, config: BlinkingConfig):
        self.width, self.height = width, height
        self.cols, self.rows = cols, rows
        self.generator = BlinkingFrameGenerator(width, height, config)

    def process_events(self, evs: np.ndarray, idx: Optional[np.ndarray] = None,
                       pol: Optional[np.ndarray] = None) -> List[BlinkingWindow]:
        results = []
        for ts, frame in self.generator.process_events(evs, idx, pol):
            corners = find_chessboard_corners(frame, self.cols, self.rows) if frame is not None else None
            results.append(BlinkingWindow(ts, frame, corners))
        return results


# ----------------------------------------------------------------------------------------------------------------------
# View collection
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class CalibrationView:
    corners: np.ndarray  # (N, 2) float32
    frame: Optional[np.ndarray]  # uint8 HxW binary frame used for the detection (needed for refinement)
    ts: int = 0


def mean_corner_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean corner displacement between two detections, robust to the 180deg ordering ambiguity of OpenCV."""
    d1 = np.linalg.norm(a - b, axis=1).mean()
    d2 = np.linalg.norm(a - b[::-1], axis=1).mean()
    return float(min(d1, d2))


class ViewCollector:
    """Decides whether a detection should be added to the calibration views."""

    def __init__(self, skip_time_us: int = 2_000_000, min_shift_px: float = 10.0):
        self.skip_time_us = skip_time_us
        self.min_shift_px = min_shift_px
        self._last_ts: Optional[int] = None

    def reset_timer(self) -> None:
        self._last_ts = None

    def accept(self, window: BlinkingWindow, existing: Sequence[np.ndarray]) -> Tuple[bool, str]:
        if window.corners is None:
            return False, "no chessboard"
        if self._last_ts is not None and window.ts - self._last_ts < self.skip_time_us:
            return False, "waiting (skip time)"
        if self.min_shift_px > 0 and existing:
            shift = min(mean_corner_shift(window.corners, c) for c in existing)
            if shift < self.min_shift_px:
                return False, "pose already captured - move the camera"
        self._last_ts = window.ts
        return True, "captured"


def coverage_map(corner_sets: Sequence[np.ndarray], width: int, height: int) -> np.ndarray:
    """Number of views covering each pixel (convex hull of the detected corners)."""
    acc = np.zeros((height, width), dtype=np.uint16)
    tmp = np.zeros((height, width), dtype=np.uint8)
    for c in corner_sets:
        add_to_coverage(acc, c, tmp)
    return acc


def add_to_coverage(acc: np.ndarray, corners: np.ndarray, tmp: Optional[np.ndarray] = None) -> None:
    if tmp is None:
        tmp = np.zeros(acc.shape, dtype=np.uint8)
    tmp[:] = 0
    hull = cv2.convexHull(np.round(corners).astype(np.int32))
    cv2.fillConvexPoly(tmp, hull, 1)
    acc += tmp


# ----------------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------------
def make_object_points(cols: int, rows: int, square_w: float, square_h: float) -> np.ndarray:
    """3D coordinates of the inner corners, in the same (row-major) order as cv2.findChessboardCorners."""
    xs, ys = np.meshgrid(np.arange(cols, dtype=np.float32) * square_w, np.arange(rows, dtype=np.float32) * square_h)
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(cols * rows, np.float32)], axis=1)


@dataclass
class CalibrationResult:
    model: str
    image_size: Tuple[int, int]  # (width, height)
    K: np.ndarray
    d: np.ndarray
    rms: float  # Overall RMS reprojection error (pixels) on the selected views
    selected: np.ndarray  # bool, one per view
    rvecs: List[np.ndarray]  # One per *selected* view
    tvecs: List[np.ndarray]
    per_view_errors: np.ndarray  # RMS error (pixels), one per *selected* view
    pts_2d: List[np.ndarray]  # 2D detections actually used (refined if refinement was enabled), one per view
    reprojected: List[np.ndarray]  # Reprojected corners, one per *selected* view
    refined: bool = False
    pattern: dict = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def n_views(self) -> int:
        return int(self.selected.size)

    @property
    def n_kept(self) -> int:
        return int(self.selected.sum())

    def view_error(self, i: int) -> Optional[float]:
        """RMS error of view i (index over all views), None if the view was rejected."""
        if not self.selected[i]:
            return None
        return float(self.per_view_errors[int(self.selected[:i].sum())])

    def view_reprojection(self, i: int) -> Optional[np.ndarray]:
        if not self.selected[i]:
            return None
        return self.reprojected[int(self.selected[:i].sum())]

    def summary(self) -> str:
        K, d = self.K, self.d.ravel()
        lines = [
            f"Model: {self.model}{'  (refined)' if self.refined else ''}",
            f"Image size: {self.image_size[0]} x {self.image_size[1]}",
            f"Views kept: {self.n_kept}/{self.n_views}",
            f"RMS reprojection error: {self.rms:.4f} px",
            "",
            f"fx = {K[0, 0]:.4f}",
            f"fy = {K[1, 1]:.4f}",
            f"cx = {K[0, 2]:.4f}",
            f"cy = {K[1, 2]:.4f}",
            "",
            "Camera matrix K:",
        ]
        lines += ["  [" + ", ".join(f"{v:12.4f}" for v in row) + "]" for row in K]
        names = ["k1", "k2", "k3", "k4"] if self.model == FISHEYE else ["k1", "k2", "p1", "p2", "k3"]
        lines.append("")
        lines.append("Distortion coefficients:")
        lines.append("  " + ", ".join(f"{n}={v:.6f}" for n, v in zip(names, d)))
        if len(d) > len(names):
            lines.append("  extra: " + ", ".join(f"{v:.6f}" for v in d[len(names):]))
        lines.append("")
        lines.append("Per-view RMS error (px):")
        for i in range(self.n_views):
            e = self.view_error(i)
            lines.append(f"  view {i + 1:3d}: " + ("REJECTED (outlier)" if e is None else f"{e:.4f}"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "camera_model": self.model,
            "image_size": {"width": int(self.image_size[0]), "height": int(self.image_size[1])},
            "camera_matrix": self.K.tolist(),
            "distortion_coefficients": self.d.ravel().tolist(),
            "fx": float(self.K[0, 0]),
            "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]),
            "cy": float(self.K[1, 2]),
            "overall_rms_reprojection_error": float(self.rms),
            "n_views": self.n_views,
            "n_kept_views": self.n_kept,
            "selected_views": [int(b) for b in self.selected],
            "per_view_rms_reprojection_errors": [float(e) for e in self.per_view_errors],
            "refined": bool(self.refined),
            "pattern": self.pattern,
            "rvecs": [np.asarray(r).ravel().tolist() for r in self.rvecs],
            "tvecs": [np.asarray(t).ravel().tolist() for t in self.tvecs],
        }


def _project(obj: np.ndarray, rvec, tvec, K, d, model: str) -> np.ndarray:
    if model == FISHEYE:
        proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, d)
    else:
        proj, _ = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K, d)
    return proj.reshape(-1, 2)


def _calibrate_once(obj: np.ndarray, pts: List[np.ndarray], size: Tuple[int, int], model: str, flags: int):
    if model == FISHEYE:
        objs = [obj.reshape(1, -1, 3).astype(np.float64) for _ in pts]
        imgs = [p.reshape(1, -1, 2).astype(np.float64) for p in pts]
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-8)
        rms, K, d, rvecs, tvecs = cv2.fisheye.calibrate(objs, imgs, size, np.zeros((3, 3)), np.zeros((4, 1)),
                                                        flags=flags, criteria=criteria)
    else:
        objs = [obj.astype(np.float32) for _ in pts]
        imgs = [p.reshape(-1, 1, 2).astype(np.float32) for p in pts]
        rms, K, d, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, np.eye(3), None, flags=flags)
    reproj = [_project(obj, r, t, K, d, model) for r, t in zip(rvecs, tvecs)]
    errs = np.array([math.sqrt(np.mean(np.sum((rp - p.reshape(-1, 2)) ** 2, axis=1))) for rp, p in zip(reproj, pts)])
    return float(rms), K, d, list(rvecs), list(tvecs), errs, reproj


def calibrate_views(obj: np.ndarray, pts_2d: Sequence[np.ndarray], image_size: Tuple[int, int], model: str,
                    flags: int, outlier_ths: float):
    """Calibrates, then removes the views whose error is above mean + outlier_ths * std and calibrates again.

    A negative ``outlier_ths`` keeps all the views.
    """
    n = len(pts_2d)
    selected = np.ones(n, dtype=bool)
    res = _calibrate_once(obj, list(pts_2d), image_size, model, flags)
    if outlier_ths >= 0 and n >= 4:
        errs = res[5]
        keep = errs <= errs.mean() + outlier_ths * errs.std()
        if not keep.all() and keep.sum() >= 3:
            selected = keep
            res = _calibrate_once(obj, [p for p, k in zip(pts_2d, keep) if k], image_size, model, flags)
    return res, selected


def _undistort_px(pts: np.ndarray, K: np.ndarray, d: np.ndarray) -> np.ndarray:
    src = pts.reshape(-1, 1, 2).astype(np.float64)
    if hasattr(cv2, "undistortPointsIter"):
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        out = cv2.undistortPointsIter(src, K, d, None, K, crit)
    else:
        out = cv2.undistortPoints(src, K, d, P=K)
    return out.reshape(-1, 2)


def _distort_px(undist_px: np.ndarray, K: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Maps undistorted pixel coordinates to distorted (raw image) pixel coordinates."""
    pts = undist_px.reshape(-1, 2).astype(np.float64)
    homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ np.linalg.inv(K).T
    homog /= homog[:, 2:3]
    out, _ = cv2.projectPoints(homog.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, d)
    return out.reshape(-1, 2)


def refine_view(img: np.ndarray, pts: np.ndarray, K: np.ndarray, d: np.ndarray, cols: int, rows: int,
                blur_radius: int = 5) -> np.ndarray:
    """Re-localizes the corners of one view in an undistorted fronto-parallel version of its image (pinhole only).

    Port of Pipeline::refine_calibration() of the metavision_mono_calibration sample.
    """
    h, w = img.shape[:2]
    n = cols * rows
    pts = pts.reshape(-1, 2).astype(np.float64)

    # 1) Homography between a canonical fronto-parallel view of the board and the undistorted image
    mx, my = 0.1 * w, 0.1 * h
    fronto_corners = np.array([[mx, my], [w - 1 - mx, my], [w - 1 - mx, h - 1 - my], [mx, h - 1 - my]])
    inner = pts[[0, cols - 1, n - 1, n - cols]]
    H, _ = cv2.findHomography(fronto_corners, _undistort_px(inner, K, d), 0)  # fronto -> undistorted image
    if H is None:
        raise RuntimeError("homography estimation failed")

    # 2) Maps that undistort and unproject the image to the fronto-parallel plane
    grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)), axis=-1)
    undist = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H)
    dist = _distort_px(undist, K, d)
    mapx = dist[:, 0].reshape(h, w).astype(np.float32)
    mapy = dist[:, 1].reshape(h, w).astype(np.float32)

    # 3) Fronto-parallel image (white background), blurred so that the corners can be localized with sub-pixel accuracy
    fronto = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    k = 2 * int(blur_radius) + 1
    fronto = cv2.GaussianBlur(fronto, (k, k), 2)

    # 4) Project the previous detections onto the fronto-parallel plane and refine them there
    kp = cv2.perspectiveTransform(_undistort_px(pts, K, d).reshape(-1, 1, 2), np.linalg.inv(H)).astype(np.float32)
    kp_init = kp.copy()
    flat = kp.reshape(-1, 2)
    dcol = float(np.linalg.norm(flat[0] - flat[1]))
    drow = float(np.linalg.norm(flat[0] - flat[cols]))
    rad_x = max(2, int(round(0.5 * (0.9 * dcol - 1))))
    rad_y = max(2, int(round(0.5 * (0.9 * drow - 1))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    kp = cv2.cornerSubPix(fronto, kp, (rad_x, rad_y), (1, 1), criteria)

    # Corners on the board border whose outer squares are (partly) outside the original image would be pulled by the
    # artificial white background with a large search window: refine those with a smaller window.
    edge_idx = _corners_close_to_edge(pts, cols, rows, w, h)
    if edge_idx:
        small = kp_init[edge_idx].copy()
        small = cv2.cornerSubPix(fronto, small, (min(11, rad_x), min(11, rad_y)), (1, 1), criteria)
        kp[edge_idx] = small

    # 5) Back to the distorted image
    undist_ref = cv2.perspectiveTransform(kp.astype(np.float64), H)
    return _distort_px(undist_ref, K, d).astype(np.float32)


def _corners_close_to_edge(pts: np.ndarray, cols: int, rows: int, w: int, h: int) -> List[int]:
    def valid(c: int, p: int) -> bool:
        q = pts[c] + 0.9 * (pts[c] - pts[p])
        return 0 <= q[0] < w and 0 <= q[1] < h

    tl, tr, br, bl = 0, cols - 1, rows * cols - 1, (rows - 1) * cols
    idx = set()
    if not valid(tl, tl + cols) or not valid(tr, tr + cols):
        idx.update(range(tl, tr + 1))
    if not valid(bl, bl - cols) or not valid(br, br - cols):
        idx.update(range(bl, br + 1))
    if not valid(tl, tl + 1) or not valid(bl, bl + 1):
        idx.update(range(tl, bl + 1, cols))
    if not valid(tr, tr - 1) or not valid(br, br - 1):
        idx.update(range(tr, br + 1, cols))
    return sorted(idx)


def run_calibration(pts_2d: Sequence[np.ndarray], image_size: Tuple[int, int], cols: int, rows: int,
                    square_w: float, square_h: float, model: str = PINHOLE, outlier_ths: float = 2.0,
                    refine: bool = False, frames: Optional[Sequence[Optional[np.ndarray]]] = None,
                    blur_radius: int = 5, log: Callable[[str], None] = print) -> CalibrationResult:
    """Full calibration: first pass (k3 fixed for pinhole), optional refinement pass (k3 free)."""
    t0 = time.time()
    if len(pts_2d) < 3:
        raise ValueError(f"At least 3 views are required (got {len(pts_2d)}).")
    obj = make_object_points(cols, rows, square_w, square_h)
    pts_2d = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in pts_2d]
    if model == FISHEYE:
        flags_first = flags_refine = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    else:
        flags_first = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3
        flags_refine = cv2.CALIB_FIX_ASPECT_RATIO

    log(f"Calibrating ({model}) with {len(pts_2d)} views...")
    (rms, K, d, rvecs, tvecs, errs, reproj), selected = calibrate_views(obj, pts_2d, image_size, model,
                                                                       flags_first, outlier_ths)
    log(f"First pass: RMS = {rms:.4f} px, kept {int(selected.sum())}/{len(pts_2d)} views")
    refined = False
    if refine:
        if model != PINHOLE:
            log("Refinement is only available for the pinhole model - skipped.")
        elif frames is None or any(f is None for f in frames):
            log("Refinement needs the pattern images of every view - skipped.")
        else:
            log("Refining corner detections on undistorted fronto-parallel views...")
            new_pts = []
            for i, (f, p) in enumerate(zip(frames, pts_2d)):
                try:
                    new_pts.append(refine_view(f, p, K, d, cols, rows, blur_radius))
                except Exception as e:  # Keep the original detection for this view
                    log(f"  view {i + 1}: refinement failed ({e}), keeping original corners")
                    new_pts.append(p)
            pts_2d = new_pts
            (rms, K, d, rvecs, tvecs, errs, reproj), selected = calibrate_views(obj, pts_2d, image_size, model,
                                                                               flags_refine, outlier_ths)
            refined = True
            log(f"Refined pass: RMS = {rms:.4f} px, kept {int(selected.sum())}/{len(pts_2d)} views")

    pattern = {"n_cols": cols, "n_rows": rows, "square_width": float(square_w), "square_height": float(square_h)}
    return CalibrationResult(model=model, image_size=tuple(image_size), K=K, d=d, rms=rms, selected=selected,
                             rvecs=rvecs, tvecs=tvecs, per_view_errors=errs, pts_2d=pts_2d, reprojected=reproj,
                             refined=refined, pattern=pattern, duration_s=time.time() - t0)


# ----------------------------------------------------------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------------------------------------------------------
def draw_view(frame: Optional[np.ndarray], pts: np.ndarray, cols: int, rows: int, image_size: Tuple[int, int],
              result: Optional[CalibrationResult] = None, view_idx: int = 0, magnification: int = 10) -> np.ndarray:
    """Image of a calibration view with its detected corners and (if available) magnified reprojection errors."""
    w, h = image_size
    if frame is None:
        img = np.full((h, w, 3), 255, np.uint8)
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    cv2.drawChessboardCorners(img, (cols, rows), pts.reshape(-1, 1, 2).astype(np.float32), True)
    if result is not None:
        if result.selected[view_idx]:
            img = (img * 0.6).astype(np.uint8)
            used = result.pts_2d[view_idx]
            rp = result.view_reprojection(view_idx)
            for p, q in zip(used, rp):
                tip = p + magnification * (q - p)
                cv2.arrowedLine(img, tuple(np.round(p).astype(int)), tuple(np.round(tip).astype(int)),
                                (0, 0, 255), 2, tipLength=0.3)
            text = f"RMS reprojection error: {result.view_error(view_idx):.3f} px  (arrows x{magnification})"
        else:
            red = np.zeros_like(img)
            red[:] = (0, 0, 255)
            img = cv2.addWeighted(img, 0.5, red, 0.4, 0)
            text = "REJECTED VIEW (outlier)"
        put_label(img, text, (10, 10))
    return img


def put_label(img: np.ndarray, text: str, org: Tuple[int, int], color=(255, 255, 255), bg=(0, 0, 0),
              scale: float = 0.8) -> None:
    """Draws text with a filled background box; org is the top-left corner of the box."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x, y), (x + tw + 12, y + th + base + 10), bg, -1)
    cv2.putText(img, text, (x + 6, y + th + 5), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------------------------------
DETECTIONS_FILE = "recorded_pattern.json"
IMAGES_DIR = "pattern_images"


def save_detections(directory: Path, image_size: Tuple[int, int], cols: int, rows: int, square_w: float,
                    square_h: float, views: Sequence[CalibrationView], camera_serial: str = "") -> Path:
    directory = Path(directory)
    img_dir = directory / IMAGES_DIR
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("pattern_*.png"):
        old.unlink()
    for i, v in enumerate(views):
        if v.frame is not None:
            cv2.imwrite(str(img_dir / f"pattern_{i + 1}.png"), v.frame)
    data = {
        "camera_serial": camera_serial or "",
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "pattern": {"n_cols": cols, "n_rows": rows, "square_width": float(square_w), "square_height": float(square_h)},
        "views": [{"ts_us": int(v.ts), "corners": np.asarray(v.corners).round(4).tolist()} for v in views],
    }
    path = directory / DETECTIONS_FILE
    path.write_text(json.dumps(data, indent=1))
    return path


def load_detections(directory: Path):
    """Returns (image_size, pattern dict, list of CalibrationView, camera serial or "")."""
    directory = Path(directory)
    data = json.loads((directory / DETECTIONS_FILE).read_text())
    image_size = (int(data["image_size"]["width"]), int(data["image_size"]["height"]))
    pattern = data["pattern"]
    n = int(pattern["n_cols"]) * int(pattern["n_rows"])
    views = []
    for i, v in enumerate(data["views"]):
        corners = np.asarray(v["corners"], dtype=np.float32).reshape(-1, 2)
        if len(corners) != n:
            raise ValueError(f"View {i + 1} has {len(corners)} corners instead of {n}")
        img_path = directory / IMAGES_DIR / f"pattern_{i + 1}.png"
        frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) if img_path.exists() else None
        views.append(CalibrationView(corners=corners, frame=frame, ts=int(v.get("ts_us", 0))))
    return image_size, pattern, views, str(data.get("camera_serial", ""))


def save_intrinsics(directory: Path, result: CalibrationResult, metadata: Optional[dict] = None,
                    name: str = "intrinsics") -> Tuple[Path, Path]:
    """Writes <name>.json (plain JSON) and <name>_opencv.yaml (readable with cv2.FileStorage)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = {"date": time.strftime("%Y-%m-%d %H:%M:%S")}
    data.update(metadata or {})
    data.update(result.to_dict())
    json_path = directory / f"{name}.json"
    json_path.write_text(json.dumps(data, indent=2))

    yaml_path = directory / f"{name}_opencv.yaml"
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_model", result.model)
    fs.write("image_width", int(result.image_size[0]))
    fs.write("image_height", int(result.image_size[1]))
    fs.write("camera_matrix", np.asarray(result.K, dtype=np.float64))
    fs.write("distortion_coefficients", np.asarray(result.d, dtype=np.float64).reshape(1, -1))
    fs.write("overall_rms_reprojection_error", float(result.rms))
    fs.write("n_kept_views", result.n_kept)
    fs.release()
    return json_path, yaml_path
