"""
UI-independent core of the event-based stereo depth tool (step 3).

Pipeline (per camera, in its own process):
    raw event (x, y) --rectification index LUT--> rectified pixel of a downscaled image
                     --> TimeSurface (last timestamp per pixel) --> exp(-(t_now - t_last) / tau) image (uint8)
Then, in the UI process, for each pair of simultaneous left/right images:
    StereoSGBM --> disparity d (px) --> depth Z = f * B / d   (f: rectified focal at the processing scale, B: baseline)
    kept only where events occurred recently (time surface above a threshold).

Calibration: stereo_<left>_<right>.json from step 2 (02_stereo_rectification), loaded with stereo_core.load_stereo.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _d in (ROOT / "01_camera_intrincis", ROOT / "02_stereo_rectification"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import stereo_core as sc  # noqa: E402  (step 2: calibration I/O, rectification of points)

TS_NEVER = np.int64(-(1 << 60))  # "No event yet" timestamp


# ----------------------------------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------------------------------
def scaled_size(image_size: Tuple[int, int], scale: float) -> Tuple[int, int]:
    return max(1, int(round(image_size[0] * scale))), max(1, int(round(image_size[1] * scale)))


@dataclass
class DepthGeometry:
    """Rectified geometry of the left (reference) camera at the processing scale."""
    width: int
    height: int
    f: float  # Rectified focal length (px, processing scale)
    cx: float
    cy: float
    baseline: float  # m
    horizontal: bool  # Horizontal rig: disparity along x
    flip: bool  # True if the disparity is negative in the native orientation (cameras swapped w.r.t. OpenCV)

    @classmethod
    def from_stereo(cls, res: sc.StereoResult, scale: float) -> "DepthGeometry":
        w, h = scaled_size(res.image_size, scale)
        horizontal = res.horizontal
        P1, P2 = res.P1, res.P2
        if horizontal:
            b = -P2[0, 3] / P2[0, 0]
        else:
            b = -P2[1, 3] / P2[1, 1]
        sx, sy = w / res.image_size[0], h / res.image_size[1]
        return cls(width=w, height=h, f=float(P1[0, 0] * sx), cx=float((P1[0, 2] + 0.5) * sx - 0.5),
                   cy=float((P1[1, 2] + 0.5) * sy - 0.5), baseline=float(abs(b)), horizontal=bool(horizontal),
                   flip=bool(b < 0))

    def depth_for_disparity(self, d: float) -> float:
        return self.f * self.baseline / d if d > 0 else float("inf")


def make_index_lut(res: Optional[sc.StereoResult], side: str, image_size: Tuple[int, int], scale: float) -> np.ndarray:
    """int32 array of size W*H (raw sensor): flat index of the rectified pixel at the processing scale, -1 if outside.

    Without calibration (res None) the LUT only downscales (unrectified preview).
    """
    if res is None:
        w, h = image_size
        coords = np.stack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)), axis=-1)
    else:
        coords = sc.compute_event_lut(res, side)  # Rectified coordinates (full resolution) of every raw pixel
    return index_lut_from_coords(coords, image_size, scale)


def index_lut_from_coords(coords: np.ndarray, image_size: Tuple[int, int], scale: float) -> np.ndarray:
    """Same as make_index_lut, from precomputed (H, W, 2) full-resolution rectified coordinates."""
    w, h = image_size
    ws, hs = scaled_size(image_size, scale)
    xs = np.round((coords[..., 0] + 0.5) * (ws / w) - 0.5)
    ys = np.round((coords[..., 1] + 0.5) * (hs / h) - 0.5)
    valid = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (xs < ws) & (ys >= 0) & (ys < hs)
    idx = np.full(w * h, -1, np.int32)
    idx[valid.ravel()] = (ys[valid] * ws + xs[valid]).astype(np.int32)
    return idx


# ----------------------------------------------------------------------------------------------------------------------
# Event representation
# ----------------------------------------------------------------------------------------------------------------------
class TimeSurface:
    """Last event timestamp of each (rectified, downscaled) pixel, rendered as exp(-(t_now - t_last) / tau)."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.last = np.full(width * height, TS_NEVER, np.int64)

    def reset(self) -> None:
        self.last[:] = TS_NEVER

    def add(self, rect_idx: np.ndarray, t: np.ndarray) -> None:
        """rect_idx: rectified flat index of each event (-1 = outside), t: timestamps (sorted)."""
        ok = rect_idx >= 0
        self.last[rect_idx[ok]] = t[ok]

    def render(self, t_now: int, tau_us: float) -> np.ndarray:
        dt = (t_now - self.last).astype(np.float32)
        img = np.exp(-np.maximum(dt, 0) / max(1.0, float(tau_us))) * 255.0
        return img.astype(np.uint8).reshape(self.height, self.width)


class EventTimeFramer:
    """Cuts an event stream into frames at fixed event-time boundaries (multiples of 1/fps).

    With hardware-synchronized cameras (shared time base), both cameras produce frames with identical keys covering
    exactly the same events, whatever the size of the buffers they receive.
    """

    def __init__(self, fps: float):
        self.period = int(round(1e6 / max(1.0, fps)))
        self.next_boundary: Optional[int] = None

    def set_fps(self, fps: float) -> None:
        self.period = int(round(1e6 / max(1.0, fps)))
        self.next_boundary = None

    def process(self, ts: TimeSurface, rect_idx: Optional[np.ndarray], t: np.ndarray, t_last: int,
                tau_us: float) -> list:
        """Adds a buffer of events (rect_idx, t sorted) to ``ts``; returns [(boundary_us, image), ...]."""
        frames = []
        if self.next_boundary is None or t_last - self.next_boundary > 10 * self.period:  # Start or long gap
            self.next_boundary = (t_last // self.period + 1) * self.period
        start = 0
        while t_last >= self.next_boundary:
            k = int(np.searchsorted(t, self.next_boundary, side="left"))  # Events strictly before the boundary
            if k > start:
                ts.add(rect_idx[start:k], t[start:k])
            start = k
            frames.append((self.next_boundary, ts.render(self.next_boundary, tau_us)))
            self.next_boundary += self.period
        if rect_idx is not None and start < len(t):
            ts.add(rect_idx[start:], t[start:])
        return frames


# ----------------------------------------------------------------------------------------------------------------------
# Stereo matching
# ----------------------------------------------------------------------------------------------------------------------
ALGORITHMS = ("SGBM 3-way", "SGBM", "SGBM HH", "BM")


@dataclass
class MatcherParams:
    algorithm: str = "SGBM 3-way"
    num_disparities: int = 64  # Multiple of 16. Min. measurable depth = f * B / num_disparities
    min_disparity: int = 0
    block_size: int = 7  # Odd
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 31
    activity_threshold: int = 20  # Min. time surface value (0-255) of the left pixel to output a depth
    mask_to_events: bool = True
    median_ksize: int = 3  # 0 (off), 3 or 5

    def key(self) -> tuple:
        return (self.algorithm, self.num_disparities, self.min_disparity, self.block_size, self.uniqueness_ratio,
                self.speckle_window, self.speckle_range, self.disp12_max_diff, self.pre_filter_cap)


@dataclass
class DepthFrame:
    disparity: np.ndarray  # float32 (h, w), NaN where invalid
    depth: np.ndarray  # float32 (h, w) meters, NaN where invalid
    valid_ratio: float  # Fraction of valid pixels among the active ones
    match_ms: float


class DepthEstimator:
    def __init__(self, geometry: DepthGeometry, params: Optional[MatcherParams] = None):
        self.geometry = geometry
        self.params = params or MatcherParams()
        self._matcher = None
        self._key = None

    def _get_matcher(self):
        p = self.params
        if self._matcher is None or self._key != p.key():
            nd = max(16, int(round(p.num_disparities / 16)) * 16)
            bs = max(3, int(p.block_size) | 1)
            if p.algorithm == "BM":
                m = cv2.StereoBM_create(numDisparities=nd, blockSize=max(5, bs))
                m.setMinDisparity(p.min_disparity)
                m.setUniquenessRatio(p.uniqueness_ratio)
                m.setSpeckleWindowSize(p.speckle_window)
                m.setSpeckleRange(p.speckle_range)
                m.setDisp12MaxDiff(p.disp12_max_diff)
                m.setPreFilterCap(max(1, min(63, p.pre_filter_cap)))
            else:
                mode = {"SGBM": cv2.STEREO_SGBM_MODE_SGBM, "SGBM HH": cv2.STEREO_SGBM_MODE_HH}.get(
                    p.algorithm, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                m = cv2.StereoSGBM_create(minDisparity=p.min_disparity, numDisparities=nd, blockSize=bs,
                                          P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=p.disp12_max_diff,
                                          preFilterCap=p.pre_filter_cap, uniquenessRatio=p.uniqueness_ratio,
                                          speckleWindowSize=p.speckle_window, speckleRange=p.speckle_range,
                                          mode=mode)
            self._matcher, self._key = m, p.key()
        return self._matcher

    def _to_matching_frame(self, img: np.ndarray) -> np.ndarray:
        """Orientation in which the epipolar lines are horizontal and the disparities positive."""
        if not self.geometry.horizontal:
            img = img.T
        if self.geometry.flip:
            img = img[:, ::-1]
        return np.ascontiguousarray(img)

    def _from_matching_frame(self, img: np.ndarray) -> np.ndarray:
        if self.geometry.flip:
            img = img[:, ::-1]
        if not self.geometry.horizontal:
            img = img.T
        return np.ascontiguousarray(img)

    def compute(self, left: np.ndarray, right: np.ndarray) -> DepthFrame:
        """left/right: rectified uint8 images of identical size (time surfaces)."""
        t0 = time.perf_counter()
        p = self.params
        raw = self._from_matching_frame(
            self._get_matcher().compute(self._to_matching_frame(left), self._to_matching_frame(right)))
        disp = raw.astype(np.float32) / 16.0
        valid = (raw >= p.min_disparity * 16) & (disp > 0)  # OpenCV marks invalid pixels with (min_disp - 1) * 16
        if p.median_ksize in (3, 5):
            filtered = cv2.medianBlur(np.where(valid, disp, 0).astype(np.float32), p.median_ksize)
            disp = np.where(valid, filtered, disp)
        active = left >= p.activity_threshold
        if p.mask_to_events:
            valid &= active
        disp = np.where(valid, disp, np.nan).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (self.geometry.f * self.geometry.baseline / disp).astype(np.float32)
        n_active = int(np.count_nonzero(active))
        ratio = float(np.count_nonzero(valid & active)) / n_active if n_active else 0.0
        return DepthFrame(disp, depth, ratio, (time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------------------------------------------------------------------
# Visualization and export
# ----------------------------------------------------------------------------------------------------------------------
def depth_range_auto(depth: np.ndarray, lo_pct: float = 5, hi_pct: float = 95) -> Optional[Tuple[float, float]]:
    d = depth[np.isfinite(depth)]
    if d.size < 50:
        return None
    lo, hi = np.percentile(d, [lo_pct, hi_pct])
    return float(lo), float(max(hi, lo * 1.05))


def colorize_depth(depth: np.ndarray, z_min: float, z_max: float, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """BGR image, near = red, far = blue (scaled in inverse depth), invalid = black."""
    valid = np.isfinite(depth)
    inv = np.zeros(depth.shape, np.float32)
    inv[valid] = 1.0 / np.clip(depth[valid], z_min, z_max)
    lo, hi = 1.0 / z_max, 1.0 / z_min
    v = np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), colormap)
    img[~valid] = 0
    return img


def overlay_depth(gray: np.ndarray, depth_bgr: np.ndarray, valid: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = base.copy()
    out[valid] = (base[valid] * (1 - alpha) + depth_bgr[valid] * alpha).astype(np.uint8)
    return out


def colorbar(z_min: float, z_max: float, width: int = 400, height: int = 36, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    v = np.linspace(0, 1, width, dtype=np.float32)[::-1]  # Left = near (red), right = far (blue), inverse depth scale
    bar = cv2.applyColorMap((np.tile(v, (height, 1)) * 255).astype(np.uint8), colormap)
    cv2.putText(bar, f"{z_min:.2f} m", (4, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    txt = f"{z_max:.2f} m"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(bar, txt, (width - tw - 4, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def point_cloud(depth: np.ndarray, geom: DepthGeometry, colors: Optional[np.ndarray] = None):
    """3D points (left rectified camera frame, meters) of the valid pixels, with optional per-pixel BGR colors."""
    ys, xs = np.nonzero(np.isfinite(depth))
    z = depth[ys, xs].astype(np.float64)
    pts = np.stack([(xs - geom.cx) * z / geom.f, (ys - geom.cy) * z / geom.f, z], axis=1)
    rgb = colors[ys, xs][:, ::-1] if colors is not None else None
    return pts, rgb


def save_ply(path, pts: np.ndarray, rgb: Optional[np.ndarray] = None) -> Path:
    """Binary little-endian PLY (readable by MeshLab, CloudCompare, Open3D)."""
    path = Path(path)
    n = len(pts)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("red", "u1"), ("green", "u1"), ("blue", "u1")]
                                                           if rgb is not None else [])
    data = np.empty(n, dtype=fields)
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if rgb is not None:
        data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}", "property float x",
              "property float y", "property float z"]
    if rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())
    return path


def save_depth_png(path, depth: np.ndarray) -> Path:
    """16-bit PNG in millimeters (0 = invalid)."""
    mm = np.where(np.isfinite(depth), np.clip(depth * 1000.0, 0, 65535), 0).astype(np.uint16)
    cv2.imwrite(str(path), mm)
    return Path(path)


def load_depth_png(path) -> np.ndarray:
    mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return np.where(mm > 0, mm / 1000.0, np.nan).astype(np.float32)


class DepthRecorder:
    """Writes a depth sequence: depth/NNNNNN.png (16-bit mm), left|right/NNNNNN.png (time surfaces), frames.csv."""

    def __init__(self, directory, metadata: dict):
        self.dir = Path(directory)
        for sub in ("depth", "left", "right"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        (self.dir / "params.json").write_text(json.dumps(metadata, indent=2))
        self._csv_file = open(self.dir / "frames.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame", "host_time_s", "left_t_us", "right_t_us", "sync_offset_ms", "valid_ratio"])
        self.n = 0

    def write(self, host_time: float, t_left: int, t_right: int, sync_ms: float, left: np.ndarray, right: np.ndarray,
              frame: DepthFrame) -> None:
        name = f"{self.n:06d}.png"
        save_depth_png(self.dir / "depth" / name, frame.depth)
        cv2.imwrite(str(self.dir / "left" / name), left)
        cv2.imwrite(str(self.dir / "right" / name), right)
        self._csv.writerow([self.n, f"{host_time:.6f}", t_left, t_right, f"{sync_ms:.2f}", f"{frame.valid_ratio:.4f}"])
        self.n += 1

    def close(self) -> None:
        self._csv_file.close()


def depth_metadata(calib_path: str, res: sc.StereoResult, geom: DepthGeometry, scale: float, params: MatcherParams,
                   tau_ms: float, fps: float) -> dict:
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_file": str(calib_path),
        "serial_left": res.serial_left, "serial_right": res.serial_right,
        "processing_scale": scale,
        "geometry": asdict(geom),
        "depth_png": "uint16 millimeters, 0 = invalid; pixel (u, v) -> X = (u - cx) Z / f, Y = (v - cy) Z / f",
        "matcher": asdict(params),
        "time_surface_tau_ms": tau_ms,
        "frame_rate_hz": fps,
    }
