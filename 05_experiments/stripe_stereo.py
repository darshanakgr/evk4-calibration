"""
Laser-stripe stereo correspondence for two hardware-synchronised, rectified EVK4 cameras (experiment code).

The laser projects a (near-vertical) line that a stepper motor sweeps across the scene. Because the cameras share one
clock, the stripe seen by the left and the right camera at the same instant is the same physical curve, and because the
views are rectified, a point of that curve lies on the same row in both images. Three ways to turn this into
correspondences are implemented, all on the ON events (OFF events are slow and smeared):

* ``match_hough``     A. Time windows of length dt; in each window, the stripe of each camera is detected as a straight
                         line (Hough transform + least-squares refinement on its inliers); disparity per row
                         = x_L(y) - x_R(y) of the two lines.
* ``match_centroid``  B. Same time windows; in each window and row, the stripe position is the median x of the events of
                         that row; disparity = median_L - median_R.
* ``match_time``      C. Per event: for every left event (x_L, y, t), the right stripe position at that exact time is
                         interpolated from the right events of the same row with |t_R - t| <= tau (local linear fit
                         of x_R against t_R); disparity = x_L - x_R(t).

Before matching, the ON events of a scan go through a spatio-temporal neighbour filter (``neighbour_filter``: an event
is kept if >= k of its 8 neighbour pixels fired within +-delta). Stripe events pass, because the stripe is a continuous
line and neighbours along it fire almost together, whereas isolated events (background activity, scattered light) do
not. Then only the first surviving ON event of each pixel within the scan is used: it marks the arrival of the stripe.

Event coordinates are rectified with the step 2 calibration (stereo_core.compute_event_lut): x keeps its sub-pixel
value, y is rounded to the nearest rectified row.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _d in (ROOT / "01_camera_intrincis", ROOT / "02_stereo_rectification", ROOT / "05_experiments"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
import stereo_core as sc  # noqa: E402
import modulation_analysis as ma  # noqa: E402  (event loading, hot pixels, sweep detection)

REC_ROOT = ROOT / "04_dual_cam_recorder" / "recordings"
CALIB_FILE = ROOT / "02_stereo_rectification" / "output" / "stereo_00050496_00051788.json"
FOLDERS = {"filters OFF": "cw", "trail filter ON": "cw_with_trail_filter"}
DWELLS = (10, 20, 50, 100)
FILES = {"left": "left_00050496.h5", "right": "right_00051788.h5"}
D_RANGE = (20.0, 300.0)  # Accepted disparities (px): Z = f B / d = 0.28 .. 4.2 m


# ----------------------------------------------------------------------------------------------------------------------
# Calibration and data
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Geometry:
    f: float  # Rectified focal length (px)
    cx: float
    cy: float
    baseline: float  # m
    width: int
    height: int
    lut: Dict[str, np.ndarray]  # side -> (H, W, 2) rectified (x, y) of each raw pixel
    rect_rms_px: float  # Mean RMS vertical disparity of the rectified calibration corners

    def depth(self, d):
        return self.f * self.baseline / np.asarray(d, np.float64)

    def points(self, y, x, d) -> np.ndarray:
        """Rectified (row, x_left, disparity) -> 3D points (m) in the rectified left camera frame."""
        z = self.depth(d)
        return np.stack([(np.asarray(x) - self.cx) * z / self.f, (np.asarray(y) - self.cy) * z / self.f, z], -1)


def load_geometry(path=CALIB_FILE) -> Geometry:
    res = sc.load_stereo(path)
    assert res.horizontal, "expected a horizontal rig"
    b = -res.P2[0, 3] / res.P2[0, 0]
    assert b > 0, "expected x_left > x_right (left camera on the left)"
    lut = {s: sc.compute_event_lut(res, s) for s in ("left", "right")}
    w, h = res.image_size
    return Geometry(float(res.P1[0, 0]), float(res.P1[0, 2]), float(res.P1[1, 2]), float(b), w, h, lut,
                    float(res.rect_stats().get("mean_rms_px", np.nan)))


@dataclass
class Stream:
    """ON events of one camera, time-sorted, hot pixels removed, with rectified coordinates."""
    t: np.ndarray  # int64 µs since recording start
    pix: np.ndarray  # raw flat pixel index
    xr: np.ndarray  # rectified x (float32)
    row: np.ndarray  # rectified row (int16), -1 outside the image
    n_raw: int
    n_hot_removed: int
    n_off: int
    duration_us: int
    gaps: list  # [(t_us, length_us)] holes in the event stream (lost data)


@dataclass
class Pair:
    folder: str
    dwell_us: int
    left: Stream
    right: Stream


def load_stream(path, lut: np.ndarray) -> Stream:
    ev = ma.load_events(path)
    order = np.argsort(ev.t, kind="stable")  # A few files hold one or two out-of-order events at the start
    ev = ma.Events(ev.t[order], ev.x[order], ev.y[order], ev.p[order], ev.width, ev.height, ev.duration_us, ev.attrs)
    ev = ev.subset((ev.t >= 0) & (ev.t < ev.duration_us))
    hot, _ = ma.hot_pixels(ev)
    n_raw = len(ev)
    gaps = ma.stream_gaps(ev.t)
    ev = ma.remove_pixels(ev, hot)
    n_hot = n_raw - len(ev)
    n_off = int((ev.p == 0).sum())
    ev = ev.subset(ev.p == 1)
    xy = lut[ev.y, ev.x]
    row = np.rint(xy[:, 1])
    inside = np.isfinite(row) & (row >= 0) & (row < ev.height) & np.isfinite(xy[:, 0])
    row = np.where(inside, row, -1).astype(np.int16)
    return Stream(ev.t, ev.pix, xy[:, 0].astype(np.float32), row, n_raw, n_hot, n_off, ev.duration_us, gaps)


def load_pair(folder: str, dwell_us: int, geom: Geometry) -> Pair:
    d = REC_ROOT / folder / f"{dwell_us}us"
    return Pair(folder, dwell_us, load_stream(d / FILES["left"], geom.lut["left"]),
                load_stream(d / FILES["right"], geom.lut["right"]))


# ----------------------------------------------------------------------------------------------------------------------
# Scans
# ----------------------------------------------------------------------------------------------------------------------
def neighbour_filter(pix: np.ndarray, t: np.ndarray, width: int, height: int, delta_us: float,
                     min_neighbours: int = 2) -> np.ndarray:
    """Boolean mask: events with >= min_neighbours of their 8 neighbour pixels having an event within +-delta_us."""
    key = pix.astype(np.int64) * (1 << 34) + t  # t < 2^34 µs (~4.8 h)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    x, y = pix % width, pix // width
    count = np.zeros(pix.size, np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            valid = (x + dx >= 0) & (x + dx < width) & (y + dy >= 0) & (y + dy < height)
            npix = pix.astype(np.int64) + dy * width + dx
            lo = np.searchsorted(ks, npix * (1 << 34) + (t - delta_us).astype(np.int64), side="left")
            hi = np.searchsorted(ks, npix * (1 << 34) + (t + delta_us).astype(np.int64), side="right")
            count += (valid & (hi > lo)).astype(np.int16)
    return count >= min_neighbours


@dataclass
class Scan:
    t0: int  # µs
    t1: int
    direction: int  # +1: stripe moves to larger x, -1: to smaller x
    speed_px_ms: float  # Median stripe speed in the rectified left image
    idx: Dict[str, np.ndarray] = field(default_factory=dict)  # side -> indices of the first ON event of each pixel


def stripe_track(s: Stream, t0: int, t1: int, bin_us: int = 500, min_events: int = 200):
    """Median rectified x of the ON events per time bin (bins with >= min_events)."""
    i0, i1 = np.searchsorted(s.t, [t0, t1])
    t, x = s.t[i0:i1], s.xr[i0:i1]
    b = (t - t0) // bin_us
    order = np.lexsort((x, b))
    b, x = b[order], x[order]
    starts = np.r_[0, np.where(np.diff(b))[0] + 1]
    counts = np.diff(np.r_[starts, b.size])
    med = x[starts + counts // 2]
    keep = counts >= min_events
    return t0 + (b[starts][keep] + 0.5) * bin_us, med[keep]


def find_scans(p: Pair, pad_us: int = 5000, delta_us: Optional[float] = 300.0, min_neighbours: int = 2,
               width: int = 1280, height: int = 720) -> List[Scan]:
    """Laser scans = bursts of ON events in both cameras; direction and speed from the left stripe track.
    delta_us None disables the neighbour filter."""
    sw = ma.find_sweeps(p.left.t, p.left.duration_us, min_len_ms=3)
    out = []
    for t0, t1 in sw:
        t0, t1 = max(0, t0 - pad_us), min(p.left.duration_us, t1 + pad_us)
        tc, xc = stripe_track(p.left, t0, t1)
        if tc.size < 5:
            continue
        v = np.diff(xc) / np.diff(tc) * 1e3  # px/ms
        direction = int(np.sign(np.median(v)))
        speed = float(np.median(np.abs(v[np.sign(v) == direction])))
        sc_ = Scan(t0, t1, direction, speed)
        for side, s in (("left", p.left), ("right", p.right)):
            i0, i1 = np.searchsorted(s.t, [t0, t1])
            cand = np.arange(i0, i1)
            if delta_us is not None:
                cand = cand[neighbour_filter(s.pix[i0:i1], s.t[i0:i1], width, height, delta_us, min_neighbours)]
            _, first = np.unique(s.pix[cand], return_index=True)  # Sorted by t, so this is the first event
            first = cand[np.sort(first)]
            sc_.idx[side] = first[s.row[first] >= 0]
        out.append(sc_)
    return out


def scan_events(s: Stream, idx: np.ndarray):
    return s.t[idx], s.xr[idx].astype(np.float64), s.row[idx].astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# Matchers. Each returns a dict of 1-D arrays: row, x (left, rectified), d (disparity), t (µs)
# ----------------------------------------------------------------------------------------------------------------------
def _result(row, x, d, t):
    row, x, d, t = (np.asarray(a) for a in (row, x, d, t))
    ok = np.isfinite(d) & (d >= D_RANGE[0]) & (d <= D_RANGE[1])
    return {"row": row[ok].astype(np.int64), "x": x[ok].astype(np.float64), "d": d[ok].astype(np.float64),
            "t": t[ok].astype(np.float64)}


def _group_median(key: np.ndarray, x: np.ndarray):
    """Median of x per distinct key: (keys, medians, counts)."""
    order = np.lexsort((x, key))
    key, x = key[order], x[order]
    starts = np.r_[0, np.where(np.diff(key))[0] + 1]
    counts = np.diff(np.r_[starts, key.size])
    lo, hi = x[starts + (counts - 1) // 2], x[starts + counts // 2]
    return key[starts], 0.5 * (lo + hi), counts


def _compensate(x, t, w, scan: Scan, dt_us: float):
    """Moves each event to where the stripe was at the centre of its window: x - v (t - t_centre)."""
    tc = scan.t0 + (w + 0.5) * dt_us
    return x - scan.direction * scan.speed_px_ms * 1e-3 * (t - tc)


def match_centroid(p: Pair, scan: Scan, dt_us: float, min_events: int = 2, height: int = 720,
                   compensate: bool = False):
    """B. Per (time window, row): median x of the left and of the right events (optionally motion-compensated to the
    window centre with the scan speed)."""
    meds = {}
    for side, s in (("left", p.left), ("right", p.right)):
        t, x, row = scan_events(s, scan.idx[side])
        w = ((t - scan.t0) // dt_us).astype(np.int64)
        if compensate:
            x = _compensate(x, t, w, scan, dt_us)
        k, m, c = _group_median(w * height + row, x)
        meds[side] = (k[c >= min_events], m[c >= min_events])
    kl, ml = meds["left"]
    kr, mr = meds["right"]
    common, il, ir = np.intersect1d(kl, kr, assume_unique=True, return_indices=True)
    w, row = common // height, common % height
    return _result(row, ml[il], ml[il] - mr[ir], scan.t0 + (w + 0.5) * dt_us)


def hough_line(x: np.ndarray, y: np.ndarray, max_angle_deg: float = 20.0, angle_step_deg: float = 0.25,
               rho_bin: float = 1.0, inlier_px: float = 3.0, min_votes: int = 20):
    """Strongest near-vertical line through the points: Hough vote over x cos(a) + y sin(a) = rho, then a least-squares
    fit x = c0 + c1 y on the inliers. Returns (c0, c1, y_min, y_max, n_inliers) or None."""
    if x.size < min_votes:
        return None
    ang = np.deg2rad(np.arange(-max_angle_deg, max_angle_deg + 1e-9, angle_step_deg))
    rho = x[None, :] * np.cos(ang)[:, None] + y[None, :] * np.sin(ang)[:, None]
    r0 = np.floor(rho.min() / rho_bin)
    bins = (np.floor(rho / rho_bin) - r0).astype(np.int64)
    nb = int(bins.max()) + 1
    flat = (np.arange(ang.size)[:, None] * nb + bins).ravel()
    acc = np.bincount(flat, minlength=ang.size * nb).reshape(ang.size, nb)
    ia, ir = np.unravel_index(np.argmax(acc), acc.shape)
    if acc[ia, ir] < min_votes:
        return None
    rho_best = (ir + r0 + 0.5) * rho_bin
    inl = np.abs(rho[ia] - rho_best) <= inlier_px
    if inl.sum() < min_votes:
        return None
    c1, c0 = np.polyfit(y[inl], x[inl], 1)
    # One refinement pass on the fitted line
    inl = np.abs(x - (c0 + c1 * y)) <= inlier_px
    if inl.sum() < min_votes:
        return None
    c1, c0 = np.polyfit(y[inl], x[inl], 1)
    return c0, c1, int(y[inl].min()), int(y[inl].max()), int(inl.sum())


def match_hough(p: Pair, scan: Scan, dt_us: float, compensate: bool = False, **kw):
    """A. Per time window: one straight line per camera (Hough + LS refinement); disparity per row from the lines
    (optionally motion-compensated to the window centre with the scan speed)."""
    ev = {side: scan_events(s, scan.idx[side]) for side, s in (("left", p.left), ("right", p.right))}
    win = {side: ((e[0] - scan.t0) // dt_us).astype(np.int64) for side, e in ev.items()}
    if compensate:
        ev = {side: (t, _compensate(x, t, win[side], scan, dt_us), row) for side, (t, x, row) in ev.items()}
    n_win = int(max(win["left"].max(initial=0), win["right"].max(initial=0))) + 1
    bounds = {side: np.searchsorted(w, np.arange(n_win + 1)) for side, w in win.items()}  # w is sorted (t sorted)
    rows, xs, ds, ts = [], [], [], []
    for w in range(n_win):
        lines = {}
        for side, (t, x, row) in ev.items():
            a, b = bounds[side][w], bounds[side][w + 1]
            lines[side] = hough_line(x[a:b], row[a:b].astype(np.float64), **kw)
        if lines["left"] is None or lines["right"] is None:
            continue
        (a0, a1, ly0, ly1, _), (b0, b1, ry0, ry1, _) = lines["left"], lines["right"]
        y = np.arange(max(ly0, ry0), min(ly1, ry1) + 1)
        if y.size == 0:
            continue
        xl = a0 + a1 * y
        rows.append(y); xs.append(xl); ds.append(xl - (b0 + b1 * y)); ts.append(np.full(y.size, scan.t0 + (w + 0.5) * dt_us))
    if not rows:
        return _result([], [], [], [])
    return _result(np.concatenate(rows), np.concatenate(xs), np.concatenate(ds), np.concatenate(ts))


def local_fit(tq, rq, tr, xr, rr, tau_us: float, row_halfwidth: int = 0, min_events: int = 3):
    """Stripe position at query times/rows (tq, rq): local linear fit x = a + b (t - tq) over the reference events
    (tr, xr, rr) with |t - tq| <= tau and |row - rq| <= row_halfwidth. Returns (a, n, ok)."""
    tq = np.asarray(tq, np.float64)
    rq = np.asarray(rq, np.int64)
    # Sort the right events by (row, t): one global array, rows as contiguous segments
    o = np.lexsort((tr, rr))
    tr, xr, rr = tr[o], xr[o], rr[o]
    key_r = rr * 1e9 + tr  # Monotonic composite key (t < 1e9 µs)
    cs = [np.r_[0.0, np.cumsum(a)] for a in (np.ones_like(tr), tr, xr, tr * tr, tr * xr)]
    n = st = sx = stt = stx = 0.0
    for dr in range(-row_halfwidth, row_halfwidth + 1):
        kq = (rq + dr) * 1e9 + tq
        i0 = np.searchsorted(key_r, kq - tau_us, side="left")
        i1 = np.searchsorted(key_r, kq + tau_us, side="right")
        n, st, sx, stt, stx = (acc + c[i1] - c[i0] for acc, c in zip((n, st, sx, stt, stx), cs))
    # Centre on the query time: u = t - tq
    su = st - n * tq
    suu = stt - 2 * tq * st + n * tq * tq
    sux = stx - tq * sx
    den = n * suu - su * su
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (sx * suu - su * sux) / den
        ok = (n >= min_events) & (den > 1e-9 * np.maximum(n * suu, 1))
        mean_u = su / n
        var_u = suu / n - mean_u ** 2
    ok &= np.abs(mean_u) <= np.sqrt(np.maximum(var_u, 0))  # Query time inside the bulk of the events (no extrapolation)
    return a, n, ok


def match_time(p: Pair, scan: Scan, tau_us: float, row_halfwidth: int = 1, min_events: int = 3):
    """C. For every left event (x_L, row, t): right stripe x at the same time, from a local linear fit
    x_R = a + b (t_R - t) over the right events with |t_R - t| <= tau in rows row +- row_halfwidth; d = x_L - a."""
    tl, xl, rl = scan_events(p.left, scan.idx["left"])
    tr, xr, rr = scan_events(p.right, scan.idx["right"])
    tl = (tl - scan.t0).astype(np.float64)
    tr = (tr - scan.t0).astype(np.float64)
    a, _, ok = local_fit(tl, rl, tr, xr, rr, tau_us, row_halfwidth, min_events)
    return _result(rl, xl, np.where(ok, xl - a, np.nan), tl + scan.t0)


def event_scatter(s: Stream, scan: Scan, side: str, tau_us: float, row_halfwidth: int = 1) -> np.ndarray:
    """Scatter of single events around their own camera's local stripe fit, x - x_fit(t) in px (the event itself is
    included in the fit, so this slightly underestimates the scatter). Divided by the stripe speed it is the
    per-pixel timing jitter."""
    t, x, r = scan_events(s, scan.idx[side])
    t = (t - scan.t0).astype(np.float64)
    a, _, ok = local_fit(t, r, t, x, r, tau_us, row_halfwidth)
    return (x - a)[ok]


METHODS = {
    "A: window + Hough line": match_hough,
    "A': Hough, motion-compensated": lambda p, s, dt: match_hough(p, s, dt, compensate=True),
    "B: window + row median": match_centroid,
    "B': row median, motion-compensated": lambda p, s, dt: match_centroid(p, s, dt, compensate=True),
    "C: per-event time matching": match_time,
}


def run_method(name: str, p: Pair, scans: List[Scan], param: float) -> Tuple[List[dict], float]:
    """Runs one matcher on every scan; returns per-scan results and the mean run time per scan (s)."""
    fn = METHODS[name]
    out, t_total = [], 0.0
    for s in scans:
        t0 = time.perf_counter()
        out.append(fn(p, s, param))
        t_total += time.perf_counter() - t0
    return out, t_total / max(len(scans), 1)


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------------------
def fit_plane_ransac(pts: np.ndarray, thr: float = 0.01, iters: int = 400, rng=None):
    """Robust plane n.X = c (|n| = 1). Returns (n, c, inlier mask)."""
    rng = np.random.default_rng(0) if rng is None else rng
    best, best_n = None, -1
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        inl = np.abs(pts @ n - n @ s[0]) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    for _ in range(3):  # LS refinement on inliers
        q = pts[best]
        c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        best = np.abs((pts - c) @ n) < thr
    return n, float(n @ c), best


def raster(res: dict, cell: int = 8, width: int = 1280, height: int = 720):
    """Median disparity per (cell x cell) block; NaN where empty."""
    gx, gy = width // cell, height // cell
    cx = np.clip((res["x"] // cell).astype(np.int64), 0, gx - 1)
    cy = np.clip(res["row"] // cell, 0, gy - 1)
    k, m, _ = _group_median(cy * gx + cx, res["d"])
    out = np.full(gx * gy, np.nan)
    out[k] = m
    return out.reshape(gy, gx)


# ----------------------------------------------------------------------------------------------------------------------
# Full evaluation of one recording (cached)
# ----------------------------------------------------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "stripe"
EVAL_VERSION = 3
CELL = 8  # px, resolution at which the methods are also compared
OUTLIER_MM = 20.0

# Parameters are expressed as stripe motion in px, converted to µs with the scan speed of each recording:
# windows (A, B) of 16 px of stripe motion, C fits over +-8 px of stripe motion.
DEFAULT_PX = {"A: window + Hough line": 16, "A': Hough, motion-compensated": 16, "B: window + row median": 16,
              "B': row median, motion-compensated": 16, "C: per-event time matching": 8}


def lit_cells(p: Pair, scan: Scan, cell: int = CELL, min_events: int = 5, width: int = 1280, height: int = 720):
    """Cells (left rectified image) that the stripe lit in this scan: >= min_events first-ON events."""
    t, x, r = scan_events(p.left, scan.idx["left"])
    gx, gy = width // cell, height // cell
    k = np.clip(r // cell, 0, gy - 1) * gx + np.clip((x // cell).astype(np.int64), 0, gx - 1)
    return (np.bincount(k, minlength=gx * gy) >= min_events).reshape(gy, gx)


def evaluate(p: Pair, scans: List[Scan], geom: Geometry, name: str, param_px: float, keep_samples=(1,)) -> dict:
    """Runs one method on all scans and scores it against a robust plane fitted to its own samples (the target is a
    flat surface). Per-scan offsets from the plane are removed before measuring the scatter, and reported separately
    by scan direction."""
    v = float(np.median([s.speed_px_ms for s in scans]))
    param_us = param_px / v * 1e3
    out, t_scan = run_method(name, p, scans, param_us)
    pts = [geom.points(r["row"], r["x"], r["d"]) for r in out]
    allp = np.concatenate(pts)
    step = max(1, len(allp) // 200000)
    n, c, _ = fit_plane_ransac(allp[::step], thr=OUTLIER_MM / 1e3)
    if n[2] < 0:
        n, c = -n, -c
    gx, gy = 1280 // CELL, 720 // CELL
    per_scan, samples = [], {}
    res_all, cell_res_all = [], []
    cell_sum = np.zeros(gx * gy)
    cell_cnt = np.zeros(gx * gy)
    for i, (s, r, q) in enumerate(zip(scans, out, pts)):
        resid = (q @ n - c) * 1e3  # mm, signed distance to the plane
        off = float(np.median(resid)) if resid.size else np.nan
        e = resid - off
        res_all.append(e)
        k = np.clip(r["row"] // CELL, 0, gy - 1) * gx + np.clip((r["x"] // CELL).astype(np.int64), 0, gx - 1)
        covered = np.zeros(gx * gy, bool)
        if e.size:
            kk, med, _ = _group_median(k, e)
            cell_res_all.append(med)
            cell_sum[kk] += med
            cell_cnt[kk] += 1
            covered[kk] = True
        lit = lit_cells(p, s).ravel()
        per_scan.append({"direction": s.direction, "offset_mm": off, "n": int(e.size),
                         "mad_mm": float(1.4826 * np.median(np.abs(e))) if e.size else np.nan,
                         "coverage": float((covered & lit).sum() / max(lit.sum(), 1))})
        if i in keep_samples:
            samples[i] = {k_: r[k_].astype(np.float32) for k_ in ("row", "x", "d")}
            samples[i]["resid"] = e.astype(np.float32)
    e = np.concatenate(res_all) if res_all else np.zeros(0)
    ce = np.concatenate(cell_res_all) if cell_res_all else np.zeros(0)
    z = float(np.median(allp[:, 2]))
    d_med = float(np.median(np.concatenate([r["d"] for r in out])))
    mm_per_px = z / d_med * 1e3
    offs = {d: [x["offset_mm"] for x in per_scan if x["direction"] == d] for d in (-1, 1)}
    bias = 0.5 * (np.mean(offs[1]) - np.mean(offs[-1])) if offs[1] and offs[-1] else np.nan
    with np.errstate(invalid="ignore"):
        cell_mean = np.where(cell_cnt > 0, cell_sum / np.maximum(cell_cnt, 1), np.nan).reshape(gy, gx)
    return {"method": name, "param_px": param_px, "param_us": param_us, "t_scan_s": t_scan, "z_m": z, "d_px": d_med,
            "mm_per_px": mm_per_px, "plane": (n, c), "per_scan": per_scan,
            "n_per_scan": float(np.mean([x["n"] for x in per_scan])),
            "mad_mm": float(1.4826 * np.median(np.abs(e))) if e.size else np.nan,
            "outliers": float(np.mean(np.abs(e) > OUTLIER_MM)) if e.size else np.nan,
            "cell_mad_mm": float(1.4826 * np.median(np.abs(ce))) if ce.size else np.nan,
            "coverage": float(np.mean([x["coverage"] for x in per_scan])),
            "dir_bias_mm": float(bias), "dir_bias_px": float(bias / mm_per_px),
            "cell_mean_resid": cell_mean, "samples": samples}


def recording_summary(p: Pair, scans: List[Scan]) -> dict:
    v = float(np.median([s.speed_px_ms for s in scans]))
    scat = {side: np.concatenate([event_scatter(st, s, side, 8 / v * 1e3) for s in scans[:3]])
            for side, st in (("left", p.left), ("right", p.right))}
    return {"folder": p.folder, "dwell_us": p.dwell_us, "n_scans": len(scans), "speed_px_ms": v,
            "cross_ms": 1280 / v, "duration_s": p.left.duration_us / 1e6,
            "first_on": {side: float(np.mean([s.idx[side].size for s in scans])) for side in ("left", "right")},
            "scatter_px": {k: float(1.4826 * np.median(np.abs(e))) for k, e in scat.items()},
            "scatter_us": {k: float(1.4826 * np.median(np.abs(e))) / v * 1e3 for k, e in scat.items()},
            "hot_removed": {"left": p.left.n_hot_removed, "right": p.right.n_hot_removed},
            "n_raw": {"left": p.left.n_raw, "right": p.right.n_raw},
            "gaps": {"left": p.left.gaps, "right": p.right.gaps},
            "directions": [s.direction for s in scans]}


def cached(tag: str, fn, use_cache: bool = True):
    import pickle
    path = CACHE_DIR / f"{tag}_v{EVAL_VERSION}.pkl"
    if use_cache and path.exists():
        return pickle.loads(path.read_bytes())
    out = fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


_PAIR_CACHE: dict = {}


def get_pair(folder: str, dwell_us: int, geom: Geometry):
    """Loaded pair + scans, kept in memory for the last recording only (they take a few hundred MB)."""
    key = (folder, dwell_us)
    if key not in _PAIR_CACHE:
        _PAIR_CACHE.clear()
        p = load_pair(folder, dwell_us, geom)
        _PAIR_CACHE[key] = (p, find_scans(p))
    return _PAIR_CACHE[key]


def evaluate_recording(folder: str, dwell_us: int, geom: Geometry, use_cache: bool = True) -> dict:
    """Summary + evaluation of every method with its default parameter, for one recording."""
    def run():
        p, scans = get_pair(folder, dwell_us, geom)
        return {"summary": recording_summary(p, scans),
                "default": {name: evaluate(p, scans, geom, name, px) for name, px in DEFAULT_PX.items()}}

    return cached(f"{folder}_{dwell_us}us", run, use_cache)


def parameter_sweep(folder: str, dwell_us: int, geom: Geometry, sweeps: Dict[str, List[float]],
                    use_cache: bool = True) -> Dict[str, List[dict]]:
    """Each method evaluated for several parameter values (px of stripe motion)."""
    def run():
        p, scans = get_pair(folder, dwell_us, geom)
        return {name: [evaluate(p, scans, geom, name, v, keep_samples=()) for v in values]
                for name, values in sweeps.items()}

    return cached(f"sweep_{folder}_{dwell_us}us", run, use_cache)


def filter_ablation(folder: str, dwell_us: int, geom: Geometry, use_cache: bool = True) -> Dict[str, dict]:
    """Every method with its default parameter, without the neighbour filter (first ON event of each pixel)."""
    def run():
        p, _ = get_pair(folder, dwell_us, geom)
        scans = find_scans(p, delta_us=None)
        return {name: evaluate(p, scans, geom, name, px, keep_samples=()) for name, px in DEFAULT_PX.items()}

    return cached(f"nofilter_{folder}_{dwell_us}us", run, use_cache)


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic object: tests the methods on a non-planar scene
# ----------------------------------------------------------------------------------------------------------------------
def plane_disparity(plane, geom: Geometry, row, x):
    """Disparity of the plane n.X = c at left rectified pixel (row, x)."""
    n, c = plane
    ray = np.stack([(np.asarray(x, np.float64) - geom.cx) / geom.f, (np.asarray(row, np.float64) - geom.cy) / geom.f,
                    np.ones(np.shape(x))], -1)
    return geom.f * geom.baseline * (ray @ n) / c


def with_object(p: Pair, scans: List[Scan], plane, geom: Geometry, shape: str, amp_px: float = 15.0) -> Tuple[Pair, callable]:
    """Copy of the pair whose right events are shifted by -delta(row, x_L) px, as if part of the surface were closer
    by delta px of disparity (timing unchanged). Returns the pair and delta(row, x_left).

    shape "box": delta = amp inside rows 250-450 and x_L 450-850 (a box in front of the wall: depth steps),
    shape "ridge": delta = amp * exp(-((row - 360) / 100)^2) for every x (a horizontal cylinder-like bulge: the
    stripe of the right camera becomes curved within each window)."""
    if shape == "box":
        def delta(row, x):
            return amp_px * ((row >= 250) & (row < 450) & (x >= 450) & (x < 850))
    else:
        def delta(row, x):
            return amp_px * np.exp(-((np.asarray(row, np.float64) - 360.0) / 100.0) ** 2) * np.ones(np.shape(x))
    r = p.right
    row = r.row.astype(np.float64)
    # Left x of each right event, from the plane: x_L = x_R + d_plane (one fixed-point iteration is plenty)
    xl = r.xr + 150.0
    for _ in range(3):
        xl = r.xr + plane_disparity(plane, geom, row, xl)
    xl_obj = xl
    for _ in range(3):  # the object moves the matching left x by delta as well
        xl_obj = r.xr + plane_disparity(plane, geom, row, xl_obj) + delta(row, xl_obj)
    shifted = Stream(r.t, r.pix, (r.xr - delta(row, xl_obj)).astype(np.float32), r.row, r.n_raw, r.n_hot_removed,
                     r.n_off, r.duration_us, r.gaps)
    return Pair(p.folder, p.dwell_us, p.left, shifted), delta


def synthetic_test(folder: str, dwell_us: int, geom: Geometry, shapes=("box", "ridge"), amp_px: float = 15.0,
                   use_cache: bool = True) -> dict:
    """Error of every method against the known synthetic disparity: |d - (d_plane + delta)| per sample."""
    def run():
        p, scans = get_pair(folder, dwell_us, geom)
        base = evaluate(p, scans, geom, "A': Hough, motion-compensated", DEFAULT_PX["A': Hough, motion-compensated"],
                        keep_samples=())
        plane = base["plane"]
        out = {}
        v = float(np.median([s.speed_px_ms for s in scans]))
        for shape in shapes:
            ps, delta = with_object(p, scans, plane, geom, shape, amp_px)
            out[shape] = {}
            for name, px in DEFAULT_PX.items():
                res, _ = run_method(name, ps, scans, px / v * 1e3)
                errs, inside, samples, cell_err, cell_in = [], [], None, [], []
                gx = 1280 // CELL
                for i, r in enumerate(res):
                    truth_delta = delta(r["row"], r["x"])
                    err = r["d"] - plane_disparity(plane, geom, r["row"], r["x"]) - truth_delta
                    errs.append(err)
                    inside.append(truth_delta > 0.5 * amp_px)
                    if err.size:
                        k = (r["row"] // CELL) * gx + np.clip((r["x"] // CELL).astype(np.int64), 0, gx - 1)
                        kk, med, _ = _group_median(k, err)
                        cell_err.append(med)
                        cy, cx = kk // gx, kk % gx
                        cell_in.append(delta(cy * CELL + CELL / 2, cx * CELL + CELL / 2.0) > 0.5 * amp_px)
                    if i == 1:
                        samples = {"row": r["row"].astype(np.int16), "x": r["x"].astype(np.float32),
                                   "err": err.astype(np.float32),
                                   "d_rel": (r["d"] - plane_disparity(plane, geom, r["row"], r["x"])).astype(np.float32)}
                err, ins = np.concatenate(errs), np.concatenate(inside)
                ce, ci = np.concatenate(cell_err), np.concatenate(cell_in)
                out[shape][name] = {
                    "mae_px": float(np.median(np.abs(err))),
                    "mae_obj_px": float(np.median(np.abs(err[ins]))) if ins.any() else np.nan,
                    "bad_cells": float(np.mean(np.abs(ce) > 3.0)),
                    "bad_cells_obj": float(np.mean(np.abs(ce[ci]) > 3.0)) if ci.any() else np.nan,
                    "bad_cells_bg": float(np.mean(np.abs(ce[~ci]) > 3.0)) if (~ci).any() else np.nan,
                    "samples": samples}
        return {"plane": plane, "amp_px": amp_px, "results": out}

    return cached(f"synthetic_{folder}_{dwell_us}us", run, use_cache)
