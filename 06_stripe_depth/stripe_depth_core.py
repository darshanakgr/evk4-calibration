"""
UI-independent core of step 6: depth from a swept laser line seen by two hardware-synchronised event cameras
("method C": per-event time matching).

Pipeline for one sweep (one pass of the laser line across the scene):

1. Sweep segmentation (``detect_sweeps``): 1 ms bins of the left camera whose event count is far above the background
   (median + n_sigma robust sigma), merged and padded. Uses only the recording's ms index: instantaneous.
2. Stripe arrival (``first_events``): ON events of the sweep; an event is kept only if >= k of its 8 neighbour pixels
   fired within +-delta (the stripe is a continuous line, noise is isolated), and only the first kept event of each
   pixel is used: it marks when the stripe reached that pixel.
3. Rectification: every event gets its rectified (x, row) from the step 2 calibration; x stays sub-pixel, the row is
   rounded.
4. Matching (``match_time``): for every left event (x_L, row, t), the right stripe position at that exact time is
   interpolated by a local linear fit x_R = a + b (t_R - t) over the right events of rows row +- h with
   |t_R - t| <= tau (tau given in px of stripe motion, converted with the sweep speed). d = x_L - a. The same instant
   selects the same laser plane, the same row selects the same epipolar plane: the match is unique.
5. Depth image (``robust_aggregate``): disparities rasterised on the rectified left grid, then an edge-preserving robust
   filter: median of the k x k window, mean of the values within +-max_dev of it, kept if >= min_support agree.
6. Z = f B / d; point cloud in the rectified left camera frame (x right, y down, z forward, metres).

The calibration sides are assigned to the recording files by camera serial number, so the result is correct even if
the roles were swapped when recording.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import recorder_core as rc  # Also puts the step 1-3 folders on sys.path

ROLES = ("left", "right")


# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class SweepParams:
    n_sigma: float = 6.0  # Activity threshold: median + n_sigma robust sigma of the 1 ms event counts
    merge_gap_ms: int = 20  # Active bins closer than this belong to the same sweep
    min_len_ms: int = 3  # Shorter bursts are ignored
    pad_ms: int = 5  # Margin added on both sides of each sweep


@dataclass
class DepthParams:
    tau_px: float = 8.0  # Matching window: +-tau of stripe motion (px), converted to us with the sweep speed
    row_halfwidth: int = 1  # Right events of rows row-h .. row+h are used
    min_events: int = 3  # Minimum right events for the local fit
    nb_delta_us: float = 300.0  # Neighbour filter: time window
    nb_min: int = 2  # Neighbour filter: minimum active neighbour pixels (0 disables the filter)
    agg_k: int = 7  # Aggregation window (odd, px)
    agg_max_dev_px: float = 1.5  # Values within +-max_dev of the window median are averaged
    agg_min_support: int = 8  # Minimum agreeing values for a depth pixel
    d_min: float = 20.0  # Accepted disparity range (px)
    d_max: float = 300.0


# ----------------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class StereoRig:
    f: float  # Rectified focal length (px)
    cx: float
    cy: float
    baseline: float  # m
    width: int
    height: int
    lut: Dict[str, np.ndarray]  # Calibration side -> (H, W, 2) float32 rectified (x, y) of each raw pixel
    serial: Dict[str, str] = field(default_factory=dict)  # Calibration side -> serial
    source: str = ""

    def depth(self, d):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.f * self.baseline / np.asarray(d, np.float64)

    def points(self, row, x, d) -> np.ndarray:
        """Rectified (row, x_left, disparity) -> (N, 3) points (m) in the rectified left camera frame."""
        z = self.depth(d)
        return np.stack([(np.asarray(x, np.float64) - self.cx) * z / self.f,
                         (np.asarray(row, np.float64) - self.cy) * z / self.f, z], -1)

    def side_of(self, serial: str, role: str) -> str:
        """Calibration side of a camera: by serial number, else its recording role."""
        for side, s in self.serial.items():
            if serial and s == serial:
                return side
        return role

    def describe(self) -> str:
        z1 = self.f * self.baseline / 100.0
        return (f"f = {self.f:.1f} px, baseline = {self.baseline * 1e3:.1f} mm, {self.width}x{self.height}\n"
                f"Z = {self.f * self.baseline:.1f} / d (m): d = 100 px -> {z1:.3f} m, "
                f"1 px of disparity there = {z1 / 100 * 1e3:.1f} mm")


def load_rig(stereo_json) -> StereoRig:
    import stereo_core as sc  # Step 2
    res = sc.load_stereo(stereo_json)
    if not res.horizontal:
        raise ValueError("This calibration is a vertical rig: method C expects a horizontal (left/right) rig.")
    b = -res.P2[0, 3] / res.P2[0, 0]
    if b <= 0:
        raise ValueError("Unexpected calibration: the 'left' camera is not on the left (P2[0, 3] >= 0).")
    w, h = res.image_size
    lut = {s: sc.compute_event_lut(res, s).astype(np.float32) for s in ROLES}
    return StereoRig(float(res.P1[0, 0]), float(res.P1[0, 2]), float(res.P1[1, 2]), float(b), int(w), int(h), lut,
                     {"left": str(res.serial_left), "right": str(res.serial_right)}, str(stereo_json))


# ----------------------------------------------------------------------------------------------------------------------
# 1. Sweep segmentation
# ----------------------------------------------------------------------------------------------------------------------
def on_counts_per_ms(rd: rc.EventFileReader, block: int = 4_000_000) -> np.ndarray:
    """Number of ON events per 1 ms bin from t_start (OFF events are excluded: they trail the stripe by tens of ms and
    would form a separate burst after each sweep). ~0.2 s per 10 M events."""
    n_ms = max(1, len(rd.ms_idx) - 1)
    c = np.zeros(n_ms, np.int64)
    for i in range(0, rd.n, block):
        t = rd.ds["t"][i:i + block]
        p = rd.ds["p"][i:i + block]
        k = np.clip((t[p > 0] - rd.t_start) // 1000, 0, n_ms - 1)
        c += np.bincount(k, minlength=n_ms)
    return c


def detect_sweeps(counts_per_ms: np.ndarray, t_start_us: int, p: SweepParams = SweepParams(),
                  t_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """Sweeps [(t0_us, t1_us)] (absolute times) from the ON event counts per ms of the left camera
    (on_counts_per_ms). t_range: optional absolute [t0, t1) to search in."""
    c = np.asarray(counts_per_ms, np.float64)
    if c.size == 0:
        return []
    med = np.median(c)
    sigma = max(1.4826 * np.median(np.abs(c - med)), np.sqrt(max(med, 1.0)))
    active = c > med + p.n_sigma * sigma + 10
    if t_range is not None:
        k = np.arange(c.size)
        active &= (t_start_us + (k + 1) * 1000 > t_range[0]) & (t_start_us + k * 1000 < t_range[1])
    act = np.where(active)[0]
    if act.size == 0:
        return []
    brk = np.where(np.diff(act) > p.merge_gap_ms)[0]
    starts, ends = np.r_[act[0], act[brk + 1]], np.r_[act[brk], act[-1]] + 1
    t_end = t_start_us + c.size * 1000
    out = []
    for s, e in zip(starts, ends):
        if e - s >= p.min_len_ms:
            out.append((max(t_start_us, t_start_us + (int(s) - p.pad_ms) * 1000),
                        min(t_end, t_start_us + (int(e) + p.pad_ms) * 1000)))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 2-3. Events of a sweep, stripe arrival, rectification
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class SweepEvents:
    """First (neighbour-filtered) ON event of each pixel within a sweep, rectified. Sorted by t."""
    t: np.ndarray  # int64 us (absolute)
    x: np.ndarray  # float64 rectified x
    row: np.ndarray  # int64 rectified row
    n_on: int = 0  # ON events read
    n_first: int = 0  # Pixels with a first event (before dropping those outside the rectified image)


def first_events(pix: np.ndarray, t: np.ndarray, width: int, height: int, delta_us: float, min_neighbours: int,
                 max_iter: int = 32) -> np.ndarray:
    """Index (into pix/t) of the first event of each pixel that has >= min_neighbours of its 8 neighbour pixels active
    within +-delta_us (neighbours counted over all events). Result sorted by time.

    Equivalent to filtering every event and then taking the first survivor per pixel, but walks the events of each
    pixel in time order and stops at the first one that passes, so only a few candidates per pixel are tested.
    Pixels needing more than max_iter steps (very noisy pixels) are dropped."""
    n = pix.size
    if n == 0:
        return np.zeros(0, np.int64)
    pix = pix.astype(np.int64)
    t = t.astype(np.int64)
    t_rel = t - t.min()
    key = pix * (1 << 34) + t_rel  # t_rel < 2^34 us (~4.8 h)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    pix_s = pix[order]
    starts = np.r_[0, np.where(np.diff(pix_s))[0] + 1]
    ends = np.r_[starts[1:], n]
    if min_neighbours <= 0:
        sel = order[starts]
        return np.sort(sel)
    cur = starts.copy()  # Candidate position (in sorted order) per pixel
    end = ends.copy()
    done = np.zeros(0, np.int64)
    found = []
    for _ in range(max_iter):
        if cur.size == 0:
            break
        p_c = pix_s[cur]
        t_c = t_rel[order[cur]]
        x, y = p_c % width, p_c // width
        cnt = np.zeros(cur.size, np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                valid = (x + dx >= 0) & (x + dx < width) & (y + dy >= 0) & (y + dy < height)
                npix = (p_c + dy * width + dx) * (1 << 34)
                lo = np.searchsorted(ks, npix + (t_c - int(delta_us)), side="left")
                hi = np.searchsorted(ks, npix + (t_c + int(delta_us)), side="right")
                cnt += (valid & (hi > lo)).astype(np.int16)
        ok = cnt >= min_neighbours
        found.append(order[cur[ok]])
        nxt = cur[~ok] + 1
        alive = nxt < end[~ok]
        cur, end = nxt[alive], end[~ok][alive]
    sel = np.concatenate(found) if found else done
    return np.sort(sel)


def read_sweep(rd: rc.EventFileReader, t0: int, t1: int, lut: np.ndarray, p: DepthParams) -> SweepEvents:
    """ON events of [t0, t1) -> first neighbour-filtered event per pixel, with rectified coordinates."""
    ev = rd.events_between(int(t0), int(t1))
    on = ev["p"] > 0
    t = ev["t"][on].astype(np.int64)
    x = ev["x"][on].astype(np.int64)
    y = ev["y"][on].astype(np.int64)
    if t.size > 1 and np.any(np.diff(t) < 0):  # A few files hold out-of-order events
        o = np.argsort(t, kind="stable")
        t, x, y = t[o], x[o], y[o]
    w, h = rd.width, rd.height
    sel = first_events(y * w + x, t, w, h, p.nb_delta_us, p.nb_min)
    xy = lut[y[sel], x[sel]]
    row = np.rint(xy[:, 1])
    inside = np.isfinite(row) & (row >= 0) & (row < h) & np.isfinite(xy[:, 0])
    sel_in = sel[inside]
    return SweepEvents(t[sel_in], xy[inside, 0].astype(np.float64), row[inside].astype(np.int64), int(t.size),
                       int(sel.size))


def stripe_motion(ev: SweepEvents, bin_us: int = 500, min_events: int = 50) -> Tuple[int, float]:
    """(direction +1 = towards larger x / -1, median speed in px/ms) from the median x per time bin."""
    if ev.t.size < 10:
        return 0, float("nan")
    b = (ev.t - ev.t[0]) // bin_us
    order = np.lexsort((ev.x, b))
    bs, xs = b[order], ev.x[order]
    starts = np.r_[0, np.where(np.diff(bs))[0] + 1]
    counts = np.diff(np.r_[starts, bs.size])
    keep = counts >= min_events
    if keep.sum() < 3:
        return 0, float("nan")
    tc = (bs[starts][keep] + 0.5) * bin_us
    xc = xs[starts + counts // 2][keep]
    v = np.diff(xc) / np.diff(tc) * 1e3
    direction = int(np.sign(np.median(v)))
    if direction == 0:
        return 0, float("nan")
    return direction, float(np.median(np.abs(v[np.sign(v) == direction])))


# ----------------------------------------------------------------------------------------------------------------------
# 4. Matching
# ----------------------------------------------------------------------------------------------------------------------
def local_fit(tq, rq, tr, xr, rr, tau_us: float, row_halfwidth: int = 1, min_events: int = 3):
    """Stripe position at query times/rows (tq, rq): least-squares line x = a + b (t - tq) through the reference events
    (tr, xr, rr) with |t - tq| <= tau and |row - rq| <= row_halfwidth. Returns (a, n, ok); ok requires n >= min_events,
    a non-degenerate time spread, and tq inside mean +- std of the window's times (interpolation, not extrapolation).

    Vectorised: the reference events are sorted by (row, t); prefix sums of 1, t, x, t^2, t x give the window sums with
    two binary searches per row."""
    tq = np.asarray(tq, np.float64)
    rq = np.asarray(rq, np.int64)
    tr, xr, rr = np.asarray(tr, np.float64), np.asarray(xr, np.float64), np.asarray(rr, np.int64)
    o = np.lexsort((tr, rr))
    tr, xr, rr = tr[o], xr[o], rr[o]
    key_r = rr * 1e9 + tr  # Monotonic composite key (t < 1e9 us relative to the sweep start)
    cs = [np.r_[0.0, np.cumsum(a)] for a in (np.ones_like(tr), tr, xr, tr * tr, tr * xr)]
    n = st = sx = stt = stx = 0.0
    for dr in range(-row_halfwidth, row_halfwidth + 1):
        kq = (rq + dr) * 1e9 + tq
        i0 = np.searchsorted(key_r, kq - tau_us, side="left")
        i1 = np.searchsorted(key_r, kq + tau_us, side="right")
        n, st, sx, stt, stx = (acc + c[i1] - c[i0] for acc, c in zip((n, st, sx, stt, stx), cs))
    su = st - n * tq  # Centred on the query time: u = t - tq
    suu = stt - 2 * tq * st + n * tq * tq
    sux = stx - tq * sx
    den = n * suu - su * su
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (sx * suu - su * sux) / den
        ok = (n >= min_events) & (den > 1e-9 * np.maximum(n * suu, 1))
        mean_u = su / n
        var_u = suu / n - mean_u ** 2
        ok &= np.abs(mean_u) <= np.sqrt(np.maximum(var_u, 0))
    return a, n, ok


def match_time(left: SweepEvents, right: SweepEvents, tau_us: float, p: DepthParams):
    """Method C: disparity of every left event. Returns (row, x, d) of the accepted matches."""
    if left.t.size == 0 or right.t.size == 0:
        z = np.zeros(0)
        return z.astype(np.int64), z, z
    t_ref = min(left.t[0], right.t[0])
    a, _, ok = local_fit(left.t - t_ref, left.row, right.t - t_ref, right.x, right.row, tau_us, p.row_halfwidth,
                         p.min_events)
    d = left.x - a
    ok &= np.isfinite(d) & (d >= p.d_min) & (d <= p.d_max)
    return left.row[ok], left.x[ok], d[ok]


# ----------------------------------------------------------------------------------------------------------------------
# 5. Depth image
# ----------------------------------------------------------------------------------------------------------------------
def rasterise(row, x, val, width: int, height: int) -> np.ndarray:
    """float32 (H, W): mean of val per rectified pixel (round(x), row); NaN where empty."""
    xi = np.rint(np.asarray(x)).astype(np.int64)
    row = np.asarray(row, np.int64)
    ok = (xi >= 0) & (xi < width) & (row >= 0) & (row < height)
    k = row[ok] * width + xi[ok]
    n = np.bincount(k, minlength=width * height)
    s = np.bincount(k, np.asarray(val, np.float64)[ok], minlength=width * height)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return img.reshape(height, width).astype(np.float32)


def robust_aggregate(img: np.ndarray, k: int = 7, max_dev: float = 1.5, min_support: int = 8):
    """Edge-preserving robust smoothing of a sparse disparity image: median m of the valid values of the k x k window,
    mean of the values within +-max_dev of m, output where >= min_support agree (also at empty pixels: fills small
    holes). Returns (out float32, support int16)."""
    h, w = img.shape
    k = max(1, int(k) | 1)  # Odd
    r = k // 2
    out = np.full((h, w), np.nan, np.float32)
    support = np.zeros((h, w), np.int16)
    pad = np.pad(img.astype(np.float32), r, constant_values=np.nan)
    valid = np.isfinite(pad)
    cnt = np.lib.stride_tricks.sliding_window_view(valid, (k, k)).sum((-1, -2))
    yy, xx = np.nonzero(cnt >= max(1, min_support))
    if yy.size == 0:
        return out, support
    view = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
    for i0 in range(0, yy.size, 400_000):  # Blocks: bounded memory
        y, x = yy[i0:i0 + 400_000], xx[i0:i0 + 400_000]
        win = view[y, x].reshape(y.size, k * k)
        srt = np.sort(win, 1)  # NaN last
        nv = np.isfinite(srt).sum(1)
        ar = np.arange(nv.size)
        med = 0.5 * (srt[ar, (nv - 1) // 2] + srt[ar, nv // 2])
        with np.errstate(invalid="ignore"):
            m = np.abs(win - med[:, None]) <= max_dev
        sup = m.sum(1)
        mean = np.where(m, win, 0).sum(1) / np.maximum(sup, 1)
        good = sup >= min_support
        out[y[good], x[good]] = mean[good]
        support[y, x] = sup
    return out, support


# ----------------------------------------------------------------------------------------------------------------------
# One sweep -> depth image + point cloud
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class SweepResult:
    t0: int
    t1: int
    direction: int
    speed_px_ms: float
    tau_us: float
    arrival_ms: Dict[str, np.ndarray]  # Calibration side -> float32 (H, W) stripe arrival time (ms from t0), NaN = none
    raw_disp: np.ndarray  # float32 (H, W) disparity per left event (before aggregation)
    disp: np.ndarray  # float32 (H, W) aggregated disparity
    support: np.ndarray  # int16 (H, W)
    stats: dict

    @property
    def depth(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return (self.stats["fB"] / self.disp).astype(np.float32)


def process_sweep(readers: Dict[str, rc.EventFileReader], rig: StereoRig, t0: int, t1: int,
                  p: DepthParams = DepthParams(), log: Callable[[str], None] = lambda s: None) -> SweepResult:
    """Depth of one sweep. readers: recording role -> reader; calibration sides are assigned by serial."""
    tim = {}
    side_reader = {}
    for role, rd in readers.items():
        side_reader[rig.side_of(rd.serial, role)] = rd
    if set(side_reader) != set(ROLES):
        raise ValueError("Could not assign the two files to the left/right calibration cameras.")
    for rd in readers.values():
        if (rd.width, rd.height) != (rig.width, rig.height):
            raise ValueError(f"Calibration is for {rig.width}x{rig.height}, recording is {rd.width}x{rd.height}.")
    ev = {}
    for side in ROLES:
        t = time.perf_counter()
        ev[side] = read_sweep(side_reader[side], t0, t1, rig.lut[side], p)
        tim[f"read_filter_{side}_s"] = time.perf_counter() - t
    direction, speed = stripe_motion(ev["left"])
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("No moving stripe found in this sweep (too few events in the left camera).")
    tau_us = p.tau_px / speed * 1e3
    t = time.perf_counter()
    row, x, d = match_time(ev["left"], ev["right"], tau_us, p)
    tim["match_s"] = time.perf_counter() - t
    t = time.perf_counter()
    w, h = rig.width, rig.height
    raw = rasterise(row, x, d, w, h)
    disp, support = robust_aggregate(raw, p.agg_k, p.agg_max_dev_px, p.agg_min_support)
    tim["aggregate_s"] = time.perf_counter() - t
    arrival = {s: rasterise(e.row, e.x, (e.t - t0) / 1e3, w, h) for s, e in ev.items()}
    fB = rig.f * rig.baseline
    ok = np.isfinite(disp)
    z = fB / disp[ok] if ok.any() else np.zeros(0)
    stats = {"t0_us": int(t0), "t1_us": int(t1), "direction": direction, "speed_px_ms": speed,
             "cross_ms": rig.width / speed, "tau_us": tau_us, "fB": fB,
             "on_events": {s: e.n_on for s, e in ev.items()}, "first_events": {s: int(e.t.size) for s, e in ev.items()},
             "matched": int(d.size), "depth_px": int(ok.sum()),
             "z_median_m": float(np.median(z)) if z.size else float("nan"),
             "z_p1_p99_m": [float(v) for v in np.percentile(z, [1, 99])] if z.size else [float("nan")] * 2,
             "timing_s": tim, "total_s": float(sum(tim.values())), "params": asdict(p)}
    return SweepResult(int(t0), int(t1), direction, speed, tau_us, arrival, raw, disp, support, stats)


def cloud_from_result(res: SweepResult, rig_like: dict) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """(N, 3) points (m) of the aggregated depth image and their (rows, cols). rig_like: dict with f, cx, cy, fB."""
    yy, xx = np.nonzero(np.isfinite(res.disp))
    d = res.disp[yy, xx].astype(np.float64)
    z = rig_like["fB"] / d
    pts = np.stack([(xx - rig_like["cx"]) * z / rig_like["f"], (yy - rig_like["cy"]) * z / rig_like["f"], z], -1)
    return pts, (yy, xx)


def fit_plane(pts: np.ndarray, thr: float = 0.01, iters: int = 300, seed: int = 0):
    """Robust plane (RANSAC + least-squares refinement). Returns (normal, distance, signed residuals in m); the normal
    points away from the camera. For checking a flat target."""
    rng = np.random.default_rng(seed)
    sub = pts[:: max(1, len(pts) // 100_000)]
    best, best_n = None, -1
    for _ in range(iters):
        s = sub[rng.choice(len(sub), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        nrm /= nn
        inl = np.abs((sub - s[0]) @ nrm) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    for _ in range(3):
        q = sub[best]
        c = q.mean(0)
        nrm = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        best = np.abs((sub - c) @ nrm) < thr
    if nrm[2] < 0:
        nrm = -nrm
    e = (pts - c) @ nrm
    return nrm, float(nrm @ c), e


def plane_summary(pts: np.ndarray) -> dict:
    if len(pts) < 100:
        return {}
    nrm, dist, e = fit_plane(pts)
    e_mm = (e - np.median(e)) * 1e3
    return {"normal": nrm.tolist(), "distance_m": dist, "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(nrm[2]))))),
            "mad_mm": float(1.4826 * np.median(np.abs(e_mm))), "within_10mm": float(np.mean(np.abs(e_mm) < 10)),
            "residual_mm": e_mm.astype(np.float32)}


# ----------------------------------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------------------------------
def write_ply(path, pts: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    """Binary little-endian PLY with float xyz (and uchar rgb)."""
    n = len(pts)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if rgb is not None
                                                          else [])
    rec = np.zeros(n, dtype=fields)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}", "property float x", "property float y",
            "property float z"]
    if rgb is not None:
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    with open(path, "wb") as f:
        f.write(("\n".join(head + ["end_header"]) + "\n").encode("ascii"))
        f.write(rec.tobytes())


def read_ply_xyz(path) -> np.ndarray:
    """Points of a PLY written by write_ply (for tests and quick checks)."""
    with open(path, "rb") as f:
        head = []
        while True:
            line = f.readline().decode("ascii").strip()
            head.append(line)
            if line == "end_header":
                break
        n = int(next(h.split()[2] for h in head if h.startswith("element vertex")))
        has_rgb = any("red" in h for h in head)
        dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if has_rgb else [])
        rec = np.frombuffer(f.read(), dtype=dt, count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], -1)


def write_png(path, img: np.ndarray) -> None:
    """PNG writer that also works with non-ASCII Windows paths (cv2.imwrite does not)."""
    import cv2
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"PNG encoding failed for {path}")
    buf.tofile(str(path))


def write_depth_png16(path, depth_m: np.ndarray) -> None:
    """16-bit PNG of the depth in millimetres (0 = no depth, values clipped to 1..65535 mm)."""
    mm = np.zeros(depth_m.shape, np.uint16)
    ok = np.isfinite(depth_m) & (depth_m > 0)
    mm[ok] = np.clip(np.rint(depth_m[ok] * 1e3), 1, 65535).astype(np.uint16)
    write_png(path, mm)


def read_depth_png16(path) -> np.ndarray:
    """Depth in metres (NaN = no depth) from a PNG written by write_depth_png16."""
    import cv2
    mm = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    return np.where(mm > 0, mm.astype(np.float64) / 1e3, np.nan)


def depth_colors(z: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    """(N, 3) uint8 RGB: viridis, near = yellow, far = purple."""
    import cv2
    u = np.clip((z_hi - z) / max(z_hi - z_lo, 1e-9), 0, 1)
    bgr = cv2.applyColorMap((u * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    return bgr[:, ::-1].copy()


def save_result(res: SweepResult, rig_like: dict, stem: Path) -> Dict[str, str]:
    """<stem>_depth.npz (images, depth in m, stats), <stem>_cloud.ply (coloured by depth), <stem>_info.json."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    pts, _ = cloud_from_result(res, rig_like)
    files = {"npz": f"{stem}_depth.npz", "ply": f"{stem}_cloud.ply", "json": f"{stem}_info.json"}
    np.savez_compressed(files["npz"], arrival_left=res.arrival_ms["left"], arrival_right=res.arrival_ms["right"],
                        raw_disp=res.raw_disp, disp=res.disp, support=res.support, depth_m=res.depth,
                        stats=json.dumps(res.stats), rig=json.dumps(rig_like))
    if len(pts):
        lo, hi = np.percentile(pts[:, 2], [1, 99])
        write_ply(files["ply"], pts, depth_colors(pts[:, 2], lo, hi))
    else:
        write_ply(files["ply"], np.zeros((0, 3)))
    Path(files["json"]).write_text(json.dumps(res.stats, indent=1))
    return files


def load_result(npz_path) -> Tuple[SweepResult, dict]:
    with np.load(npz_path) as z:
        stats = json.loads(str(z["stats"]))
        rig_like = json.loads(str(z["rig"]))
        res = SweepResult(stats["t0_us"], stats["t1_us"], stats["direction"], stats["speed_px_ms"], stats["tau_us"],
                          {"left": z["arrival_left"], "right": z["arrival_right"]}, z["raw_disp"], z["disp"],
                          z["support"], stats)
    return res, rig_like


def rig_dict(rig: StereoRig) -> dict:
    return {"f": rig.f, "cx": rig.cx, "cy": rig.cy, "baseline": rig.baseline, "fB": rig.f * rig.baseline,
            "width": rig.width, "height": rig.height, "source": rig.source}


def params_key(p: DepthParams, stereo_json: str) -> str:
    return json.dumps({"params": asdict(p), "calib": str(stereo_json)}, sort_keys=True)


# ----------------------------------------------------------------------------------------------------------------------
# Worker process
# ----------------------------------------------------------------------------------------------------------------------
def run_depth_job(rec_files: Dict[str, str], stereo_json: str, sweeps: Sequence[Tuple[int, int, int, str]],
                  params: dict, msg_q, cancel_evt) -> None:
    """Process entry point. sweeps: [(index, t0_us, t1_us, output stem)]. Messages:
    ("log", text), ("progress", done, total, text), ("sweep_done", index, files, stats), ("sweep_error", index, text),
    ("done", n_ok), ("cancelled",), ("error", text)."""
    readers = {}
    try:
        p = DepthParams(**params)
        msg_q.put(("progress", 0, len(sweeps), "loading the calibration"))
        rig = load_rig(stereo_json)
        rd_like = rig_dict(rig)
        readers = {r: rc.EventFileReader(f) for r, f in rec_files.items()}
        n_ok = 0
        for i, (k, t0, t1, stem) in enumerate(sweeps):
            if cancel_evt is not None and cancel_evt.is_set():
                msg_q.put(("cancelled",))
                return
            msg_q.put(("progress", i, len(sweeps), f"sweep #{k}"))
            try:
                res = process_sweep(readers, rig, t0, t1, p)
                res.stats["calib"] = str(stereo_json)  # With the parameters: identifies the result for reuse
                files = save_result(res, rd_like, Path(stem))
                n_ok += 1
                msg_q.put(("sweep_done", k, files, {kk: v for kk, v in res.stats.items() if kk != "params"}))
            except Exception as e:
                msg_q.put(("sweep_error", k, str(e)))
        msg_q.put(("progress", len(sweeps), len(sweeps), "done"))
        msg_q.put(("done", n_ok))
    except Exception as e:
        import traceback
        msg_q.put(("error", f"{e}\n{traceback.format_exc(limit=4)}"))
    finally:
        for rd in readers.values():
            rd.close()
