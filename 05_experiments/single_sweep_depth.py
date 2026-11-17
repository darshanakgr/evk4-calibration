"""
Depth image and point cloud from a single laser sweep (experiment code).

Pipeline for one sweep (one pass of the stripe across the scene):

1. Segment the recording into sweeps (bursts of ON events in the left camera) and pick one.
2. Within that sweep, keep the ON events that pass the spatio-temporal neighbour filter, and the first such event per
   pixel: its time is when the stripe reached that pixel (the arrival time).
3. Per-event time matching (method C of stripe_stereo): for every left event, the right stripe x at the same instant
   on the same rectified row -> one disparity per left event.
4. Rasterise to the rectified left pixel grid, then aggregate with an edge-preserving robust filter: in a k x k window
   (default 7 x 7, >= 8 agreeing values),
   take the median disparity, keep only the values within +-max_dev px of it, and average them. Pixels with too little
   support are left empty. This also fills single-pixel holes between stripe events.
5. Depth Z = f B / d; point cloud in the rectified left camera frame (x right, y down, z forward, metres).

Usage:  python single_sweep_depth.py [folder] [dwell_us] [sweep_index]     (defaults: cw 20 1)
Outputs go to 05_experiments/output/single_sweep/<folder>_<dwell>us_sweep<k>_*
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stripe_stereo as ss  # noqa: E402
import modulation_analysis as ma  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output" / "single_sweep"


# ----------------------------------------------------------------------------------------------------------------------
# Segmentation of the recording into sweeps
# ----------------------------------------------------------------------------------------------------------------------
def list_sweeps(p: ss.Pair, pad_us: int = 5000):
    """(t0, t1) of every sweep, padded, without filtering (cheap)."""
    return [(max(0, a - pad_us), min(p.left.duration_us, b + pad_us))
            for a, b in ma.find_sweeps(p.left.t, p.left.duration_us, min_len_ms=3)]


def segment_sweep(p: ss.Pair, t0: int, t1: int, delta_us: float = 300.0, min_neighbours: int = 2,
                  width: int = 1280, height: int = 720) -> ss.Scan:
    """One sweep -> Scan with direction, speed and the first neighbour-filtered ON event of each pixel per camera."""
    tc, xc = ss.stripe_track(p.left, t0, t1)
    v = np.diff(xc) / np.diff(tc) * 1e3
    direction = int(np.sign(np.median(v)))
    scan = ss.Scan(t0, t1, direction, float(np.median(np.abs(v[np.sign(v) == direction]))))
    for side, s in (("left", p.left), ("right", p.right)):
        i0, i1 = np.searchsorted(s.t, [t0, t1])
        cand = np.arange(i0, i1)
        cand = cand[ss.neighbour_filter(s.pix[i0:i1], s.t[i0:i1], width, height, delta_us, min_neighbours)]
        _, first = np.unique(s.pix[cand], return_index=True)
        first = cand[np.sort(first)]
        scan.idx[side] = first[s.row[first] >= 0]
    return scan


# ----------------------------------------------------------------------------------------------------------------------
# Rasterisation and aggregation
# ----------------------------------------------------------------------------------------------------------------------
def rasterise(row, x, val, width: int = 1280, height: int = 720):
    """Mean of val per rectified pixel (round(x), row); NaN where empty."""
    xi = np.rint(x).astype(np.int64)
    ok = (xi >= 0) & (xi < width) & (row >= 0) & (row < height)
    k = row[ok] * width + xi[ok]
    n = np.bincount(k, minlength=width * height)
    s = np.bincount(k, val[ok], minlength=width * height)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.where(n > 0, s / n, np.nan)
    return img.reshape(height, width)


def robust_aggregate(img: np.ndarray, k: int = 7, max_dev: float = 1.5, min_support: int = 8):
    """Edge-preserving robust smoothing of a sparse disparity image.
    For each pixel: median m of the valid values in the k x k window, then the mean of the values within +-max_dev of
    m. Output where >= min_support values agree (also at empty pixels, which fills small holes). Returns (out, support)."""
    h, w = img.shape
    r = k // 2
    pad = np.pad(img, r, constant_values=np.nan)
    valid = np.isfinite(pad)
    # Pixels with enough valid neighbours (box count)
    cnt = np.lib.stride_tricks.sliding_window_view(valid, (k, k)).sum((-1, -2))
    yy, xx = np.nonzero(cnt >= min_support)
    win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))[yy, xx].reshape(yy.size, k * k).astype(np.float32)
    srt = np.sort(win, 1)  # NaN sorts last
    n = np.isfinite(srt).sum(1)
    med = 0.5 * (srt[np.arange(n.size), (n - 1) // 2] + srt[np.arange(n.size), n // 2])
    m = np.abs(win - med[:, None]) <= max_dev  # NaN compares False
    sup = m.sum(1)
    mean = np.where(m, win, 0).sum(1) / np.maximum(sup, 1)
    out = np.full((h, w), np.nan)
    support = np.zeros((h, w), np.int16)
    good = sup >= min_support
    out[yy[good], xx[good]] = mean[good]
    support[yy, xx] = sup
    return out, support


# ----------------------------------------------------------------------------------------------------------------------
# One sweep -> depth image + point cloud
# ----------------------------------------------------------------------------------------------------------------------
def sweep_depth(p: ss.Pair, scan: ss.Scan, geom: ss.Geometry, tau_px: float = 8.0, k: int = 7, max_dev: float = 1.5,
                min_support: int = 8) -> dict:
    tim = {}
    t = time.perf_counter()
    res = ss.match_time(p, scan, tau_px / scan.speed_px_ms * 1e3)
    tim["match_s"] = time.perf_counter() - t
    t = time.perf_counter()
    raw = rasterise(res["row"], res["x"], res["d"], geom.width, geom.height)
    disp, support = robust_aggregate(raw, k, max_dev, min_support)
    tim["aggregate_s"] = time.perf_counter() - t
    t = time.perf_counter()
    depth = geom.depth(disp)  # NaN stays NaN
    yy, xx = np.nonzero(np.isfinite(disp))
    pts = geom.points(yy, xx, disp[yy, xx])
    tim["points_s"] = time.perf_counter() - t
    tl, xl, rl = ss.scan_events(p.left, scan.idx["left"])
    tr, xr, rr = ss.scan_events(p.right, scan.idx["right"])
    arrival = {"left": rasterise(rl, xl, (tl - scan.t0) / 1e3, geom.width, geom.height),
               "right": rasterise(rr, xr, (tr - scan.t0) / 1e3, geom.width, geom.height)}
    return {"events": res, "raw_disp": raw, "disp": disp, "support": support, "depth": depth, "points": pts,
            "pix": (yy, xx), "arrival_ms": arrival, "timing": tim}


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray):
    """Binary little-endian PLY with float xyz and uchar rgb."""
    n = len(pts)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(rec.tobytes())


def plane_stats(pts: np.ndarray):
    """Signed distance (mm) of each point to a robust plane fit, and summary numbers."""
    step = max(1, len(pts) // 200000)
    n, c, _ = ss.fit_plane_ransac(pts[::step], thr=0.02)
    if n[2] < 0:
        n, c = -n, -c
    e = (pts @ n - c) * 1e3
    e -= np.median(e)
    return e, {"mad_mm": float(1.4826 * np.median(np.abs(e))), "outliers_20mm": float(np.mean(np.abs(e) > 20)),
               "rms_inliers_mm": float(np.sqrt(np.mean(e[np.abs(e) < 20] ** 2))), "normal": n.tolist(),
               "distance_m": float(c)}


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------
def figures(r: dict, geom: ss.Geometry, scan: ss.Scan, title: str, stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = r["depth"] * 1e3
    zlo, zhi = np.nanpercentile(z, [1, 99])
    raw_z = geom.depth(r["raw_disp"]) * 1e3
    e_img = np.full(z.shape, np.nan)
    e_img[r["pix"]] = r["plane_err_mm"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 8.4), constrained_layout=True)
    for a, side in zip(ax[0], ("left", "right")):
        im = a.imshow(r["arrival_ms"][side], cmap="turbo", interpolation="nearest")
        a.set_title(f"{side} camera: stripe arrival time in this sweep (first filtered ON event)")
        fig.colorbar(im, ax=a, shrink=0.8, label="ms since sweep start")
    im = ax[1, 0].imshow(raw_z, cmap="viridis_r", vmin=zlo, vmax=zhi, interpolation="nearest")
    ax[1, 0].set_title("depth per left event (method C, before aggregation)")
    fig.colorbar(im, ax=ax[1, 0], shrink=0.8, label="Z (mm)")
    im = ax[1, 1].imshow(z, cmap="viridis_r", vmin=zlo, vmax=zhi, interpolation="nearest")
    ax[1, 1].set_title("depth image after robust 7×7 aggregation")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.8, label="Z (mm)")
    for a in ax.ravel():
        a.set_xlabel("rectified x (px)")
        a.set_ylabel("rectified row")
    fig.suptitle(title)
    fig.savefig(f"{stem}_depth.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
    im = ax[0].imshow(e_img, cmap="RdBu_r", vmin=-10, vmax=10, interpolation="nearest")
    ax[0].set_title("distance of each depth pixel from the best-fit plane")
    fig.colorbar(im, ax=ax[0], shrink=0.85, label="mm")
    ax[0].set_xlabel("rectified x (px)")
    ax[0].set_ylabel("rectified row")
    ax[1].hist(np.clip(r["plane_err_mm"], -30, 30), bins=241, color="0.3")
    ax[1].set_xlabel("distance from plane (mm, clipped to ±30)")
    ax[1].set_ylabel("pixels")
    ax[1].set_yscale("log")
    ax[1].set_title("residual distribution")
    fig.savefig(f"{stem}_residual.png", dpi=110)
    plt.close(fig)

    pts = r["points"]
    step = max(1, len(pts) // 60000)
    sub = pts[::step]
    zc = sub[:, 2] * 1e3
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    a = fig.add_subplot(1, 3, 1, projection="3d")
    a.scatter(sub[:, 0], sub[:, 2], -sub[:, 1], c=zc, cmap="viridis_r", vmin=zlo, vmax=zhi, s=0.3, linewidths=0)
    a.view_init(elev=20, azim=-60)
    a.set_box_aspect((np.ptp(sub[:, 0]), np.ptp(sub[:, 2]), np.ptp(sub[:, 1])))  # true scale
    a.set_xlabel("x (m)")
    a.set_ylabel("z (m)")
    a.set_zlabel("−y (m)")
    a.set_title("oblique view, true scale")
    for i, (u, lab) in enumerate([(0, "x (m)"), (1, "y (m), down")]):
        a = fig.add_subplot(1, 3, i + 2)
        a.scatter(sub[:, u], sub[:, 2] * 1e3, c=zc, cmap="viridis_r", vmin=zlo, vmax=zhi, s=0.3, linewidths=0)
        a.set_xlabel(lab)
        a.set_ylabel("Z (mm)")
        a.set_ylim(np.percentile(zc, 0.1) - 15, np.percentile(zc, 99.9) + 15)
        a.invert_yaxis()
        a.set_title(("top view (x–Z)" if u == 0 else "side view (y–Z)") + ", Z stretched")
        a.grid(alpha=0.3)
    fig.suptitle(f"{title}: point cloud ({len(pts):,} points, every {step}th shown)")
    fig.savefig(f"{stem}_cloud.png", dpi=110)
    plt.close(fig)


def main(folder="cw", dwell=20, k_sweep=1):
    geom = ss.load_geometry()
    t = time.perf_counter()
    p = ss.load_pair(folder, dwell, geom)
    t_load = time.perf_counter() - t
    sweeps = list_sweeps(p)
    t0, t1 = sweeps[k_sweep]
    t = time.perf_counter()
    scan = segment_sweep(p, t0, t1)
    t_seg = time.perf_counter() - t
    r = sweep_depth(p, scan, geom)
    r["timing"] = {"segment_s": t_seg, **r["timing"]}
    e, st = plane_stats(r["points"])
    r["plane_err_mm"] = e
    e_raw, st_raw = plane_stats(geom.points(r["events"]["row"], r["events"]["x"], r["events"]["d"]))

    # Coverage: fraction of the area where both cameras saw the stripe (right image shifted by the median disparity)
    # that received a depth value
    from scipy.ndimage import binary_closing
    L = binary_closing(np.isfinite(r["arrival_ms"]["left"]), iterations=2)
    R = binary_closing(np.isfinite(r["arrival_ms"]["right"]), iterations=2)
    dm = int(round(np.nanmedian(r["disp"])))
    lit = np.zeros_like(L)
    lit[:, dm:] = L[:, dm:] & R[:, :-dm]
    info = {"folder": folder, "dwell_us": dwell, "sweep": k_sweep, "n_sweeps": len(sweeps),
            "t0_us": int(t0), "t1_us": int(t1), "direction": "left-to-right" if scan.direction > 0 else "right-to-left",
            "speed_px_ms": scan.speed_px_ms, "cross_ms": geom.width / scan.speed_px_ms,
            "first_on_events": {s: int(v.size) for s, v in scan.idx.items()},
            "matched_events": int(r["events"]["d"].size),
            "depth_pixels": int(np.isfinite(r["disp"]).sum()),
            "coverage_of_both_lit": float(np.isfinite(r["disp"])[lit].mean()),
            "z_median_m": float(np.nanmedian(r["depth"])),
            "per_event": st_raw, "aggregated": st,
            "timing_s": {"load_recording": t_load, **r["timing"]}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"{folder}_{dwell}us_sweep{k_sweep}"
    np.save(f"{stem}_depth_m.npy", r["depth"].astype(np.float32))
    import matplotlib
    zlo, zhi = np.nanpercentile(r["depth"], [1, 99])
    rgb = (matplotlib.colormaps["viridis_r"](np.clip((r["points"][:, 2] - zlo) / (zhi - zlo), 0, 1))[:, :3] * 255)
    write_ply(Path(f"{stem}_cloud.ply"), r["points"], rgb.astype(np.uint8))
    title = (f"{folder} {dwell} µs dwell, sweep {k_sweep} ({info['direction']}, {info['cross_ms']:.0f} ms)")
    figures(r, geom, scan, title, stem)
    with open(f"{stem}_info.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if len(a) > 0 else "cw", int(a[1]) if len(a) > 1 else 20, int(a[2]) if len(a) > 2 else 1)
