"""Tests of stripe_depth_core on synthetic stripe sweeps (no camera, no calibration file needed).

Run:  python -m pytest 06_stripe_depth -q
"""
import numpy as np
import pytest

import recorder_core as rc
import stripe_depth_core as sd

W, H = 320, 120
V = 20.0  # Stripe speed, px/ms
T0 = 1_000_000  # Sweep start (us)


def identity_rig(f=500.0, baseline=0.1):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lut = np.stack([xx, yy], -1)
    return sd.StereoRig(f, W / 2, H / 2, baseline, W, H, {"left": lut, "right": lut.copy()},
                        {"left": "L1", "right": "R1"})


def disparity_map(kind="ridge"):
    """True disparity of every left pixel."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    if kind == "flat":
        return np.full((H, W), 40.0)
    return 40.0 + 10.0 * np.exp(-((xx - 180) / 30.0) ** 2)  # A ridge (closer) in the middle


def stripe_events(side, disp, jitter_us=0.0, n_noise=0, seed=0, direction=1):
    """ON events of one camera for a stripe moving at V px/ms. The left pixel x is reached at t = x / V; the right
    camera sees the same point at x_R = x_L - d, i.e. the right pixel x_R fires when the left stripe is at x_R + d."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    if side == "left":
        x_stripe = xx.astype(np.float64)
        ok = np.ones_like(xx, bool)
    else:
        # Right pixel x_R corresponds to the left pixel x_L with x_L - d(x_L) = x_R: fixed-point iteration (converges
        # because |d'| < 1), with d interpolated along the row
        x_l = xx + disp
        grid = np.arange(W, dtype=np.float64)
        for _ in range(60):
            x_l = xx + np.stack([np.interp(x_l[r], grid, disp[r]) for r in range(H)])
        x_stripe = x_l
        ok = (x_l >= 0) & (x_l < W)
    pos = x_stripe if direction > 0 else (W - 1 - x_stripe)
    t = T0 + 5000 + pos / V * 1e3 + rng.normal(0, jitter_us, xx.shape)
    x, y, t = xx[ok], yy[ok], t[ok]
    # Two ON events per pixel (stripe width), noise events anywhere in the sweep
    x = np.r_[x, x, rng.integers(0, W, n_noise)]
    y = np.r_[y, y, rng.integers(0, H, n_noise)]
    t = np.r_[t, t + 60, rng.uniform(T0, T0 + 199_000, n_noise)]  # Noise over the whole recording
    o = np.argsort(t, kind="stable")
    ev = np.zeros(o.size, [("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("t", "<i8")])
    ev["x"], ev["y"], ev["p"], ev["t"] = x[o], y[o], 1, np.rint(t[o]).astype(np.int64)
    return ev


def write_recording(folder, disp, **kw):
    t1 = T0 + 200_000  # Mostly quiet: the sweep must stand out from the background
    for side, serial, seed in (("left", "L1", 1), ("right", "R1", 2)):
        ev = stripe_events(side, disp, seed=seed, **kw)
        w = rc.EventFileWriter(folder / f"{side}_{serial}.h5", W, H, T0, {"serial": serial, "role": side}).start()
        w.put(ev)
        w.finish(t1)
        assert w.join(30) and w.error is None
    return {r: rc.EventFileReader(p) for r, p in rc.find_recording_files(folder).items()}


def test_detect_sweeps():
    c = np.full(3000, 5.0)
    c[500:560] = 4000  # Sweep 1
    c[565:570] = 4000  # Gap of 5 ms: same sweep
    c[1500:1502] = 4000  # Too short (2 ms)
    c[2000:2100] = 4000
    sw = sd.detect_sweeps(c, 10_000_000, sd.SweepParams(min_len_ms=3, merge_gap_ms=20, pad_ms=5))
    assert sw == [(10_000_000 + 495_000, 10_000_000 + 575_000), (10_000_000 + 1_995_000, 10_000_000 + 2_105_000)]
    only2 = sd.detect_sweeps(c, 10_000_000, sd.SweepParams(pad_ms=0), t_range=(10_000_000 + 1_000_000, 10_000_000 + 3_000_000))
    assert only2 == [(10_000_000 + 2_000_000, 10_000_000 + 2_100_000)]
    assert sd.detect_sweeps(np.full(100, 3.0), 0) == []


def test_first_events_skip_isolated_noise():
    """A pixel's isolated early event is skipped; its first event with active neighbours is kept."""
    w, h = 10, 10
    pix, t = [], []
    for x in range(2, 8):  # A vertical stripe segment at time 1000 + x, column x, rows 3..6
        for y in range(3, 7):
            pix.append(y * w + x)
            t.append(1000 + x)
    pix.append(4 * w + 5)  # Isolated noise at a stripe pixel, long before the stripe
    t.append(10)
    pix.append(0)  # Isolated noise pixel alone
    t.append(500)
    pix, t = np.array(pix), np.array(t)
    o = np.argsort(t, kind="stable")
    pix, t = pix[o], t[o]
    sel = sd.first_events(pix, t, w, h, delta_us=3, min_neighbours=2)
    got = dict(zip(pix[sel].tolist(), t[sel].tolist()))
    assert 0 not in got  # Lonely pixel removed
    assert got[4 * w + 5] == 1005  # Noise at t = 10 skipped, the stripe event kept
    assert np.all(np.diff(t[sel]) >= 0)
    # Filter off: plain first event per pixel
    sel0 = sd.first_events(pix, t, w, h, 3, 0)
    assert dict(zip(pix[sel0].tolist(), t[sel0].tolist()))[4 * w + 5] == 10


def test_first_events_equals_brute_force():
    rng = np.random.default_rng(4)
    w, h, n = 30, 20, 4000
    pix = rng.integers(0, w * h, n)
    t = np.sort(rng.integers(0, 20_000, n))
    sel = sd.first_events(pix, t, w, h, 300, 2, max_iter=10_000)
    # Brute force: filter every event, then first survivor per pixel
    keep = []
    for i in range(n):
        x, y = pix[i] % w, pix[i] // w
        c = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dx or dy) and 0 <= x + dx < w and 0 <= y + dy < h:
                    q = (y + dy) * w + x + dx
                    c += bool(np.any((pix == q) & (np.abs(t - t[i]) <= 300)))
        keep.append(c >= 2)
    keep = np.array(keep)
    expected = {}
    for i in np.where(keep)[0]:
        expected.setdefault(int(pix[i]), i)
    assert sorted(sel.tolist()) == sorted(expected.values())


def test_local_fit_exact_on_a_line():
    tr = np.arange(0, 2000, 50.0)
    xr = 100 + 0.02 * tr
    rr = np.zeros_like(tr, dtype=np.int64)
    a, n, ok = sd.local_fit(np.array([1000.0, 1975.0, 5000.0]), np.array([0, 0, 0]), tr, xr, rr, tau_us=200,
                            row_halfwidth=0)
    assert ok[0] and a[0] == pytest.approx(120.0) and n[0] == 9
    assert not ok[1]  # At the edge of the data: would extrapolate
    assert not ok[2]  # No data


@pytest.mark.parametrize("direction", [1, -1])
def test_process_sweep_recovers_disparity(tmp_path, direction):
    disp = disparity_map("ridge")
    readers = write_recording(tmp_path, disp, jitter_us=30.0, n_noise=3000, direction=direction)
    rig = identity_rig()
    try:
        counts = sd.on_counts_per_ms(readers["left"])
        sw = sd.detect_sweeps(counts, readers["left"].t_start, sd.SweepParams(pad_ms=2))
        assert len(sw) == 1
        p = sd.DepthParams(agg_k=5, agg_min_support=6)
        res = sd.process_sweep(readers, rig, *sw[0], p)
    finally:
        for rd in readers.values():
            rd.close()
    assert res.direction == direction and res.speed_px_ms == pytest.approx(V, rel=0.1)
    ok = np.isfinite(res.disp)
    inner = np.zeros_like(ok)
    inner[5:H - 5, 70:W - 5] = True  # Right camera sees x_L >= d (~40-50 px)
    assert ok[inner].mean() > 0.95
    err = (res.disp - disp)[ok & inner]
    assert np.median(np.abs(err)) < 0.25 and np.mean(np.abs(err) < 1.0) > 0.97
    # Depth and point cloud
    rd_like = {"f": rig.f, "cx": rig.cx, "cy": rig.cy, "fB": rig.f * rig.baseline}
    pts, (yy, xx) = sd.cloud_from_result(res, rd_like)
    assert np.allclose(pts[:, 2], rig.f * rig.baseline / res.disp[yy, xx])
    assert res.stats["matched"] > 0.8 * (H * (W - 50))


def test_roles_follow_serials(tmp_path):
    """Files recorded with swapped roles: the calibration sides are taken from the serial numbers."""
    disp = disparity_map("flat")
    readers = write_recording(tmp_path, disp)
    rig = identity_rig()
    rig.serial = {"left": "L1", "right": "R1"}
    swapped = {"left": readers["right"], "right": readers["left"]}
    try:
        res = sd.process_sweep(swapped, rig, T0, T0 + 30_000, sd.DepthParams(agg_k=5, agg_min_support=6))
    finally:
        for rd in readers.values():
            rd.close()
    assert np.nanmedian(res.disp) == pytest.approx(40.0, abs=0.2)


def test_robust_aggregate_rejects_outliers_and_fills_holes():
    img = np.full((40, 40), 50.0, np.float32)
    img[::3, ::2] = np.nan  # Holes
    img[20, 20] = 90.0  # Outlier
    out, sup = sd.robust_aggregate(img, k=5, max_dev=1.5, min_support=6)
    assert np.allclose(out[2:-2, 2:-2], 50.0)  # Outlier rejected, holes filled
    assert np.isfinite(out).mean() > 0.9


def test_save_load_and_ply(tmp_path):
    rig = identity_rig()
    disp = np.full((H, W), np.nan, np.float32)
    disp[10:20, 100:150] = 50.0
    res = sd.SweepResult(T0, T0 + 1000, 1, 20.0, 400.0, {"left": disp.copy(), "right": disp.copy()}, disp, disp,
                         np.zeros((H, W), np.int16), {"t0_us": T0, "t1_us": T0 + 1000, "direction": 1,
                                                      "speed_px_ms": 20.0, "tau_us": 400.0, "fB": 50.0})
    files = sd.save_result(res, sd.rig_dict(rig), tmp_path / "s")
    back, rig_like = sd.load_result(files["npz"])
    assert np.array_equal(np.isfinite(back.disp), np.isfinite(disp)) and rig_like["fB"] == pytest.approx(50.0)
    pts = sd.read_ply_xyz(files["ply"])
    assert len(pts) == 500 and np.allclose(pts[:, 2], 1.0)
    assert np.isclose(np.nanmedian(back.depth), 1.0)


def test_depth_png16_roundtrip(tmp_path):
    depth = np.full((H, W), np.nan)
    depth[10:30, 50:90] = 0.5664  # m
    depth[40, 40] = 70.0  # Beyond 65.535 m: clipped
    path = tmp_path / "dé_depth_mm.png"  # Non-ASCII path
    sd.write_depth_png16(path, depth)
    back = sd.read_depth_png16(path)
    assert np.array_equal(np.isfinite(back), np.isfinite(depth))
    assert np.allclose(back[10:30, 50:90], 0.566) and back[40, 40] == pytest.approx(65.535)


def test_plane_summary_on_tilted_plane():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-0.2, 0.2, 20000), rng.uniform(-0.1, 0.1, 20000)
    z = 0.6 + 0.1 * y + rng.normal(0, 0.002, x.size)  # Tilted by atan(0.1) = 5.7 deg, 2 mm noise
    p = sd.plane_summary(np.stack([x, y, z], -1))
    assert p["tilt_deg"] == pytest.approx(5.71, abs=0.2) and p["mad_mm"] == pytest.approx(2.0 * np.cos(0.0997), abs=0.2)
