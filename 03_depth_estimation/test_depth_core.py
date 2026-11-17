"""Synthetic tests for depth_core (no camera needed).   Run:  python -m pytest 03_depth_estimation -q"""
import cv2
import numpy as np
import pytest

import depth_core as dc
from depth_core import sc

import test_stereo_core as T  # Synthetic stereo rig of step 2 (K1, D1, K2, D2, R_TRUE, T_TRUE)

W, H = 1280, 720
SCALE = 0.5


@pytest.fixture(scope="module")
def rig():
    res = sc.StereoResult(model=sc.PINHOLE, image_size=(W, H), serial_left="L", serial_right="R", K1=T.K1, d1=T.D1,
                          K2=T.K2, d2=T.D2, R=T.R_TRUE, T=T.T_TRUE, E=np.eye(3), F=np.eye(3), rms=0.1,
                          selected=np.ones(3, bool), per_pair_errors=np.zeros(3), rvecs=[], tvecs=[])
    sc.compute_rectification(res, alpha=0.0)
    res.rect_errors = np.zeros(3)
    luts = {s: dc.make_index_lut(res, s, (W, H), SCALE) for s in ("left", "right")}
    return res, luts


def render_scene(res, luts, pts3d, t_pts, t_now=30_000, tau=30_000):
    """Events at the raw (distorted) pixels where each 3D point (left camera frame) projects, then time surfaces."""
    ws, hs = dc.scaled_size((W, H), SCALE)
    images = {}
    for side, K, D, rvec, tvec in (("left", T.K1, T.D1, np.zeros(3), np.zeros(3)),
                                   ("right", T.K2, T.D2, cv2.Rodrigues(T.R_TRUE)[0], T.T_TRUE)):
        px, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D)
        px = np.round(px.reshape(-1, 2)).astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        order = np.argsort(t_pts[ok])
        raw_idx = (px[ok, 1] * W + px[ok, 0])[order]
        ts = dc.TimeSurface(ws, hs)
        ts.add(luts[side][raw_idx], t_pts[ok][order])
        images[side] = ts.render(t_now, tau)
    return images


def plane_points(n, z_fn, x_range=(-0.45, 0.35), y_range=(-0.25, 0.25), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)
    y = rng.uniform(*y_range, n)
    return np.stack([x, y, z_fn(x, y)], axis=1), rng.integers(0, 30_000, n).astype(np.int64)


def check_depth(res, frame, geom, pts3d, tol=0.02):
    """Compares the estimated depth with the true depth (in the rectified left frame) at each 3D point's pixel."""
    xr = pts3d @ res.R1.T
    u = np.round(geom.f * xr[:, 0] / xr[:, 2] + geom.cx).astype(int)
    v = np.round(geom.f * xr[:, 1] / xr[:, 2] + geom.cy).astype(int)
    ok = (u >= 0) & (u < geom.width) & (v >= 0) & (v < geom.height)
    est, true = frame.depth[v[ok], u[ok]], xr[ok, 2]
    m = np.isfinite(est)
    assert m.sum() > 2000, "too few valid depths"
    rel = np.abs(est[m] - true[m]) / true[m]
    assert np.median(rel) < tol, f"median relative error {np.median(rel):.3f}"
    return float(np.median(rel)), float(m.mean())


def test_geometry_and_luts(rig):
    res, luts = rig
    geom = dc.DepthGeometry.from_stereo(res, SCALE)
    assert (geom.width, geom.height) == (640, 360)
    assert abs(geom.baseline - np.linalg.norm(T.T_TRUE)) < 1e-6 and geom.horizontal and not geom.flip
    # LUT agrees with the step 2 point rectification
    p = sc.rectify_points(np.array([[700.0, 400.0]]), res.K1, res.d1, res.R1, res.P1, res.model)[0]
    idx = luts["left"][400 * W + 700]
    assert abs(idx % 640 - ((p[0] + 0.5) * 0.5 - 0.5)) <= 0.51 and abs(idx // 640 - ((p[1] + 0.5) * 0.5 - 0.5)) <= 0.51
    ident = dc.make_index_lut(None, "left", (W, H), 0.25)
    assert ident[0] == 0 and ident.max() == 320 * 180 - 1


def test_time_surface():
    ts = dc.TimeSurface(4, 2)
    ts.add(np.array([0, 5, -1]), np.array([100, 200, 300]))
    img = ts.render(200, 100)
    assert img[0, 0] == int(255 * np.exp(-1)) and img[1, 1] == 255 and img[0, 1] == 0


def test_event_time_framer_is_independent_of_buffering():
    """Hardware-synced cameras receive the same time base but different buffer splits: frames must be identical."""
    rng = np.random.default_rng(4)
    n = 20_000
    t = np.sort(rng.integers(1_000, 400_000, n)).astype(np.int64)
    idx = rng.integers(0, 64 * 36, n).astype(np.int32)
    outputs = []
    for splits in (np.arange(0, n, 997), np.sort(rng.choice(n, 40, replace=False))):  # Two different bufferings
        framer, ts, frames = dc.EventTimeFramer(20.0), dc.TimeSurface(64, 36), []
        bounds = sorted(set([0] + list(splits) + [n]))  # Every event fed exactly once
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                frames += framer.process(ts, idx[a:b], t[a:b], int(t[b - 1]), 30_000)
        outputs.append(frames)
    keys = [[k for k, _ in f] for f in outputs]
    common = sorted(set(keys[0]) & set(keys[1]))
    assert len(common) >= 6 and all(k % 50_000 == 0 for k in common)
    d0, d1 = dict(outputs[0]), dict(outputs[1])
    for k in common:
        assert np.array_equal(d0[k], d1[k]), f"frame {k} differs"
    # A frame only contains events strictly before its boundary
    ts = dc.TimeSurface(64, 36)
    ts.add(idx[t < common[0]], t[t < common[0]])
    assert np.array_equal(ts.render(common[0], 30_000), d0[common[0]])


def test_depth_tilted_plane(rig):
    res, luts = rig
    pts, t = plane_points(160_000, lambda x, y: 1.0 + 0.4 * x)  # 0.82 - 1.14 m
    imgs = render_scene(res, luts, pts, t)
    geom = dc.DepthGeometry.from_stereo(res, SCALE)
    est = dc.DepthEstimator(geom, dc.MatcherParams(num_disparities=128, block_size=7, activity_threshold=5))
    frame = est.compute(imgs["left"], imgs["right"])
    rel, valid = check_depth(res, frame, geom, pts)
    print(f"\ntilted plane: median rel. error {rel * 100:.2f}%, {valid * 100:.0f}% of the points with a depth, "
          f"{frame.match_ms:.1f} ms")


def test_depth_two_planes_and_bm(rig):
    res, luts = rig
    near, tn = plane_points(80_000, lambda x, y: np.full_like(x, 0.8), x_range=(-0.35, -0.05), seed=1)
    far, tf = plane_points(80_000, lambda x, y: np.full_like(x, 1.6), x_range=(0.05, 0.6), seed=2)
    pts, t = np.concatenate([near, far]), np.concatenate([tn, tf])
    imgs = render_scene(res, luts, pts, t)
    geom = dc.DepthGeometry.from_stereo(res, SCALE)
    for algo in ("SGBM 3-way", "BM"):
        est = dc.DepthEstimator(geom, dc.MatcherParams(algorithm=algo, num_disparities=128, block_size=9,
                                                       activity_threshold=5))
        frame = est.compute(imgs["left"], imgs["right"])
        check_depth(res, frame, geom, near, tol=0.03)
        check_depth(res, frame, geom, far, tol=0.03)


def test_swapped_cameras_are_handled(rig):
    """Left/right swapped w.r.t. OpenCV's convention: negative disparities are handled by mirroring."""
    res, luts = rig
    swapped = sc.StereoResult(model=res.model, image_size=res.image_size, serial_left="R", serial_right="L",
                              K1=T.K2, d1=T.D2, K2=T.K1, d2=T.D1, R=T.R_TRUE.T, T=-T.R_TRUE.T @ T.T_TRUE,
                              E=np.eye(3), F=np.eye(3), rms=0.1, selected=np.ones(3, bool),
                              per_pair_errors=np.zeros(3), rvecs=[], tvecs=[])
    sc.compute_rectification(swapped, 0.0)
    geom = dc.DepthGeometry.from_stereo(swapped, SCALE)
    assert geom.flip and abs(geom.baseline - np.linalg.norm(T.T_TRUE)) < 1e-6
    # Scene in the frame of the new "left" camera (= old right camera)
    pts, t = plane_points(160_000, lambda x, y: np.full_like(x, 1.0), x_range=(-0.3, 0.45))
    l_luts = {"left": dc.make_index_lut(swapped, "left", (W, H), SCALE),
              "right": dc.make_index_lut(swapped, "right", (W, H), SCALE)}
    pts_old_left = pts @ swapped.R.T + swapped.T.ravel()  # X_newright (= old left) = R_s X_newleft + T_s
    imgs = render_scene(res, {"left": l_luts["right"], "right": l_luts["left"]}, pts_old_left, t)
    frame = dc.DepthEstimator(geom, dc.MatcherParams(num_disparities=128, activity_threshold=5)).compute(
        imgs["right"], imgs["left"])
    check_depth(swapped, frame, geom, pts, tol=0.03)


def test_io(rig, tmp_path):
    res, _ = rig
    jp, _ = sc.save_stereo(tmp_path, res, "stereo_L_R")
    loaded = sc.load_stereo(jp)
    for k in ("K1", "d1", "K2", "d2", "R", "T", "R1", "R2", "P1", "P2", "Q"):
        assert np.allclose(getattr(loaded, k), getattr(res, k)), k
    assert sc.find_stereo_file(tmp_path, "L", "R") == jp and sc.find_stereo_file(tmp_path, "R", "L") is None

    geom = dc.DepthGeometry.from_stereo(loaded, SCALE)
    depth = np.full((geom.height, geom.width), np.nan, np.float32)
    depth[100:200, 100:300] = 1.234
    dc.save_depth_png(tmp_path / "d.png", depth)
    back = dc.load_depth_png(tmp_path / "d.png")
    assert np.isnan(back[0, 0]) and abs(back[150, 150] - 1.234) < 1e-3
    pts, rgb = dc.point_cloud(depth, geom, np.zeros((geom.height, geom.width, 3), np.uint8))
    assert len(pts) == 100 * 200 and np.allclose(pts[:, 2], 1.234)
    ply = dc.save_ply(tmp_path / "c.ply", pts, rgb)
    assert ply.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 20000")
    color = dc.colorize_depth(depth, 0.5, 3.0)
    assert color[0, 0].sum() == 0 and color[150, 150].sum() > 0
    meta = dc.depth_metadata(str(jp), loaded, geom, SCALE, dc.MatcherParams(), 30.0, 20.0)
    rec = dc.DepthRecorder(tmp_path / "rec", meta)  # Metadata must be JSON serializable
    gray = np.zeros((geom.height, geom.width), np.uint8)
    rec.write(1.0, 10, 12, 0.5, gray, gray, dc.DepthFrame(depth, depth, 0.5, 1.0))
    rec.close()
    assert (tmp_path / "rec" / "depth" / "000000.png").exists()
    assert len((tmp_path / "rec" / "frames.csv").read_text().splitlines()) == 2
