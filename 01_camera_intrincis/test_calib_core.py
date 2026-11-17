"""Synthetic tests for calib_core (no camera needed).   Run:  python -m pytest 01_camera_intrincis -q"""
import math

import cv2
import numpy as np
import pytest

import calib_core as cc

W, H = 1280, 720
COLS, ROWS, SQ = 9, 6, 0.02
K_TRUE = np.array([[1000.0, 0, 652.0], [0, 1000.0, 355.0], [0, 0, 1]])
D_TRUE = np.array([-0.15, 0.08, 0.0008, -0.0005, 0.0])


def random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    obj = cc.make_object_points(COLS, ROWS, SQ, SQ)
    center = obj.mean(axis=0)
    poses = []
    while len(poses) < n:
        rvec = np.deg2rad(rng.uniform([-35, -35, -20], [35, 35, 20]))
        R, _ = cv2.Rodrigues(rvec)
        z = rng.uniform(0.22, 0.45)
        cam_center = np.array([rng.uniform(-0.6, 0.6) * z * 0.6, rng.uniform(-0.4, 0.4) * z * 0.6, z])
        tvec = cam_center - R @ center
        pts, _ = cv2.projectPoints(obj.astype(np.float64), rvec, tvec, K_TRUE, D_TRUE)
        pts = pts.reshape(-1, 2)
        if pts[:, 0].min() > 30 and pts[:, 0].max() < W - 30 and pts[:, 1].min() > 30 and pts[:, 1].max() < H - 30:
            poses.append((rvec, tvec, pts.astype(np.float32)))
    return poses


def render_board(rvec, tvec):
    """Binary image as produced by the blinking frame generator: black squares = 0, everything else = 255."""
    R, _ = cv2.Rodrigues(rvec)
    grid = np.stack(np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0), -1).reshape(-1, 1, 2)
    norm = cv2.undistortPointsIter(grid, K_TRUE, D_TRUE, None, None,
                                   (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)
    rays = np.concatenate([norm, np.ones((len(norm), 1))], axis=1)
    n_cam = R[:, 2]
    s = (n_cam @ tvec) / (rays @ n_cam)
    pb = (s[:, None] * rays - tvec) @ R  # = R^T (P_c - t)
    ix = np.floor(pb[:, 0] / SQ).astype(int) + 1
    iy = np.floor(pb[:, 1] / SQ).astype(int) + 1
    inside = (ix >= 0) & (ix <= COLS) & (iy >= 0) & (iy <= ROWS) & (s > 0)
    black = inside & ((ix + iy) % 2 == 0)
    return np.where(black, 0, 255).astype(np.uint8).reshape(H, W)


def match_order(found, gt):
    d1 = np.linalg.norm(found - gt, axis=1).mean()
    d2 = np.linalg.norm(found[::-1] - gt, axis=1).mean()
    return min(d1, d2)


def test_points_on_grid():
    obj = cc.make_object_points(COLS, ROWS, 1.0, 1.0)[:, :2] * 50 + 100
    assert cc.points_on_grid(obj, COLS, ROWS)
    bad = obj.copy()
    bad[[3, 4]] = bad[[4, 3]]
    assert not cc.points_on_grid(bad, COLS, ROWS)


def make_blinking_events(mask, duration_us=400_000, period_us=16_667, noise_per_toggle=2000, seed=1):
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(mask)
    dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
    chunks = []
    for k, t0 in enumerate(range(1000, duration_us, period_us)):
        n = len(xs)
        ev = np.empty(n + noise_per_toggle, dtype)
        ev["x"][:n], ev["y"][:n], ev["p"][:n] = xs, ys, k % 2
        ev["x"][n:] = rng.integers(0, W, noise_per_toggle)
        ev["y"][n:] = rng.integers(0, H, noise_per_toggle)
        ev["p"][n:] = rng.integers(0, 2, noise_per_toggle)
        ev["t"] = np.sort(t0 + rng.integers(0, 3000, len(ev)))
        chunks.append(ev)
    ev = np.concatenate(chunks)
    return ev[np.argsort(ev["t"], kind="stable")]


def test_detector_on_synthetic_blinking_board():
    rvec, tvec, gt = random_poses(1, seed=3)[0]
    board = render_board(rvec, tvec)
    events = make_blinking_events(board == 0)
    det = cc.BlinkingChessboardDetector(W, H, COLS, ROWS, cc.BlinkingConfig())
    windows = []
    for t0 in range(0, int(events["t"][-1]) + 10_000, 10_000):  # 10 ms buffers, like the live camera
        sl = events[(events["t"] >= t0) & (events["t"] < t0 + 10_000)]
        windows += det.process_events(sl)
    found = [w for w in windows if w.corners is not None]
    assert len(windows) >= 3
    assert found, "chessboard never detected"
    assert match_order(found[0].corners, gt) < 0.6


def test_static_noise_does_not_trigger():
    events = make_blinking_events(np.zeros((H, W), bool), noise_per_toggle=5000)
    det = cc.BlinkingChessboardDetector(W, H, COLS, ROWS, cc.BlinkingConfig())
    assert all(w.frame is None for w in det.process_events(events))


def test_view_collector():
    coll = cc.ViewCollector(skip_time_us=1_000_000, min_shift_px=10)
    c = np.random.rand(54, 2).astype(np.float32) * 500
    ok, _ = coll.accept(cc.BlinkingWindow(0, None, c), [])
    assert ok
    assert not coll.accept(cc.BlinkingWindow(500_000, None, c + 50), [c])[0]  # skip time
    assert not coll.accept(cc.BlinkingWindow(2_000_000, None, c + 1), [c])[0]  # same pose
    assert coll.accept(cc.BlinkingWindow(2_000_000, None, c + 30), [c])[0]


@pytest.mark.parametrize("model", [cc.PINHOLE, cc.FISHEYE])
def test_calibration_recovers_intrinsics(model):
    rng = np.random.default_rng(5)
    poses = random_poses(25, seed=7)
    pts = [p + rng.normal(0, 0.1, p.shape).astype(np.float32) for _, _, p in poses]
    pts[4] = pts[4] + rng.normal(0, 4.0, pts[4].shape).astype(np.float32)  # one bad view -> outlier
    res = cc.run_calibration(pts, (W, H), COLS, ROWS, SQ, SQ, model=model, outlier_ths=2.0, log=lambda s: None)
    assert not res.selected[4]
    if model == cc.PINHOLE:
        assert res.rms < 0.3
        assert abs(res.K[0, 0] - 1000) < 5 and abs(res.K[1, 1] - 1000) < 5
        assert abs(res.K[0, 2] - 652) < 5 and abs(res.K[1, 2] - 355) < 5
        assert abs(res.d.ravel()[0] - D_TRUE[0]) < 0.02
    else:  # The fisheye model can only approximate the radial-tangential distortion
        assert res.rms < 1.0
        assert abs(res.K[0, 0] - 1000) < 30
    assert "fx =" in res.summary()


def test_refinement_on_rendered_views(tmp_path):
    poses = random_poses(12, seed=11)
    views = []
    for rvec, tvec, gt in poses:
        frame = render_board(rvec, tvec)
        corners = cc.find_chessboard_corners(frame, COLS, ROWS)
        assert corners is not None
        views.append(cc.CalibrationView(corners, frame))
    pts = [v.corners for v in views]
    frames = [v.frame for v in views]
    base = cc.run_calibration(pts, (W, H), COLS, ROWS, SQ, SQ, outlier_ths=-1, log=lambda s: None)
    ref = cc.run_calibration(pts, (W, H), COLS, ROWS, SQ, SQ, outlier_ths=-1, refine=True, frames=frames,
                             log=lambda s: None)
    assert ref.refined
    assert abs(ref.K[0, 0] - 1000) < 10 and abs(ref.K[0, 2] - 652) < 10 and abs(ref.K[1, 2] - 355) < 10
    assert ref.rms < 0.5
    print(f"\nbase rms={base.rms:.3f} fx={base.K[0, 0]:.2f} | refined rms={ref.rms:.3f} fx={ref.K[0, 0]:.2f}")

    # I/O round trip
    cc.save_detections(tmp_path, (W, H), COLS, ROWS, SQ, SQ, views, camera_serial="00051788")
    size, pattern, loaded, serial = cc.load_detections(tmp_path)
    assert size == (W, H) and pattern["n_cols"] == COLS and len(loaded) == len(views)
    assert serial == "00051788"
    assert np.allclose(loaded[0].corners, views[0].corners, atol=1e-3)
    assert np.array_equal(loaded[0].frame, views[0].frame)
    jp, yp = cc.save_intrinsics(tmp_path, ref, {"camera_serial": serial}, name=f"intrinsics_{serial}")
    assert jp.name == "intrinsics_00051788.json" and yp.name == "intrinsics_00051788_opencv.yaml"
    fs = cv2.FileStorage(str(yp), cv2.FILE_STORAGE_READ)
    assert np.allclose(fs.getNode("camera_matrix").mat(), ref.K)
    img = cc.draw_view(frames[0], ref.pts_2d[0], COLS, ROWS, (W, H), ref, 0)
    assert img.shape == (H, W, 3)
