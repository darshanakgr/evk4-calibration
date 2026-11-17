"""Synthetic tests for stereo_core (no camera needed).   Run:  python -m pytest 02_stereo_rectification -q"""
import cv2
import numpy as np
import pytest

import stereo_core as sc
from stereo_core import cc

W, H = 1280, 720
COLS, ROWS, SQ = 9, 6, 0.03
K1 = np.array([[1000.0, 0, 645.0], [0, 1000.0, 362.0], [0, 0, 1]])
K2 = np.array([[1010.0, 0, 632.0], [0, 1010.0, 351.0], [0, 0, 1]])
D1 = np.array([[-0.12, 0.06, 0.0005, -0.0003, 0.0]])
D2 = np.array([[-0.10, 0.04, -0.0004, 0.0006, 0.0]])
R_TRUE, _ = cv2.Rodrigues(np.deg2rad([0.8, -2.0, 0.5]))
T_TRUE = np.array([[-0.12], [0.003], [0.002]])  # Right camera 12 cm to the right of the left one
OBJ = cc.make_object_points(COLS, ROWS, SQ, SQ)


def intrinsics(K, D, serial, model=sc.PINHOLE):
    return sc.Intrinsics(serial=serial, model=model, image_size=(W, H), K=K, d=D, rms=0.1, path="")


def project(rvec, tvec, K, D, model=sc.PINHOLE):
    return sc._project(OBJ, rvec, tvec, K, D, model).astype(np.float32)


def right_pose(rvec, tvec):
    Rl, _ = cv2.Rodrigues(rvec)
    return cv2.Rodrigues(R_TRUE @ Rl)[0], R_TRUE @ tvec.reshape(3, 1) + T_TRUE


def stereo_poses(n, seed=0, K=(K1, K2), D=(D1, D2), model=sc.PINHOLE):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        rvec = np.deg2rad(rng.uniform([-30, -30, -15], [30, 30, 15]))
        R, _ = cv2.Rodrigues(rvec)
        z = rng.uniform(0.6, 1.1)
        centre = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.1), z])
        tvec = (centre - R @ OBJ.mean(0)).reshape(3, 1)
        rv_r, tv_r = right_pose(rvec, tvec)
        pl, pr = project(rvec, tvec, K[0], D[0], model), project(rv_r, tv_r, K[1], D[1], model)
        if all(p[:, 0].min() > 40 and p[:, 0].max() < W - 40 and p[:, 1].min() > 40 and p[:, 1].max() < H - 40
               for p in (pl, pr)):
            out.append((rvec, tvec, pl, pr))
    return out


def render(rvec, tvec, K, D):
    """Binary detection frame: blinking (black) squares where (ix + iy) is even, like chessboard_pattern.html."""
    R, _ = cv2.Rodrigues(rvec)
    tvec = tvec.ravel()
    grid = np.stack(np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0), -1).reshape(-1, 1, 2)
    norm = cv2.undistortPointsIter(grid, K, D, None, None,
                                   (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)
    rays = np.concatenate([norm, np.ones((len(norm), 1))], axis=1)
    s = (R[:, 2] @ tvec) / (rays @ R[:, 2])
    pb = (s[:, None] * rays - tvec) @ R
    ix = np.floor(pb[:, 0] / SQ).astype(int) + 1
    iy = np.floor(pb[:, 1] / SQ).astype(int) + 1
    black = (ix >= 0) & (ix <= COLS) & (iy >= 0) & (iy <= ROWS) & (s > 0) & ((ix + iy) % 2 == 0)
    return np.where(black, 0, 255).astype(np.uint8).reshape(H, W)


def test_canonicalize_restores_physical_order():
    rvec, tvec, pl, _ = stereo_poses(1, seed=1)[0]
    frame = render(rvec, tvec, K1, D1)
    found = cc.find_chessboard_corners(frame, COLS, ROWS)
    for candidate in (found, found[::-1].copy()):
        canon, ok = sc.canonicalize_corners(candidate, frame, COLS, ROWS)
        assert ok
        assert np.linalg.norm(canon - pl, axis=1).mean() < 0.6  # Corner 0 = object point 0
    _, ok = sc.canonicalize_corners(found, frame, 8, 6)  # cols + rows even: ambiguous
    assert not ok


def test_pair_collector():
    coll = sc.StereoPairCollector(skip_time_s=1.0, min_shift_px=10, sync_window_s=0.3, stability_px=1.5)
    c = (np.random.rand(54, 2) * 500).astype(np.float32)
    prev = sc.Detection(0.0, c, None)
    assert coll.is_stable(prev, c + 0.5, 0.1) and not coll.is_stable(prev, c + 5, 0.1)
    assert not coll.is_stable(prev, c, 0.9)
    L, Rt = sc.Detection(1.0, c, None, True), sc.Detection(1.1, c + 3, None, True)
    assert coll.accept(L, sc.Detection(1.5, c, None, True), [])[0] is False  # not simultaneous
    assert coll.accept(L, sc.Detection(1.1, c, None, False), [])[0] is False  # not stable
    assert coll.accept(L, Rt, [])[0]
    pair = sc.StereoPair(c, c + 3)
    assert not coll.accept(sc.Detection(3.0, c, None, True), sc.Detection(3.0, c + 3, None, True), [pair])[0]
    assert coll.accept(sc.Detection(3.0, c + 40, None, True), sc.Detection(3.0, c + 40, None, True), [pair])[0]


def check_result(res, tol_t=0.002, tol_r_deg=0.1):
    assert np.linalg.norm(res.T - T_TRUE) < tol_t
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(res.R @ R_TRUE.T)[0])) < tol_r_deg
    assert abs(res.baseline - np.linalg.norm(T_TRUE)) < tol_t
    assert res.horizontal


def test_stereo_calibration_and_rectification(tmp_path):
    rng = np.random.default_rng(3)
    poses = stereo_poses(20, seed=5)
    ptsL = [p[2] + rng.normal(0, 0.1, p[2].shape).astype(np.float32) for p in poses]
    ptsR = [p[3] + rng.normal(0, 0.1, p[3].shape).astype(np.float32) for p in poses]
    ptsR[6] = ptsR[6] + rng.normal(0, 4, ptsR[6].shape).astype(np.float32)  # Outlier pair
    res = sc.calibrate_stereo(ptsL, ptsR, intrinsics(K1, D1, "L"), intrinsics(K2, D2, "R"), COLS, ROWS, SQ, SQ,
                              outlier_ths=2.0, alpha=0.0, log=lambda s: None)
    assert not res.selected[6] and res.n_kept == 19
    assert res.rms < 0.3
    check_result(res)
    assert res.rect_stats()["mean_rms_px"] < 0.2  # Rectified corners on the same rows

    # Remap maps and event LUT are consistent: raw -> rectified (LUT) then rectified -> raw (maps) ~ identity
    maps_l = sc.compute_maps(res, "left")
    lut_l = sc.compute_event_lut(res, "left")
    for (x, y) in [(300, 200), (640, 360), (1000, 550)]:
        u, v = lut_l[y, x]
        back_x = cv2.remap(maps_l[0], np.array([[u]], np.float32), np.array([[v]], np.float32), cv2.INTER_LINEAR)
        back_y = cv2.remap(maps_l[1], np.array([[u]], np.float32), np.array([[v]], np.float32), cv2.INTER_LINEAR)
        assert abs(back_x[0, 0] - x) < 0.05 and abs(back_y[0, 0] - y) < 0.05
    assert np.allclose(lut_l[200, 300], sc.rectify_points(np.array([[300.0, 200.0]]), res.K1, res.d1, res.R1,
                                                          res.P1, res.model)[0], atol=1e-3)

    # I/O
    jp, yp = sc.save_stereo(tmp_path, res, "stereo_L_R", {"note": "test"})
    fs = cv2.FileStorage(str(yp), cv2.FILE_STORAGE_READ)
    assert np.allclose(fs.getNode("T").mat(), res.T) and np.allclose(fs.getNode("P2").mat(), res.P2)
    mp = sc.save_maps(tmp_path / "maps.npz", res)
    with np.load(mp) as z:
        assert z["map1_x"].shape == (H, W) and z["event_lut2"].shape == (H, W, 2)
    assert "Baseline" in res.summary()


def test_fisheye_stereo():
    Df1, Df2 = np.array([[0.05], [-0.01], [0.002], [0.0]]), np.array([[0.04], [-0.02], [0.001], [0.0]])
    poses = stereo_poses(20, seed=9, D=(Df1, Df2), model=sc.FISHEYE)
    res = sc.calibrate_stereo([p[2] for p in poses], [p[3] for p in poses], intrinsics(K1, Df1, "L", sc.FISHEYE),
                              intrinsics(K2, Df2, "R", sc.FISHEYE), COLS, ROWS, SQ, SQ, log=lambda s: None)
    check_result(res)
    assert res.rect_stats()["mean_rms_px"] < 0.2
    assert sc.compute_maps(res, "right")[0].shape == (H, W)


def test_full_pipeline_from_rendered_frames(tmp_path):
    pairs = []
    for rvec, tvec, _, _ in stereo_poses(12, seed=21):
        fl = render(rvec, tvec, K1, D1)
        fr = render(*right_pose(rvec, tvec), K2, D2)
        cl, ok_l = sc.canonicalize_corners(cc.find_chessboard_corners(fl, COLS, ROWS), fl, COLS, ROWS)
        cr, ok_r = sc.canonicalize_corners(cc.find_chessboard_corners(fr, COLS, ROWS), fr, COLS, ROWS)
        assert ok_l and ok_r
        pairs.append(sc.StereoPair(cl, cr, fl, fr))
    res = sc.calibrate_stereo([p.left for p in pairs], [p.right for p in pairs], intrinsics(K1, D1, "L"),
                              intrinsics(K2, D2, "R"), COLS, ROWS, SQ, SQ, outlier_ths=-1, log=lambda s: None)
    check_result(res, tol_t=0.004, tol_r_deg=0.2)
    sc.save_pairs(tmp_path, pairs, (W, H), COLS, ROWS, SQ, SQ, "L", "R")
    size, pattern, loaded, sl, sr = sc.load_pairs(tmp_path)
    assert size == (W, H) and (sl, sr) == ("L", "R") and len(loaded) == len(pairs)
    assert np.array_equal(loaded[0].frame_right, pairs[0].frame_right)
    maps = {s: sc.compute_maps(res, s) for s in ("left", "right")}
    assert sc.draw_pair(loaded[0], COLS, ROWS, (W, H), res, maps, 0).shape == (H, 2 * W, 3)


def test_mixed_models_rejected():
    with pytest.raises(ValueError):
        poses = stereo_poses(4, seed=2)
        sc.calibrate_stereo([p[2] for p in poses], [p[3] for p in poses], intrinsics(K1, D1, "L"),
                            intrinsics(K2, np.zeros((4, 1)), "R", sc.FISHEYE), COLS, ROWS, SQ, SQ, log=lambda s: None)
