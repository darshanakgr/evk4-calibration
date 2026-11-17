"""Tests of recorder_core (no camera needed).   Run:  python -m pytest 04_dual_cam_recorder -q"""
import json
import threading

import h5py
import numpy as np
import pytest

import recorder_core as rc

W, H = 1280, 720
DT = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])


def make_events(n, t0, t1, seed=0):
    rng = np.random.default_rng(seed)
    ev = np.empty(n, DT)
    ev["x"], ev["y"] = rng.integers(0, W, n), rng.integers(0, H, n)
    ev["p"] = rng.integers(0, 2, n)
    ev["t"] = np.sort(rng.integers(t0, t1, n))
    return ev


def record(path, ev, t_start, t_stop, seed=1, attrs=None, compression="lzf"):
    """Feeds the writer from another thread in irregular batches, like the camera loop."""
    w = rc.EventFileWriter(path, W, H, t_start, attrs or {"serial": "S", "role": "left"}, compression=compression,
                           block=50_000).start()
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, ev.size), 300, replace=False))

    def producer():
        for part in np.split(ev, cuts):
            w.put(part)
        w.finish(t_stop)

    th = threading.Thread(target=producer)
    th.start()
    th.join()
    assert w.join(30) and w.error is None
    assert w.backlog_events == 0 and w.n_written == int((ev["t"] < t_stop).sum())  # Cut exactly at t_stop
    return w


def test_write_read_roundtrip(tmp_path):
    t_start, t_stop = 1_000_000, 3_500_000
    ev = make_events(400_000, t_start, t_stop)
    record(tmp_path / "cam.h5", ev, t_start, t_stop)
    rd = rc.EventFileReader(tmp_path / "cam.h5")
    assert rd.n == ev.size and rd.serial == "S" and rd.role == "left" and rd.t_stop == t_stop
    assert abs(rd.duration_s - 2.5) < 1e-9
    for name in ("x", "y", "t"):
        assert np.array_equal(rd.ds[name][:], ev[name])
    assert np.array_equal(rd.ds["p"][:], ev["p"].astype(np.uint8))
    # Index: ms_to_idx[k] = #events with t < t_start + k ms
    k = np.arange(len(rd.ms_idx))
    assert np.array_equal(rd.ms_idx, np.searchsorted(ev["t"], t_start + k * 1000, "left"))
    assert rd.ms_idx[-1] == ev.size
    rng = np.random.default_rng(5)
    for _ in range(50):
        a, b = np.sort(rng.integers(t_start - 50_000, t_stop + 50_000, 2))
        got = rd.events_between(int(a), int(b))["t"]
        expected = ev["t"][(ev["t"] >= a) & (ev["t"] < b)]
        assert np.array_equal(got, expected)
    times, rate = rd.rates(10_000)
    assert abs(rate.sum() * 0.01 - ev.size) < 1e-6 and times[1] == pytest.approx(0.01)
    rd.close()


def test_late_stop_is_truncated_exactly(tmp_path):
    """Events queued past t_stop (stop command arriving late) are removed: the file covers [t_start, t_stop)."""
    t_start, t_stop = 0, 600_000
    ev = make_events(200_000, t_start, 900_000, seed=7)  # Goes 300 ms past the stop time
    w = record(tmp_path / "cam.h5", ev, t_start, t_stop)
    kept = ev[ev["t"] < t_stop]
    assert w.n_written == kept.size
    rd = rc.EventFileReader(tmp_path / "cam.h5")
    assert rd.n == kept.size and rd.t_stop == t_stop and np.array_equal(rd.ds["t"][:], kept["t"])
    k = np.arange(len(rd.ms_idx))
    assert len(rd.ms_idx) == 601 and np.array_equal(rd.ms_idx, np.searchsorted(kept["t"], k * 1000, "left"))
    rd.close()


def test_interrupted_recording_index_is_rebuilt(tmp_path):
    t_start = 0
    ev = make_events(100_000, t_start, 900_000, seed=3)
    record(tmp_path / "cam.h5", ev, t_start, 900_000)
    with h5py.File(tmp_path / "cam.h5", "a") as f:  # Simulate a crash before finalization
        del f["index"]
        del f.attrs["t_stop_us"]
    rd = rc.EventFileReader(tmp_path / "cam.h5")
    k = np.arange(len(rd.ms_idx) - 1)
    assert np.array_equal(rd.ms_idx[:-1], np.searchsorted(ev["t"], k * 1000, "left")) and rd.ms_idx[-1] == ev.size
    assert np.array_equal(rd.events_between(100_000, 200_000)["t"], ev["t"][(ev["t"] >= 100_000) & (ev["t"] < 200_000)])
    rd.close()


def test_render_state_polarity_and_lut():
    ev = {"x": np.array([1, 2, 1], np.uint16), "y": np.array([0, 0, 0], np.uint16), "p": np.array([0, 1, 1], np.uint8)}
    s = rc.render_state(ev, 4, 2)
    assert s[0, 1] == 2 and s[0, 2] == 2  # Latest event wins at (1, 0)
    assert rc.render_state(ev, 4, 2, "OFF")[0, 1] == 1 and rc.render_state(ev, 4, 2, "OFF")[0, 2] == 0
    assert rc.render_state(ev, 4, 2, "ON")[0, 1] == 2
    lut = np.full(8, -1, np.int32)
    lut[1], lut[2] = 7, 6  # (1, 0) -> (3, 1), (2, 0) -> (2, 1)
    r = rc.render_state(ev, 4, 2, "BOTH", lut)
    assert r[1, 3] == 2 and r[1, 2] == 2 and r[0].sum() == 0
    assert rc.colorize(r).shape == (2, 4, 3)


def test_render_state_time_bins():
    # Window [1000, 1400) us, 4 bins of 100 us; pixels 0..4 on row 0
    ev = {"x": np.array([0, 1, 2, 3, 4, 0], np.uint16), "y": np.zeros(6, np.uint16),
          "p": np.array([0, 1, 0, 1, 1, 1], np.uint8), "t": np.array([1000, 1099, 1100, 1250, 1399, 1350], np.int64)}
    s = rc.render_state(ev, 5, 1, "BOTH", time_bins=(1000, 400, 4))
    assert s[0].tolist() == [4, 1, 2, 3, 4]  # Polarity ignored; pixel 0: latest event (1350 us, bin 3) wins
    s_on = rc.render_state(ev, 5, 1, "ON", time_bins=(1000, 400, 4))
    assert s_on[0].tolist() == [4, 1, 0, 3, 4]  # ON events only, still colored by bin
    img = rc.colorize(s, time_bins=True)
    assert np.array_equal(img[0, 1], rc.BIN_PALETTE_RGB[0][::-1]) and np.array_equal(img[0, 0], rc.BIN_PALETTE_RGB[3][::-1])
    assert len(rc.BIN_PALETTE_RGB) == rc.MAX_BINS and len({tuple(c) for c in rc.BIN_PALETTE_RGB}) == rc.MAX_BINS
    assert rc.bin_legend(20_000, 10).shape == (26, 960, 3)


def make_recording(folder, serials=("00050496", "00051788")):
    t_start, t_stop = 5_000_000, 6_000_000
    evs = {}
    for i, (role, serial) in enumerate(zip(("left", "right"), serials)):
        ev = make_events(60_000, t_start, t_stop, seed=10 + i)
        record(folder / f"{role}_{serial}.h5", ev, t_start, t_stop, attrs={"serial": serial, "role": role})
        evs[role] = ev
    return evs, t_start, t_stop


def test_frame_plan_example():
    # 5 s recorded, 1000 us accumulation, 10 fps -> 5000 frames, 500 s of replay (0.01x)
    n, replay_s, speed = rc.frame_plan(5_000_000, 1000, 10)
    assert n == 5000 and replay_s == pytest.approx(500) and speed == pytest.approx(0.01)
    assert rc.frame_plan(1_000_000, 30_000, 25)[0] == 34  # Last frame partial: ceil


def test_iter_windows_covers_every_event_once(tmp_path):
    ev = make_events(150_000, 0, 1_000_000, seed=9)
    record(tmp_path / "cam.h5", ev, 0, 1_000_000)
    rd = rc.EventFileReader(tmp_path / "cam.h5")
    acc, n = 7_000, 143
    chunks = list(rc.iter_windows(rd, 0, acc, n, block=10_000))  # Small blocks: many refills
    for k in (0, 5, 77, n - 1):
        expected = ev["t"][(ev["t"] >= k * acc) & (ev["t"] < (k + 1) * acc)]
        assert np.array_equal(chunks[k]["t"], expected)
    assert sum(c["t"].size for c in chunks) == ev.size
    rd.close()


def test_preprocess_and_frames(tmp_path):
    evs, t_start, t_stop = make_recording(tmp_path)
    files = rc.find_recording_files(tmp_path)
    assert set(files) == {"left", "right"}
    params = rc.PreprocessParams(accumulation_us=30_000, polarity="ON")
    progress = []
    summary = rc.preprocess({r: str(p) for r, p in files.items()}, tmp_path / "frames.h5", params,
                            progress=lambda k, n: progress.append((k, n)))
    assert summary["frames"] == 34 and progress[-1] == (34, 34)  # ceil(1 s / 30 ms): frames = duration / acc
    ff = rc.FrameFile(tmp_path / "frames.h5")
    assert ff.n == 34 and ff.times[0] == t_start + 30_000 and ff.params.polarity == "ON"
    assert np.all(np.diff(ff.times) == 30_000)  # Consecutive, non-overlapping windows
    assert ff.counts["left"].sum() == int((evs["left"]["p"] > 0).sum())  # Every ON event in exactly one frame
    assert ff.index_for_time(t_start) == 0 and ff.index_for_time(t_start + 30_000) == 1
    k = 7
    t = ff.times[k]
    sel = evs["left"][(evs["left"]["t"] >= t - 30_000) & (evs["left"]["t"] < t)]
    assert ff.counts["left"][k] == int((sel["p"] > 0).sum())  # Polarity ON: only the events shown
    expected = rc.render_state({"x": sel["x"], "y": sel["y"], "p": sel["p"].astype(np.uint8)}, W, H, "ON")
    assert np.array_equal(ff.frame("left", k), expected)
    assert set(np.unique(ff.frame("right", k))) <= {0, 2}  # ON only
    assert ff.index_for_time(t - 1) == k and ff.index_for_time(t) == k + 1  # Windows are [start, end)
    assert ff.start_time(k) == t - 30_000
    ff.close()


def test_preprocess_time_range(tmp_path):
    evs, t_start, t_stop = make_recording(tmp_path)  # 1 s recording
    files = {r: str(p) for r, p in rc.find_recording_files(tmp_path).items()}
    params = rc.PreprocessParams(accumulation_us=10_000, start_us=250_000, end_us=600_000)
    summary = rc.preprocess(files, tmp_path / "frames.h5", params)
    assert summary["frames"] == 35 and summary["t_begin_us"] == t_start + 250_000  # 350 ms / 10 ms
    ff = rc.FrameFile(tmp_path / "frames.h5")
    assert ff.times[0] == t_start + 260_000 and ff.times[-1] == t_start + 600_000
    inside = evs["left"][(evs["left"]["t"] >= t_start + 250_000) & (evs["left"]["t"] < t_start + 600_000)]
    assert ff.counts["left"].sum() == inside.size  # Only (and all) the events of the range
    ff.close()
    assert rc.trim_range(100, 1100, -5, 5000) == (100, 1100)  # Clipped to the recording
    assert rc.trim_range(100, 1100, 200, -1) == (300, 1100)  # end <= 0: until the end
    with pytest.raises(ValueError):
        rc.preprocess(files, tmp_path / "f2.h5", rc.PreprocessParams(start_us=500_000, end_us=400_000))


def test_preprocess_time_bins(tmp_path):
    evs, t_start, _ = make_recording(tmp_path)
    files = {r: str(p) for r, p in rc.find_recording_files(tmp_path).items()}
    params = rc.PreprocessParams(accumulation_us=50_000, time_bins=True, n_bins=5)
    rc.preprocess(files, tmp_path / "frames.h5", params)
    ff = rc.FrameFile(tmp_path / "frames.h5")
    assert ff.params.time_bins and ff.params.n_bins == 5
    k = 3
    t1 = ff.times[k]
    sel = evs["right"][(evs["right"]["t"] >= t1 - 50_000) & (evs["right"]["t"] < t1)]
    expected = rc.render_state({"x": sel["x"], "y": sel["y"], "p": sel["p"].astype(np.uint8), "t": sel["t"]}, W, H,
                               "BOTH", time_bins=(t1 - 50_000, 50_000, 5))
    frame = ff.frame("right", k)
    assert np.array_equal(frame, expected) and set(np.unique(frame)) == {0, 1, 2, 3, 4, 5}
    # Each value = bin: the event drawn at a pixel belongs to that 10 ms bin
    e = sel[-1]
    assert frame[e["y"], e["x"]] == 1 + (e["t"] - (t1 - 50_000)) // 10_000
    ff.close()


def test_preprocess_with_rectification(tmp_path):
    import test_depth_core  # noqa: F401  (puts the step 2 test rig on the path)
    import stereo_core as sc
    import test_stereo_core as T
    res = sc.StereoResult(model=sc.PINHOLE, image_size=(W, H), serial_left="00050496", serial_right="00051788",
                          K1=T.K1, d1=T.D1, K2=T.K2, d2=T.D2, R=T.R_TRUE, T=T.T_TRUE, E=np.eye(3), F=np.eye(3),
                          rms=0.1, selected=np.ones(3, bool), per_pair_errors=np.zeros(3), rvecs=[], tvecs=[])
    sc.compute_rectification(res, 0.0)
    res.rect_errors = np.zeros(3)
    jp, _ = sc.save_stereo(tmp_path, res, "stereo_00050496_00051788")
    # Roles swapped on purpose: the LUTs must follow the serial numbers
    evs, _, _ = make_recording(tmp_path, serials=("00051788", "00050496"))
    files = {r: str(p) for r, p in rc.find_recording_files(tmp_path).items()}
    params = rc.PreprocessParams(accumulation_us=100_000, rectify=True, stereo_json=str(jp))
    rc.preprocess(files, tmp_path / "frames.h5", params)
    ff = rc.FrameFile(tmp_path / "frames.h5")
    # Role "left" holds camera 00051788 = calibration "right": check one event goes where rectify_points says
    sel = evs["left"][evs["left"]["t"] < ff.times[0]]
    e = sel[-1]
    u, v = sc.rectify_points(np.array([[float(e["x"]), float(e["y"])]]), res.K2, res.d2, res.R2, res.P2, res.model)[0]
    ui, vi = int(round(u)), int(round(v))
    if 0 <= ui < W and 0 <= vi < H:
        assert ff.frame("left", 0)[vi, ui] > 0
    assert json.loads(h5py.File(tmp_path / "frames.h5", "r").attrs["params"])["rectify"]
    ff.close()
