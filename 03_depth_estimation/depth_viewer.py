#!/usr/bin/env python3
"""
Step 3 - Real-time depth estimation with two calibrated Prophesee event cameras.

Prerequisite: the stereo calibration of step 2 (02_stereo_rectification/output/stereo_<left>_<right>.json).

How it works
  * Each camera runs in its own process (same real-time measures as steps 1-2: sensor trail filter, event rate
    controller, skipping of buffers if the processing falls behind). Every event is rectified on the fly with a
    per-pixel look-up table and written into a time surface (recent events bright, older ones fading with tau).
  * Left/right alignment: without a cable, both processes publish their image on the same host-clock ticks (a few ms
    apart). With the SYNC cable ("Hardware sync", one camera master, the other slave), both cameras share the same
    event time base and each frame is cut at the same event-time boundary: pairs are exactly simultaneous.
  * Stereo block matching (OpenCV SGBM/BM) gives the disparity d, and the depth is Z = f * B / d with the rectified
    focal length f and the baseline B of step 2. Depth is only kept where events occurred recently.

Workflow
  1. Select the LEFT/RIGHT cameras (same roles as in step 2) and press "Open cameras". The stereo calibration of the
     pair is loaded automatically.
  2. Move in front of the cameras (event cameras only see changes): depth appears on the edges of moving objects.
     Tune tau, the disparity range and the block size; hover over the depth image to read the depth.
  3. "Start recording" saves a depth sequence (16-bit PNG in mm + time surfaces + frames.csv); "Snapshot" saves the
     current depth map and a colored point cloud (.ply).

Usage
  python depth_viewer.py
  python depth_viewer.py --left 00050496 --right 00051788
  python depth_viewer.py --left 00050496 --right 00051788 --sync hardware --master left
  python depth_viewer.py --left-input left.raw --right-input right.raw    # replay two RAW recordings
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import depth_camera_process
import depth_core as dc
from depth_core import sc
from stereo_rectify import safe_name, to_texture  # Step 2 UI helpers

SCRIPT_DIR = Path(__file__).resolve().parent
STEP2_DIR = SCRIPT_DIR.parent / "02_stereo_rectification"
TEX_W, TEX_H = 640, 360
CBAR_W, CBAR_H = 640, 32
LEFT_W = 470
LOG_H = 120
ROLES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
SCALES = ("1.0", "0.5", "0.25")
SYNC_MODES = ("Host clock (no cable)", "Hardware sync (cable)")


def placeholder(text: str, w: int = TEX_W, h: int = TEX_H) -> np.ndarray:
    img = np.full((h, w, 3), 35, np.uint8)
    sc.cc.put_label(img, text, (10, 10), color=(200, 200, 200), bg=(60, 60, 60), scale=0.7)
    return img


def letterbox_params(img_w: int, img_h: int, tw: int = TEX_W, th: int = TEX_H):
    s = min(tw / img_w, th / img_h)
    return s, (tw - img_w * s) / 2, (th - img_h * s) / 2


@dataclass
class Frame:
    host_time: float
    t_us: int
    image: np.ndarray
    seq: int


@dataclass
class CamState:
    role: str
    proc: Optional[object] = None
    cmd_q: Optional[object] = None
    out_q: Optional[object] = None
    stop_evt: Optional[object] = None
    pump: Optional[threading.Thread] = None
    pending: bool = False
    running: bool = False
    serial: str = ""
    size: Optional[tuple] = None
    label: str = ""
    rate: float = 0.0
    lag: float = 0.0
    skipped: float = 0.0
    frame: Optional[Frame] = None
    recent: deque = field(default_factory=lambda: deque(maxlen=16))  # Recent frames, for matching by event time
    n_frames: int = 0
    sync_role: str = ""  # "master" / "slave" with hardware sync
    tex: Optional[np.ndarray] = None
    tex_seq: int = 0

    def alive(self) -> bool:
        return self.pending or (self.proc is not None and self.proc.is_alive()) or \
            (self.pump is not None and self.pump.is_alive())


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.ctx = mp.get_context("spawn")
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._log_lines: List[str] = []
        self._scroll_log = 0
        self.cams = {r: CamState(r) for r in ROLES}
        self.sources: dict = {}

        # Calibration
        self.calib: Optional[sc.StereoResult] = None
        self.calib_path = ""
        self.calib_override: Optional[str] = None
        self.calib_coords = None  # {"left": (H, W, 2), "right": ...} rectified coordinates of the raw pixels
        self.luts = None  # {"left": index LUT, "right": ...} at self.scale
        self.geom: Optional[dc.DepthGeometry] = None
        self.estimator: Optional[dc.DepthEstimator] = None
        self._calib_dirty = True
        self._calib_loading = False
        self._calib_msg = ""

        # Processing parameters
        self.scale = float(args.scale)
        self.tau_ms = args.tau_ms
        self.fps = args.fps
        self.sync_tol_ms = args.sync_tol_ms
        self.sync_hw = args.sync == "hardware"  # Applied when the cameras are opened
        self.master_role = args.master
        self.session_sync = None  # (hardware?, master role) of the cameras currently open
        self._reaper: Optional[threading.Thread] = None
        self.params = dc.MatcherParams(num_disparities=args.num_disparities, block_size=args.block_size)
        self.show_epilines = False
        self.auto_range = True
        self.z_range = (args.z_min, args.z_max)

        # Depth worker
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self.latest = None  # dict with the last depth result (for the probe and the snapshots)
        self.tex = {"depth": None, "overlay": None, "cbar": None}
        self.tex_seq = 0
        self.stats = {"fps": 0.0, "match_ms": 0.0, "sync_ms": 0.0, "valid": 0.0, "median": float("nan")}

        # Recording
        self.recording = False
        self.recorder: Optional[dc.DepthRecorder] = None
        self.session_dir: Optional[Path] = None
        self.raw_index = 0

        self._shown = {"cams": None, "rec": None, "tex": -1}
        self._seq_shown = {}

    # ------------------------------------------------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_queue.put(line)

    # ------------------------------------------------------------------------------------------------------- cameras
    def refresh_cameras(self, *_):
        sources = {}
        for path in (self.args.left_input, self.args.right_input):
            if path:
                sources[f"File: {Path(path).name}"] = ("file", path)
        try:
            from metavision_hal import DeviceDiscovery
            for uid in DeviceDiscovery.list():
                sources[uid.split(":")[-1]] = ("camera", uid.split(":")[-1])
        except Exception as e:
            self.log(f"Camera discovery failed: {e}")
        self.sources = sources
        cams = [lbl for lbl, (k, _) in sources.items() if k == "camera"]
        wanted = {"left": f"File: {Path(self.args.left_input).name}" if self.args.left_input else self.args.left,
                  "right": f"File: {Path(self.args.right_input).name}" if self.args.right_input else self.args.right}
        for i, role in enumerate(ROLES):
            dpg.configure_item(f"combo_{role}", items=list(sources))
            if wanted[role] in sources:
                dpg.set_value(f"combo_{role}", wanted[role])
            elif dpg.get_value(f"combo_{role}") not in sources:
                dpg.set_value(f"combo_{role}", cams[i] if i < len(cams) else "")
        self.log(f"{len(cams)} camera(s) found" + (": " + ", ".join(cams) if cams else "."))
        self._calib_dirty = True
        return sources

    def swap_cameras(self, *_):
        if self.any_camera_alive():
            self.log("Close the cameras before swapping them.")
            return
        left, right = dpg.get_value("combo_left"), dpg.get_value("combo_right")
        dpg.set_value("combo_left", right)
        dpg.set_value("combo_right", left)
        self.master_role = OTHER[self.master_role]  # Keep the same physical master camera
        dpg.set_value("combo_master", self.master_role)
        self._calib_dirty = True

    def any_camera_alive(self) -> bool:
        return any(c.alive() for c in self.cams.values())

    def toggle_cameras(self, *_):
        if self.any_camera_alive():
            self.close_cameras()
        else:
            self.open_cameras()

    def open_cameras(self):
        choice = {r: self.sources.get(dpg.get_value(f"combo_{r}")) for r in ROLES}
        if any(c is None for c in choice.values()) or choice["left"] == choice["right"]:
            self.log("Select two different cameras (press 'Refresh' if one is missing).")
            return
        hw = self.sync_hw
        if hw and any(choice[r][0] == "file" for r in ROLES):
            self.log("Hardware sync only applies to live cameras: using the host clock for the RAW replay.")
            hw = False
        sync_modes = ({r: ("master" if r == self.master_role else "slave") for r in ROLES} if hw
                      else {r: "standalone" for r in ROLES})
        self.session_sync = (hw, self.master_role)
        with self.lock:
            luts = self.luts
        base = {"bias_file": self.args.biases, "hw_filters": not self.args.no_hw_filters,
                "erc_mev": self.args.erc_mev, "scale": self.scale, "tau_ms": self.tau_ms, "fps": self.fps}
        for role in ROLES:
            kind, value = choice[role]
            st = self.cams[role]
            st.cmd_q, st.out_q, st.stop_evt = self.ctx.Queue(), self.ctx.Queue(maxsize=16), self.ctx.Event()
            st.frame, st.running, st.serial, st.rate, st.lag, st.skipped = None, False, "", 0.0, 0.0, 0.0
            st.recent.clear()
            st.sync_role = sync_modes[role] if hw else ""
            opts = dict(base, lut=luts[role] if luts else None, sync_mode=sync_modes[role], aligned=hw)
            st.proc = self.ctx.Process(target=depth_camera_process.run_camera, name=f"depth_cam_{role}", daemon=True,
                                       args=(role, value if kind == "camera" else "", value if kind == "file" else "",
                                             opts, st.cmd_q, st.out_q, st.stop_evt))
            st.pump = threading.Thread(target=self._pump, args=(role,), name=f"pump_{role}", daemon=True)
            st.pending = True

        # One camera at a time (simultaneous opening can fail on the USB bus). With hardware sync the slave starts
        # first and waits for the master clock (like the SDK metavision_sync sample).
        order = [OTHER[self.master_role], self.master_role] if hw else list(ROLES)

        def launcher():
            for role in order:
                st = self.cams[role]
                if not st.pending or any(s.stop_evt.is_set() for s in self.cams.values()):
                    return
                st.proc.start()
                st.pump.start()
                st.pending = False
                t0 = time.time()
                while not st.running and st.proc.is_alive() and time.time() - t0 < 20:
                    time.sleep(0.05)
        threading.Thread(target=launcher, name="launcher", daemon=True).start()

    def close_cameras(self, *_):
        if self.recording:
            self.stop_recording()
        hw, master = self.session_sync or (False, self.master_role)
        # With hardware sync the slave is stopped first, while the master still provides its clock
        states = [self.cams[OTHER[master]], self.cams[master]] if hw else list(self.cams.values())
        for st in states:
            st.pending = False
            if not hw and st.stop_evt is not None:
                st.stop_evt.set()

        def reaper():
            for st in states:
                if st.stop_evt is not None:
                    st.stop_evt.set()
                if st.proc is not None and st.proc.pid is not None:
                    st.proc.join(timeout=10)  # Killing a process during a USB transfer can wedge the camera
                    if st.proc.is_alive():
                        self.log(f"[{st.role}] camera process did not stop, terminating it.")
                        st.proc.terminate()
        self._reaper = threading.Thread(target=reaper, name="reaper", daemon=True)
        self._reaper.start()

    def send(self, cmd, roles=ROLES):
        for role in roles:
            st = self.cams[role]
            if st.cmd_q is not None and st.alive():
                st.cmd_q.put(cmd)

    def _pump(self, role: str) -> None:
        st = self.cams[role]
        while True:
            try:
                msg = st.out_q.get(timeout=0.3)
            except queue.Empty:
                if st.proc is None or not st.proc.is_alive():
                    break
                continue
            try:
                kind = msg[0]
                if kind == "frame":
                    self._on_frame(role, msg[2], msg[3], msg[4])
                elif kind == "stats":
                    st.rate, st.lag, st.skipped = msg[2], msg[3], msg[4]
                elif kind == "log":
                    self.log(f"[{role}] {msg[2]}")
                elif kind == "started":
                    with self.lock:
                        st.running, st.serial, st.size, st.label = True, msg[2], (msg[3], msg[4]), msg[5]
                        self._calib_dirty = True
                    self.log(f"[{role}] {msg[5]} started ({msg[3]}x{msg[4]}).")
                elif kind == "stopped":
                    break
            except Exception as e:
                self.log(f"[{role}] UI error: {e}")
                print(traceback.format_exc())
        with self.lock:
            st.running, st.frame = False, None
            st.rate = st.lag = st.skipped = 0.0
        self.log(f"[{role}] camera closed.")
        if self.recording:
            self.stop_recording()

    def _on_frame(self, role: str, host_time: float, t_us: int, img: np.ndarray) -> None:
        st = self.cams[role]
        with self.lock:
            st.n_frames += 1
            st.frame = Frame(host_time, t_us, img, st.n_frames)
            st.recent.append(st.frame)
            epi = self.show_epilines and self.calib is not None
            horizontal = self.calib.horizontal if self.calib is not None else True
        self._new_frame.set()
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if epi:
            sc.draw_epipolar_lines(vis, horizontal, step=max(10, int(30 * vis.shape[1] / TEX_W)))
        tex = to_texture(vis, TEX_W, TEX_H)
        with self.lock:
            st.tex, st.tex_seq = tex, st.tex_seq + 1

    # ------------------------------------------------------------------------------------------------------ calibration
    def _role_serial(self, role: str) -> str:
        st = self.cams[role]
        if st.running and st.serial:
            return st.serial
        src = self.sources.get(dpg.get_value(f"combo_{role}"))
        return src[1] if src and src[0] == "camera" else ""

    def update_calibration(self):
        """Finds and loads the stereo calibration of the selected pair (main thread; heavy work in a thread)."""
        self._calib_dirty = False
        if self._calib_loading:
            self._calib_dirty = True  # Try again once the current load is done
            return
        sl, sr = self._role_serial("left"), self._role_serial("right")
        directory = dpg.get_value("calib_dir")
        path = self.calib_override or (sc.find_stereo_file(directory, sl, sr) if sl and sr else None)
        if path is None:
            swapped = sc.find_stereo_file(directory, sr, sl) if sl and sr else None
            msg = (f"Calibration found for the swapped roles (left {sr}, right {sl}): press 'Swap left/right'."
                   if swapped else f"No stereo calibration for left {sl or '?'} / right {sr or '?'} - run step 2 "
                                   "or select a file. Showing the unrectified view, no depth.")
            self._set_calib(None, "", None, msg)
            return
        if str(path) == self.calib_path and self.calib is not None:
            self._send_luts()
            return
        self._calib_loading = True
        dpg.set_value("calib_status", f"Loading {Path(path).name}...")

        def job():
            try:
                res = sc.load_stereo(path)
                coords = {s: sc.compute_event_lut(res, s) for s in ROLES}
                msg = ""
                if res.serial_left and sl and (res.serial_left, res.serial_right) != (sl, sr):
                    msg = f"WARNING: calibration is for {res.serial_left}/{res.serial_right}, cameras are {sl}/{sr}.\n"
                self._set_calib(res, str(path), coords, msg)
            except Exception as e:
                self._set_calib(None, "", None, f"Failed to load {path}: {e}")
            finally:
                self._calib_loading = False

        threading.Thread(target=job, name="calib_loader", daemon=True).start()

    def _set_calib(self, res, path, coords, msg):
        with self.lock:
            self.calib, self.calib_path, self.calib_coords = res, path, coords
            self._calib_msg = msg
            self._apply_scale_locked()
        if res is not None:
            g = self.geom
            self.log(f"Stereo calibration loaded: {Path(path).name} (baseline {g.baseline * 1000:.1f} mm, "
                     f"f = {g.f:.1f} px at scale {self.scale}).")
        elif msg:
            self.log(msg)
        self._send_luts()

    def _apply_scale_locked(self):
        if self.calib is None:
            self.luts = self.geom = self.estimator = None
            return
        size = tuple(self.calib.image_size)
        self.luts = {s: dc.index_lut_from_coords(self.calib_coords[s], size, self.scale) for s in ROLES}
        self.geom = dc.DepthGeometry.from_stereo(self.calib, self.scale)
        self.estimator = dc.DepthEstimator(self.geom, self.params)

    def _send_luts(self):
        with self.lock:
            luts, scale, size = self.luts, self.scale, (tuple(self.calib.image_size) if self.calib else None)
            for role in ROLES:
                st = self.cams[role]
                if luts is not None and st.size is not None and st.size != size:
                    self.log(f"[{role}] camera resolution {st.size} differs from the calibration {size}.")
                    luts = None
        for role in ROLES:
            self.send(("lut", luts[role] if luts else None, None, scale), (role,))

    def calibration_text(self) -> str:
        with self.lock:
            res, g, msg = self.calib, self.geom, self._calib_msg
        if res is None:
            return msg or "No calibration loaded."
        return (msg + f"{Path(self.calib_path).name}\n"
                f"left {res.serial_left}, right {res.serial_right}, {res.model}, {res.image_size[0]}x{res.image_size[1]}\n"
                f"baseline {g.baseline * 1000:.1f} mm, f {g.f:.1f} px @ scale {self.scale}, "
                f"{'horizontal' if g.horizontal else 'vertical'} rig, epipolar error "
                f"{res.rect_stats().get('mean_rms_px', float('nan')):.2f} px")

    # ---------------------------------------------------------------------------------------------------- depth worker
    def _depth_worker(self):
        last_pair, t_prev = None, None
        while not self._stop.is_set():
            if not self._new_frame.wait(0.2):
                continue
            self._new_frame.clear()
            with self.lock:
                est, geom = self.estimator, self.geom
                tol = self.sync_tol_ms / 1000
                hw = bool(self.session_sync and self.session_sync[0])
                if hw:  # Same event-time key in both cameras = exactly the same instant
                    left_keys = {f.t_us: f for f in self.cams["left"].recent}
                    common = [f for f in self.cams["right"].recent if f.t_us in left_keys]
                    fr = max(common, key=lambda f: f.t_us) if common else None
                    fl = left_keys[fr.t_us] if fr is not None else None
                else:
                    fl, fr = self.cams["left"].frame, self.cams["right"].frame
            if fl is None or fr is None or est is None or (fl.seq, fr.seq) == last_pair:
                continue
            if (not hw and abs(fl.host_time - fr.host_time) > tol) or fl.image.shape != fr.image.shape or \
                    fl.image.shape != (geom.height, geom.width):
                continue  # Wait for the frame of the other camera at the same tick
            last_pair = (fl.seq, fr.seq)
            try:
                frame = est.compute(fl.image, fr.image)
                self._publish_depth(fl, fr, frame, geom)
                now = time.time()
                if t_prev is not None:
                    self.stats["fps"] = 0.9 * self.stats["fps"] + 0.1 / max(1e-3, now - t_prev)
                t_prev = now
                self.stats["match_ms"] = 0.9 * self.stats["match_ms"] + 0.1 * frame.match_ms
                self.stats["sync_ms"] = (fl.host_time - fr.host_time) * 1000
                self.stats["valid"] = frame.valid_ratio
                with self.lock:
                    rec = self.recorder if self.recording else None
                if rec is not None:
                    rec.write(fl.host_time, fl.t_us, fr.t_us, self.stats["sync_ms"], fl.image, fr.image, frame)
            except Exception as e:
                self.log(f"Depth error: {e}")
                print(traceback.format_exc())
                time.sleep(0.5)

    def _publish_depth(self, fl: Frame, fr: Frame, frame: dc.DepthFrame, geom: dc.DepthGeometry):
        valid = np.isfinite(frame.depth)
        if self.auto_range:
            r = dc.depth_range_auto(frame.depth)
            if r is not None:  # Smoothed so that the colors do not flicker
                z0, z1 = self.z_range
                self.z_range = (0.8 * z0 + 0.2 * r[0], 0.8 * z1 + 0.2 * r[1])
        z_min, z_max = self.z_range
        z_max = max(z_max, z_min * 1.05)
        color = dc.colorize_depth(frame.depth, z_min, z_max)
        overlay = dc.overlay_depth(fl.image, color, valid)
        self.stats["median"] = float(np.nanmedian(frame.depth)) if valid.any() else float("nan")
        texs = {"depth": to_texture(color, TEX_W, TEX_H), "overlay": to_texture(overlay, TEX_W, TEX_H),
                "cbar": to_texture(dc.colorbar(z_min, z_max, CBAR_W, CBAR_H), CBAR_W, CBAR_H)}
        with self.lock:
            self.latest = {"left": fl, "right": fr, "frame": frame, "geom": geom, "color": color, "overlay": overlay,
                           "z_range": (z_min, z_max)}
            self.tex.update(texs)
            self.tex_seq += 1

    # ------------------------------------------------------------------------------------------------------- recording
    def _metadata(self) -> dict:
        with self.lock:
            meta = dc.depth_metadata(self.calib_path, self.calib, self.geom, self.scale, self.params, self.tau_ms,
                                     self.fps)
        hw, master = self.session_sync or (False, self.master_role)
        meta["timestamp_sync"] = ({"mode": "hardware", "master": master,
                                   "frames": "cut at identical event-time boundaries (left_t_us == right_t_us)"}
                                  if hw else {"mode": "host clock", "tolerance_ms": self.sync_tol_ms})
        return meta

    def _ensure_session_dir(self) -> Path:
        with self.lock:
            if self.session_dir is None:
                sl, sr = (safe_name(self.cams[r].serial or self._role_serial(r)) for r in ROLES)
                root = Path(dpg.get_value("output_dir") or self.args.output_dir)
                self.session_dir = root / time.strftime(f"depth_{sl}_{sr}_%Y%m%d_%H%M%S")
            self.session_dir.mkdir(parents=True, exist_ok=True)
            return self.session_dir

    def toggle_recording(self, *_):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        with self.lock:
            if not all(self.cams[r].running for r in ROLES) or self.calib is None:
                self.log("Cannot record: both cameras must be open and the stereo calibration loaded.")
                return
        session = self._ensure_session_dir()
        seq_dir = session / time.strftime("sequence_%H%M%S")
        rec = dc.DepthRecorder(seq_dir, self._metadata())
        with self.lock:
            self.recorder, self.recording = rec, True
        if dpg.get_value("save_raw"):
            self.raw_index += 1
            for role in ROLES:
                self.send(("start_raw", str(seq_dir / f"{role}_{safe_name(self.cams[role].serial)}.raw")), (role,))
        self.log(f"Recording depth to {seq_dir}")

    def stop_recording(self):
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            rec, self.recorder = self.recorder, None
        self.send(("stop_raw",))
        if rec is not None:
            time.sleep(0.1)  # Let the depth worker finish the frame it may be writing
            rec.close()
            self.log(f"Recording stopped: {rec.n} depth frames in {rec.dir}")

    def snapshot(self, *_):
        with self.lock:
            latest = self.latest
        if latest is None:
            self.log("No depth map yet.")
            return
        session = self._ensure_session_dir()
        meta = self._metadata()

        def job():
            try:
                d = session / time.strftime("snapshot_%H%M%S")
                d.mkdir(parents=True, exist_ok=True)
                frame, geom = latest["frame"], latest["geom"]
                dc.save_depth_png(d / "depth_mm.png", frame.depth)
                cv2.imwrite(str(d / "depth_color.png"), latest["color"])
                cv2.imwrite(str(d / "depth_overlay.png"), latest["overlay"])
                cv2.imwrite(str(d / "left.png"), latest["left"].image)
                cv2.imwrite(str(d / "right.png"), latest["right"].image)
                np.save(d / "disparity.npy", frame.disparity)
                pts, rgb = dc.point_cloud(frame.depth, geom, latest["color"])
                dc.save_ply(d / "point_cloud.ply", pts, rgb)
                (d / "params.json").write_text(json.dumps(meta, indent=2))
                self.log(f"Snapshot saved to {d} ({len(pts)} 3D points)")
            except Exception as e:
                self.log(f"Snapshot failed: {e}")

        threading.Thread(target=job, daemon=True).start()

    def open_session_folder(self, *_):
        path = self.session_dir or Path(dpg.get_value("output_dir") or self.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except AttributeError:
            webbrowser.open(path.resolve().as_uri())

    # ------------------------------------------------------------------------------------------------------ callbacks
    def on_sync_mode(self, sender, value, user_data):
        self.sync_hw = value == SYNC_MODES[1]
        dpg.configure_item("combo_master", enabled=self.sync_hw)
        if self.sync_hw:
            self.log(f"Hardware sync: the {self.master_role} camera will be the MASTER (its SYNC OUT connected to the "
                     "SYNC IN of the other camera). The slave is opened first.")

    def on_scale(self, sender, value, user_data):
        with self.lock:
            self.scale = float(value)
            self._apply_scale_locked()
        self._send_luts()
        self.log(f"Processing scale {self.scale}")

    def on_param(self, sender, value, user_data):
        p = self.params
        if user_data == "tau":
            self.tau_ms = float(value)
            self.send(("tau", self.tau_ms))
        elif user_data == "fps":
            self.fps = float(value)
            self.send(("fps", self.fps))
        elif user_data == "sync":
            self.sync_tol_ms = float(value)
        elif user_data == "num_disp":
            p.num_disparities = max(16, int(round(value / 16)) * 16)
        elif user_data == "block":
            p.block_size = int(value) | 1
        elif user_data in ("uniqueness_ratio", "speckle_window", "speckle_range", "activity_threshold"):
            setattr(p, user_data, int(value))
        elif user_data == "median":
            p.median_ksize = int(value)
        elif user_data == "algorithm":
            p.algorithm = value
        elif user_data == "mask":
            p.mask_to_events = bool(value)
        elif user_data == "epilines":
            self.show_epilines = bool(value)
        elif user_data == "auto_range":
            self.auto_range = bool(value)
        elif user_data == "z_min":
            self.z_range = (max(0.01, float(value)), self.z_range[1])
        elif user_data == "z_max":
            self.z_range = (self.z_range[0], max(0.02, float(value)))

    # ---------------------------------------------------------------------------------------------------------- UI
    def build_ui(self):
        with dpg.texture_registry():
            for tag, text in (("tex_left", "left camera closed"), ("tex_right", "right camera closed"),
                              ("tex_depth", "No depth yet"), ("tex_overlay", "No depth yet")):
                dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder(text), TEX_W, TEX_H),
                                    format=dpg.mvFormat_Float_rgb, tag=tag)
            dpg.add_raw_texture(CBAR_W, CBAR_H, to_texture(dc.colorbar(*self.z_range, CBAR_W, CBAR_H), CBAR_W, CBAR_H),
                                format=dpg.mvFormat_Float_rgb, tag="tex_cbar")
        for tag, colors in (("theme_rec_idle", ((30, 120, 50), (40, 150, 65), (50, 170, 80))),
                            ("theme_rec_active", ((170, 35, 35), (200, 45, 45), (220, 60, 60))),
                            ("theme_action", ((40, 80, 150), (55, 100, 180), (70, 120, 200)))):
            with dpg.theme(tag=tag):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, colors[0])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colors[1])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colors[2])

        dlg = dict(show=False, width=750, height=450)
        dpg.add_file_dialog(directory_selector=True, tag="dlg_output", **dlg,
                            callback=lambda s, a: dpg.set_value("output_dir", a["file_path_name"]))
        dpg.add_file_dialog(directory_selector=True, tag="dlg_calib_dir", **dlg,
                            callback=lambda s, a: (dpg.set_value("calib_dir", a["file_path_name"]),
                                                   setattr(self, "_calib_dirty", True)))
        with dpg.file_dialog(tag="dlg_calib_file", **dlg, default_path=str(self.args.calib_dir),
                             callback=lambda s, a: (setattr(self, "calib_override", a["file_path_name"]),
                                                    setattr(self, "_calib_dirty", True))):
            dpg.add_file_extension(".json")

        a, p = self.args, self.params
        wrap = LEFT_W - 30
        mark_dirty = lambda: setattr(self, "_calib_dirty", True)  # noqa: E731
        with dpg.window(tag="main"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=LEFT_W, tag="left_panel", border=True):
                    with dpg.collapsing_header(label="How to", default_open=False):
                        dpg.add_text("Event cameras only see changes: depth appears on the edges of moving objects "
                                     "(or when the rig moves). Hover over the depth image to read the depth. Increase "
                                     "'Disparity range' for close objects (min. depth shown below), increase tau or the "
                                     "block size if the depth is too sparse/noisy.", wrap=wrap)

                    with dpg.collapsing_header(label="Cameras", default_open=True):
                        dpg.add_combo([], label="Left", tag="combo_left", width=LEFT_W - 170, callback=mark_dirty)
                        dpg.add_combo([], label="Right", tag="combo_right", width=LEFT_W - 170, callback=mark_dirty)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Refresh", tag="btn_refresh", callback=lambda: self.refresh_cameras())
                            dpg.add_button(label="Swap left/right", tag="btn_swap", callback=self.swap_cameras)
                        dpg.add_combo(SYNC_MODES, label="Timestamp sync", tag="combo_sync", width=LEFT_W - 170,
                                      default_value=SYNC_MODES[1] if self.sync_hw else SYNC_MODES[0],
                                      callback=self.on_sync_mode)
                        dpg.add_combo(ROLES, label="Master camera", tag="combo_master", width=LEFT_W - 170,
                                      default_value=self.master_role, enabled=self.sync_hw,
                                      callback=lambda s, v: setattr(self, "master_role", v))
                        btn = dpg.add_button(label="Open cameras", tag="btn_cameras", width=-1, height=30,
                                             callback=self.toggle_cameras)
                        dpg.bind_item_theme(btn, "theme_action")
                        for role in ROLES:
                            dpg.add_text("", tag=f"cam_status_{role}")

                    with dpg.collapsing_header(label="Stereo calibration (step 2)", default_open=True):
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="calib_dir", default_value=str(a.calib_dir), width=LEFT_W - 160,
                                               callback=mark_dirty, on_enter=True)
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dlg_calib_dir"))
                            dpg.add_button(label="Reload", callback=mark_dirty)
                        dpg.add_button(label="Select calibration file...",
                                       callback=lambda: dpg.show_item("dlg_calib_file"))
                        dpg.add_text("", tag="calib_status", wrap=wrap)

                    with dpg.collapsing_header(label="Event frames", default_open=True):
                        dpg.add_slider_float(label="Decay tau (ms)", default_value=self.tau_ms, min_value=5,
                                             max_value=200, width=200, format="%.0f", callback=self.on_param,
                                             user_data="tau")
                        dpg.add_slider_float(label="Frame rate (Hz)", default_value=self.fps, min_value=5,
                                             max_value=60, width=200, format="%.0f", callback=self.on_param,
                                             user_data="fps")
                        dpg.add_combo(SCALES, label="Processing scale", default_value=str(self.scale), width=200,
                                      callback=self.on_scale)
                        dpg.add_input_float(label="Sync tolerance (ms)", default_value=self.sync_tol_ms, width=120,
                                            format="%.0f", step=5, callback=self.on_param, user_data="sync")
                        dpg.add_checkbox(label="Show epipolar lines on the rectified views", callback=self.on_param,
                                         user_data="epilines")

                    with dpg.collapsing_header(label="Stereo matching", default_open=True):
                        dpg.add_combo(dc.ALGORITHMS, label="Algorithm", default_value=p.algorithm, width=200,
                                      callback=self.on_param, user_data="algorithm")
                        dpg.add_slider_int(label="Disparity range (px)", default_value=p.num_disparities,
                                           min_value=16, max_value=256, width=200, callback=self.on_param,
                                           user_data="num_disp")
                        dpg.add_slider_int(label="Block size", default_value=p.block_size, min_value=3, max_value=21,
                                           width=200, callback=self.on_param, user_data="block")
                        for label, key, lo, hi in (("Uniqueness ratio", "uniqueness_ratio", 0, 40),
                                                   ("Speckle window", "speckle_window", 0, 300),
                                                   ("Speckle range", "speckle_range", 0, 16),
                                                   ("Min. event activity", "activity_threshold", 0, 255)):
                            dpg.add_slider_int(label=label, default_value=getattr(p, key), min_value=lo, max_value=hi,
                                               width=200, callback=self.on_param, user_data=key)
                        dpg.add_combo(("0", "3", "5"), label="Median filter", default_value=str(p.median_ksize),
                                      width=200, callback=lambda s, v: self.on_param(s, int(v), "median"))
                        dpg.add_checkbox(label="Depth only where events occurred", default_value=p.mask_to_events,
                                         callback=self.on_param, user_data="mask")
                        dpg.add_text("", tag="range_info", wrap=wrap, color=(170, 170, 170))

                    with dpg.collapsing_header(label="Depth display", default_open=False):
                        dpg.add_checkbox(label="Automatic range", default_value=True, callback=self.on_param,
                                         user_data="auto_range")
                        dpg.add_input_float(label="Near (m)", tag="z_min", default_value=self.z_range[0], width=120,
                                            format="%.2f", step=0.1, callback=self.on_param, user_data="z_min")
                        dpg.add_input_float(label="Far (m)", tag="z_max", default_value=self.z_range[1], width=120,
                                            format="%.2f", step=0.1, callback=self.on_param, user_data="z_max")

                    with dpg.collapsing_header(label="Recording", default_open=True):
                        dpg.add_text("Output folder:")
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="output_dir", default_value=str(a.output_dir), width=LEFT_W - 110)
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dlg_output"))
                        dpg.add_checkbox(label="Also save RAW events of both cameras (large files!)", tag="save_raw",
                                         default_value=a.save_raw)
                        dpg.add_button(label="Start recording", tag="btn_record", width=-1, height=40,
                                       callback=self.toggle_recording)
                        dpg.bind_item_theme("btn_record", "theme_rec_idle")
                        with dpg.group(horizontal=True):
                            b = dpg.add_button(label="Snapshot (+ point cloud)", callback=self.snapshot)
                            dpg.bind_item_theme(b, "theme_action")
                            dpg.add_button(label="Open folder", callback=self.open_session_folder)
                        dpg.add_text("", tag="rec_status", wrap=wrap, color=(170, 170, 170))

                with dpg.child_window(tag="right_panel", border=False):
                    dpg.add_text("", tag="stats")
                    with dpg.group(horizontal=True):
                        for role in ROLES:
                            with dpg.group():
                                dpg.add_text(f"{role.capitalize()} rectified time surface", tag=f"lbl_{role}")
                                dpg.add_image(f"tex_{role}", tag=f"img_{role}", width=TEX_W // 2, height=TEX_H // 2)
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("Depth (near = red, far = blue)")
                            dpg.add_image("tex_depth", tag="img_depth", width=TEX_W // 2, height=TEX_H // 2)
                        with dpg.group():
                            dpg.add_text("Depth over the left view")
                            dpg.add_image("tex_overlay", tag="img_overlay", width=TEX_W // 2, height=TEX_H // 2)
                    with dpg.group(horizontal=True):
                        dpg.add_image("tex_cbar", tag="img_cbar", width=CBAR_W // 2, height=CBAR_H)
                        dpg.add_text("Hover over a depth image to read the depth", tag="probe")
                    dpg.add_text("Log")
                    with dpg.child_window(tag="log_child", height=LOG_H, border=True):
                        dpg.add_text("", tag="log_text", wrap=0)

    def layout(self, *_):
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        dpg.configure_item("left_panel", height=vh - 16)
        dpg.configure_item("right_panel", height=vh - 16)
        right_w = max(300, vw - LEFT_W - 36)
        avail_h = max(200, vh - 110 - LOG_H - 40 - CBAR_H)
        aspect = TEX_W / TEX_H
        iw = int(min((right_w - 16) / 2, (avail_h - 50) / 2 * aspect))
        for tag in ("img_left", "img_right", "img_depth", "img_overlay"):
            dpg.configure_item(tag, width=iw, height=int(iw / aspect))
        dpg.configure_item("img_cbar", width=iw, height=CBAR_H)

    def update_probe(self):
        with self.lock:
            latest = self.latest
        if latest is None:
            return
        for tag in ("img_depth", "img_overlay"):
            if not dpg.is_item_hovered(tag):
                continue
            mx, my = dpg.get_mouse_pos(local=False)
            x0, y0 = dpg.get_item_rect_min(tag)
            w, h = dpg.get_item_rect_size(tag)
            depth, geom = latest["frame"].depth, latest["geom"]
            s, ox, oy = letterbox_params(depth.shape[1], depth.shape[0])
            u = int(((mx - x0) / max(w, 1) * TEX_W - ox) / s)
            v = int(((my - y0) / max(h, 1) * TEX_H - oy) / s)
            if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                # Median over a small neighbourhood: depth is sparse (edges only)
                patch = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
                z = float(np.nanmedian(patch)) if np.isfinite(patch).any() else float("nan")
                if np.isfinite(z):
                    X, Y = (u - geom.cx) * z / geom.f, (v - geom.cy) * z / geom.f
                    dpg.set_value("probe", f"({u}, {v}):  Z = {z:.3f} m   X = {X:+.3f} m  Y = {Y:+.3f} m   "
                                           f"(disparity {geom.f * geom.baseline / z:.1f} px)")
                else:
                    dpg.set_value("probe", f"({u}, {v}): no depth here")
            return

    def update_ui(self):
        new = False
        while True:
            try:
                self._log_lines.append(self._log_queue.get_nowait())
                new = True
            except queue.Empty:
                break
        if new:
            self._log_lines = self._log_lines[-300:]
            dpg.set_value("log_text", "\n".join(self._log_lines))
            self._scroll_log = 3
        if self._scroll_log:
            dpg.set_y_scroll("log_child", dpg.get_y_scroll_max("log_child"))
            self._scroll_log -= 1

        if self._calib_dirty:
            self.update_calibration()
        if not self._calib_loading:
            dpg.set_value("calib_status", self.calibration_text())
            dpg.configure_item("calib_status", color=(150, 220, 150) if self.calib is not None else (255, 170, 80))

        updates = []
        with self.lock:
            for role in ROLES:
                st = self.cams[role]
                if st.tex is not None and self._seq_shown.get(role) != st.tex_seq:
                    self._seq_shown[role] = st.tex_seq
                    updates.append((f"tex_{role}", st.tex))
            if self.tex_seq != self._shown["tex"]:
                self._shown["tex"] = self.tex_seq
                updates += [(f"tex_{k}", v) for k, v in self.tex.items() if v is not None]
            geom, recording, rec = self.geom, self.recording, self.recorder
        for tag, tex in updates:
            dpg.set_value(tag, tex)

        alive = {r: self.cams[r].alive() for r in ROLES}
        for role in ROLES:
            st = self.cams[role]
            if st.running:
                txt = (f"{role.capitalize()}: {st.label}" + (f" [{st.sync_role}]" if st.sync_role else "")
                       + f"  {st.rate / 1e6:.2f} Mev/s  latency {st.lag * 1000:.0f} ms"
                       + (f"  skipping {st.skipped * 100:.0f}%" if st.skipped > 0 else ""))
            else:
                txt = f"{role.capitalize()}: " + ("opening..." if alive[role] else "closed")
            dpg.set_value(f"cam_status_{role}", txt)
        state = tuple(alive.values())
        if state != self._shown["cams"]:
            self._shown["cams"] = state
            any_alive = any(state)
            dpg.configure_item("btn_cameras", label="Close cameras" if any_alive else "Open cameras")
            for tag in ("combo_left", "combo_right", "btn_refresh", "btn_swap", "combo_sync"):
                dpg.configure_item(tag, enabled=not any_alive)
            dpg.configure_item("combo_master", enabled=not any_alive and self.sync_hw)
            for role in ROLES:
                if not alive[role]:
                    dpg.set_value(f"tex_{role}", to_texture(placeholder(f"{role} camera closed"), TEX_W, TEX_H))

        s = self.stats
        if any(self.cams[r].running for r in ROLES):
            hw = bool(self.session_sync and self.session_sync[0])
            sync = ("hardware sync (pairs at identical event times), arrival offset" if hw else
                    "host clock, left/right offset")
            dpg.set_value("stats", f"Depth {s['fps']:.1f} fps   matching {s['match_ms']:.1f} ms   {sync} "
                                   f"{s['sync_ms']:+.1f} ms   valid {s['valid'] * 100:.0f}% of the active pixels   "
                                   f"median depth {s['median']:.2f} m")
        if geom is not None:
            nd = max(16, int(round(self.params.num_disparities / 16)) * 16)
            z_min = geom.depth_for_disparity(nd)
            # dZ = Z^2 / (f B) per pixel of disparity
            dpg.set_value("range_info", f"Measurable depth: Z >= {z_min:.2f} m (disparity range {nd} px). "
                                        f"Depth step per disparity pixel: {100 / (geom.f * geom.baseline):.1f} cm at "
                                        f"1 m, {400 / (geom.f * geom.baseline):.1f} cm at 2 m.")
        if self.auto_range:
            dpg.set_value("z_min", round(self.z_range[0], 2))
            dpg.set_value("z_max", round(self.z_range[1], 2))
        if recording != self._shown["rec"]:
            self._shown["rec"] = recording
            dpg.configure_item("btn_record", label="Stop recording" if recording else "Start recording")
            dpg.bind_item_theme("btn_record", "theme_rec_active" if recording else "theme_rec_idle")
        dpg.set_value("rec_status", f"Recording: {rec.n} frames -> {rec.dir}" if recording and rec else
                      (f"Session: {self.session_dir}" if self.session_dir else ""))
        self.update_probe()

    def run(self):
        dpg.create_context()
        self.build_ui()
        dpg.create_viewport(title="Event-based stereo depth", width=1700, height=980)
        dpg.setup_dearpygui()
        dpg.set_global_font_scale(self.args.ui_scale)
        dpg.set_viewport_resize_callback(self.layout)
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        self.layout()

        self._worker = threading.Thread(target=self._depth_worker, name="depth", daemon=True)
        self._worker.start()
        sources = self.refresh_cameras()
        n_cams = sum(k == "camera" for k, _ in sources.values())
        left_given = self.args.left_input or self.args.left
        right_given = self.args.right_input or self.args.right
        if (left_given and right_given) or (n_cams == 2 and not (left_given or right_given)):
            self.update_calibration()  # Start loading the calibration before the cameras open
            self.open_cameras()
        elif n_cams > 2:
            self.log("More than 2 cameras: select the left and right ones and press 'Open cameras'.")

        t_start = time.time()
        try:
            while dpg.is_dearpygui_running():
                self.update_ui()
                dpg.render_dearpygui_frame()
                if self.args.exit_after and time.time() - t_start > self.args.exit_after:
                    break
        finally:
            self.close_cameras()
            self._stop.set()
            if self._reaper is not None:
                self._reaper.join(timeout=25)
            for st in self.cams.values():
                if st.proc is not None and st.proc.pid is not None:
                    st.proc.join(timeout=10)
                    if st.proc.is_alive():
                        st.proc.terminate()
            dpg.destroy_context()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time stereo depth from two calibrated event cameras.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--left", default=None, help="Serial number of the left camera.")
    p.add_argument("--right", default=None, help="Serial number of the right camera.")
    p.add_argument("--left-input", default=None, help="RAW file to replay as the left camera.")
    p.add_argument("--right-input", default=None, help="RAW file to replay as the right camera.")
    p.add_argument("--calib-dir", default=str(STEP2_DIR / "output"),
                   help="Output folder of step 2 (stereo_<left>_<right>.json).")
    p.add_argument("-o", "--output-dir", default=str(SCRIPT_DIR / "output"), help="Folder of the recordings.")
    p.add_argument("-b", "--biases", default=None, help="Metavision .bias file applied to both cameras.")
    p.add_argument("--erc-mev", type=float, default=20.0, help="Sensor event rate limit per camera (Mev/s), 0 = off.")
    p.add_argument("--no-hw-filters", action="store_true", help="Do not enable the sensor trail filter / ERC.")
    p.add_argument("--sync", choices=("host", "hardware"), default="host",
                   help="Left/right alignment: host clock (no cable) or hardware sync (SYNC cable, master/slave).")
    p.add_argument("--master", choices=ROLES, default="left", help="Master camera role with --sync hardware.")
    p.add_argument("--scale", choices=SCALES, default="0.5", help="Processing scale of the rectified images.")
    p.add_argument("--tau-ms", type=float, default=30.0, help="Decay constant of the time surfaces (ms).")
    p.add_argument("--fps", type=float, default=20.0, help="Depth frame rate (Hz).")
    p.add_argument("--sync-tol-ms", type=float, default=25.0, help="Max. host-time offset of a left/right pair (ms).")
    p.add_argument("--num-disparities", type=int, default=64, help="Disparity search range (px, multiple of 16).")
    p.add_argument("--block-size", type=int, default=7, help="Matching block size (odd).")
    p.add_argument("--z-min", type=float, default=0.3, help="Initial near limit of the depth colormap (m).")
    p.add_argument("--z-max", type=float, default=3.0, help="Initial far limit of the depth colormap (m).")
    p.add_argument("--save-raw", action="store_true", help="Enable 'Also save RAW events' by default.")
    p.add_argument("--ui-scale", type=float, default=1.0, help="Global font scale of the UI.")
    p.add_argument("--exit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # Automated smoke tests
    return p.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    App(parse_args()).run()
