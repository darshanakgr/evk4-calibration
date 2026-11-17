#!/usr/bin/env python3
"""
Step 2 - Stereo extrinsics and rectification of two Prophesee event cameras (blinking chessboard).

Prerequisite: the intrinsics of both cameras from step 1 (01_camera_intrincis/output/intrinsics_<serial>.json).

Workflow
  1. Select the LEFT and RIGHT cameras (as seen from behind the rig, looking where the cameras look) and press
     "Open cameras". Check in the live view that the roles are right (otherwise close, "Swap", open again).
     The intrinsics of each camera are loaded automatically from the step 1 output folder.
  2. The blinking chessboard is launched in the browser (F = fullscreen). Enter the real size of one square: it sets
     the scale of the baseline.
  3. Press "Start recording". Place the rig (or the screen) so that BOTH cameras see the whole chessboard and keep
     still ~1-2 s: a stereo pair is captured when both cameras detect the board at the same time (green flash). Vary
     the distance, the tilt and the position of the board in the images. 20-40 pairs is a good target.
  4. Press "Stop recording" (pairs are saved automatically), then "Compute extrinsics + rectification". Check the
     epipolar error, the "Pairs" tab and the rectified live view (horizontal lines must go through the same features
     in both images). Adjust alpha if needed.
  5. Press "Save parameters": stereo_<left>_<right>.json / _opencv.yaml (R, T, E, F, R1, R2, P1, P2, Q, ...) and
     rectification_maps_<left>_<right>.npz (remap maps for frames + per-pixel LUT to rectify events).

Each camera runs in its own process (no GIL contention between the two streams) with the step 1 real-time measures:
sensor trail filter, sensor event rate controller and skipping of buffers if the processing falls behind.

Usage
  python stereo_rectify.py                                   # two connected cameras
  python stereo_rectify.py --left 00050496 --right 00051788
  python stereo_rectify.py --left-input left.raw --right-input right.raw   # replay two RAW recordings
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import camera_process
import stereo_core as sc
from stereo_core import cc

SCRIPT_DIR = Path(__file__).resolve().parent
PATTERN_TEMPLATE = sc.STEP1_DIR / "chessboard_pattern.html"
TEX_W, TEX_H = camera_process.SMALL_W, camera_process.SMALL_H  # Live textures (4 of them)
PAIR_TEX_W, PAIR_TEX_H = 1600, 450  # "Pairs" tab texture (two 16:9 images side by side)
LEFT_W = 460
LOG_H = 130
ROLES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
ORANGE, GREEN, WHITE = (0, 165, 255), (0, 220, 0), (255, 255, 255)
COVERAGE_COLOR = np.array([60, 200, 60], np.float32)


def to_texture(img_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Letterboxes a BGR image into a (th, tw) RGB float32 buffer, as expected by dearpygui raw textures."""
    h, w = img_bgr.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if (nw, nh) != (w, h):
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img_bgr
    return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)).ravel()


def placeholder(text: str, w: int = TEX_W, h: int = TEX_H) -> np.ndarray:
    img = np.full((h, w, 3), 35, np.uint8)
    cc.put_label(img, text, (10, 10), color=(200, 200, 200), bg=(60, 60, 60), scale=0.7)
    return img


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s or "unknown")


def small_maps(maps, image_size):
    """Full resolution remap maps -> maps for the (TEX_W, TEX_H) display images."""
    w, h = image_size
    mx = cv2.resize(maps[0], (TEX_W, TEX_H), interpolation=cv2.INTER_LINEAR)
    my = cv2.resize(maps[1], (TEX_W, TEX_H), interpolation=cv2.INTER_LINEAR)
    return (mx + 0.5) * (TEX_W / w) - 0.5, (my + 0.5) * (TEX_H / h) - 0.5


@dataclass
class CamState:
    role: str
    proc: Optional[object] = None
    cmd_q: Optional[object] = None
    out_q: Optional[object] = None
    stop_evt: Optional[object] = None
    pump: Optional[threading.Thread] = None
    running: bool = False
    serial: str = ""
    size: Optional[tuple] = None
    label: str = ""
    rate: float = 0.0
    lag: float = 0.0
    skipped: float = 0.0
    latest: Optional[sc.Detection] = None
    ev_tex: Optional[np.ndarray] = None
    ev_seq: int = 0
    det_tex: Optional[np.ndarray] = None
    det_seq: int = 0
    flash_until: float = 0.0
    pending: bool = False  # Process created, waiting for its turn to open the camera

    def alive(self) -> bool:
        return self.pending or (self.proc is not None and self.proc.is_alive()) or             (self.pump is not None and self.pump.is_alive())


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.ctx = mp.get_context("spawn")
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._log_lines: List[str] = []
        self._scroll_log = 0

        self.cams = {r: CamState(r) for r in ROLES}
        self.sources: dict = {}  # Dropdown label -> ("camera", serial) or ("file", path)

        # Pattern (taken from the step 1 intrinsics when not given on the command line)
        self.pattern_from_cli = args.cols is not None or args.rows is not None or args.square_size_mm is not None
        self.cols = args.cols or 9
        self.rows = args.rows or 6
        self.square_mm = args.square_size_mm or 20.0

        self.blink = cc.BlinkingConfig(accumulation_time_us=int(args.accumulation_ms * 1000),
                                       min_blinking_pixels=args.min_blink_pixels, ratio_on=args.ratio_on,
                                       ratio_off=args.ratio_off)
        self.collector = sc.StereoPairCollector(skip_time_s=args.skip_time_s, min_shift_px=args.min_shift_px,
                                                sync_window_s=args.sync_window_s, stability_px=args.stability_px)

        # Intrinsics
        self.intr = {r: None for r in ROLES}
        self.intr_override = {r: None for r in ROLES}  # Manually selected files
        self.intr_msg = {r: "" for r in ROLES}
        self._intr_dirty = True

        # Stereo pairs
        self.recording = False
        self.pairs: List[sc.StereoPair] = []
        self.pairs_version = 0
        self.pairs_saved_version = 0
        self.pairs_serials = ("", "")
        self.image_size: Optional[tuple] = None
        self.coverage = {r: np.zeros((TEX_H, TEX_W), np.uint16) for r in ROLES}
        self.session_dir: Optional[Path] = None
        self.raw_index = 0

        # Result
        self.result: Optional[sc.StereoResult] = None
        self.result_pts = None  # (ptsL, ptsR) used for the result
        self.result_pairs_version = -1
        self.result_version = 0
        self.maps = None  # {"left": (mx, my), "right": ...} full resolution
        self.small_maps = None
        self.calc_running = False
        self._alpha_request: Optional[float] = None
        self._alpha_thread: Optional[threading.Thread] = None
        self.show_rectified = False

        self._shown = {"rec": None, "cams": None, "pairs": -1, "result": -1, "locked": None}
        self._seq_shown = {}

    # ------------------------------------------------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_queue.put(line)

    # ------------------------------------------------------------------------------------------------------- cameras
    def refresh_cameras(self, *_):
        sources = {}
        for role, path in (("left", self.args.left_input), ("right", self.args.right_input)):
            if path:
                sources[f"File: {Path(path).name}"] = ("file", path)
        try:
            from metavision_hal import DeviceDiscovery
            for uid in DeviceDiscovery.list():
                serial = uid.split(":")[-1]
                sources[serial] = ("camera", serial)
        except Exception as e:
            self.log(f"Camera discovery failed: {e}")
        self.sources = sources
        labels = list(sources)
        for role in ROLES:
            dpg.configure_item(f"combo_{role}", items=labels)
        cams = [lbl for lbl, (k, _) in sources.items() if k == "camera"]
        wanted = {"left": f"File: {Path(self.args.left_input).name}" if self.args.left_input else self.args.left,
                  "right": f"File: {Path(self.args.right_input).name}" if self.args.right_input else self.args.right}
        for i, role in enumerate(ROLES):
            current = dpg.get_value(f"combo_{role}")
            if wanted[role] in sources:
                dpg.set_value(f"combo_{role}", wanted[role])
            elif current not in sources:
                dpg.set_value(f"combo_{role}", cams[i] if i < len(cams) else "")
        self.log(f"{len(cams)} camera(s) found" + (": " + ", ".join(cams) if cams else "."))
        self._intr_dirty = True
        return sources

    def swap_cameras(self, *_):
        if self.any_camera_alive():
            self.log("Close the cameras before swapping them.")
            return
        left, right = dpg.get_value("combo_left"), dpg.get_value("combo_right")
        dpg.set_value("combo_left", right)
        dpg.set_value("combo_right", left)
        self.intr_override = {"left": self.intr_override["right"], "right": self.intr_override["left"]}
        self._intr_dirty = True

    def any_camera_alive(self) -> bool:
        return any(c.alive() for c in self.cams.values())

    def toggle_cameras(self, *_):
        if self.any_camera_alive():
            self.close_cameras()
        else:
            self.open_cameras()

    def open_cameras(self):
        choice = {r: self.sources.get(dpg.get_value(f"combo_{r}")) for r in ROLES}
        if any(c is None for c in choice.values()):
            self.log("Select a left and a right camera (press 'Refresh' if a camera is missing).")
            return
        if choice["left"] == choice["right"]:
            self.log("The left and right cameras must be different.")
            return
        opts = {"bias_file": self.args.biases, "hw_filters": not self.args.no_hw_filters,
                "erc_mev": self.args.erc_mev, "cols": self.cols, "rows": self.rows,
                "blink": dict(self.blink.__dict__)}
        for role in ROLES:  # Create the processes now so that the UI shows them as "opening"
            kind, value = choice[role]
            st = self.cams[role]
            st.cmd_q, st.out_q, st.stop_evt = self.ctx.Queue(), self.ctx.Queue(maxsize=64), self.ctx.Event()
            st.latest, st.running, st.serial, st.rate, st.lag, st.skipped = None, False, "", 0.0, 0.0, 0.0
            st.proc = self.ctx.Process(target=camera_process.run_camera, name=f"camera_{role}", daemon=True,
                                       args=(role, value if kind == "camera" else "", value if kind == "file" else "",
                                             opts, st.cmd_q, st.out_q, st.stop_evt))
            st.pump = threading.Thread(target=self._pump, args=(role,), name=f"pump_{role}", daemon=True)
            st.pending = True

        def launcher():
            # One camera at a time: two processes opening cameras simultaneously can make one of them fail
            # (each one probes the other camera on the USB bus while looking for its own serial).
            for role in ROLES:
                st = self.cams[role]
                if any(s.stop_evt.is_set() for s in self.cams.values()):
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
        for st in self.cams.values():
            st.pending = False
            if st.stop_evt is not None:
                st.stop_evt.set()

        def reaper(states):
            for st in states:
                if st.proc is not None and st.proc.pid is not None:
                    st.proc.join(timeout=10)  # Killing a process during a USB transfer can wedge the camera
                    if st.proc.is_alive():
                        self.log(f"[{st.role}] camera process did not stop, terminating it.")
                        st.proc.terminate()
        threading.Thread(target=reaper, args=(list(self.cams.values()),), daemon=True).start()

    def send(self, cmd, roles=ROLES):
        for role in roles:
            st = self.cams[role]
            if st.cmd_q is not None and st.alive():
                st.cmd_q.put(cmd)

    # ---------------------------------------------------------------------------------------- messages from cameras
    def _pump(self, role: str) -> None:
        st = self.cams[role]
        q = st.out_q
        while True:
            try:
                msg = q.get(timeout=0.3)
            except queue.Empty:
                if st.proc is None or not st.proc.is_alive():
                    break
                continue
            try:
                kind = msg[0]
                if kind == "events":
                    self._on_events(role, msg[2])
                elif kind == "window":
                    self._on_window(role, *msg[2:])
                elif kind == "stats":
                    st.rate, st.lag, st.skipped = msg[2], msg[3], msg[4]
                elif kind == "log":
                    self.log(f"[{role}] {msg[2]}")
                elif kind == "started":
                    self._on_started(role, *msg[2:])
                elif kind == "stopped":
                    break
            except Exception as e:
                self.log(f"[{role}] UI error: {e}")
                print(traceback.format_exc())
        with self.lock:
            st.running = False
            st.latest = None
            st.rate = st.lag = st.skipped = 0.0
        self.log(f"[{role}] camera closed.")
        if self.recording:
            self.stop_recording()

    def _on_started(self, role, serial, width, height, label):
        st = self.cams[role]
        with self.lock:
            st.running, st.serial, st.size, st.label = True, serial, (width, height), label
            expected = self.pairs_serials[0 if role == "left" else 1]
            if self.pairs and expected and serial and expected != serial:
                # Never mix pairs of different camera setups: keep the previous pairs on disk and start over
                if self.pairs_saved_version != self.pairs_version:
                    self._save_pairs_now()
                old = self.session_dir
                self._reset_pairs()
                self.log(f"Camera setup changed ({role}: {expected} -> {serial}): previous pairs cleared "
                         f"(saved in {old}).")
            self._intr_dirty = True
        self.log(f"[{role}] {label} started ({width}x{height}).")

    def _on_events(self, role: str, img: np.ndarray) -> None:
        with self.lock:
            sm, res = self.small_maps, self.result
            rect = self.show_rectified and sm is not None
        if rect:
            img = cv2.remap(img, sm[role][0], sm[role][1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(60, 60, 60))
            sc.draw_epipolar_lines(img, res.horizontal, step=30)
        tex = to_texture(img, TEX_W, TEX_H)
        st = self.cams[role]
        with self.lock:
            st.ev_tex = tex
            st.ev_seq += 1

    def _on_window(self, role, wall_time, status, corners, frame_small, frame_full) -> None:
        st = self.cams[role]
        other = self.cams[OTHER[role]]
        captured = False
        with self.lock:
            cols, rows = self.cols, self.rows
            if status == "board" and corners is not None and len(corners) == cols * rows:
                corners, resolved = sc.canonicalize_corners(corners, frame_full, cols, rows)
                if not resolved and (cols + rows) % 2 == 1:
                    corners, status = None, "order"  # Could not verify the corner order: do not use it
            else:
                corners = None
            stable = self.collector.is_stable(st.latest, corners, wall_time)
            st.latest = sc.Detection(wall_time, corners, frame_full if corners is not None else None, stable)
            text, color = {"none": ("No blinking pattern", ORANGE),
                           "blink": ("Blinking, chessboard not found", ORANGE),
                           "order": ("Corner order unclear", ORANGE)}.get(status, ("Chessboard detected", GREEN))
            if corners is not None:
                text += "  (stable)" if stable else "  (moving)"
                if self.recording and stable:
                    det = {role: st.latest, OTHER[role]: other.latest}
                    if det["left"] is not None and det["right"] is not None:
                        ok, reason = self.collector.accept(det["left"], det["right"], self.pairs)
                        if ok:
                            self._add_pair(sc.StereoPair(det["left"].corners.copy(), det["right"].corners.copy(),
                                                         det["left"].frame.copy(), det["right"].frame.copy(),
                                                         wall_time))
                            captured = True
                            text, color = f"Pair {len(self.pairs)} captured!", GREEN
                        elif "skip" not in reason:
                            text, color = reason, ORANGE
                    else:
                        text, color = "waiting for the other camera", ORANGE
            coverage = self.coverage[role]
            n_pairs, recording = len(self.pairs), self.recording
            flash = time.time() < st.flash_until
            size = st.size or (1280, 720)
        if captured:
            self.log(f"Stereo pair {n_pairs} captured.")

        img = cv2.cvtColor(frame_small, cv2.COLOR_GRAY2BGR) if frame_small is not None else \
            np.full((TEX_H, TEX_W, 3), 70, np.uint8)
        if n_pairs:
            alpha = (np.minimum(coverage, 4).astype(np.float32) * (0.5 / 4))[..., None]
            img = (img * (1 - alpha) + COVERAGE_COLOR * alpha).astype(np.uint8)
        if corners is not None:
            pts = (corners * np.array([TEX_W / size[0], TEX_H / size[1]], np.float32)).reshape(-1, 1, 2)
            cv2.drawChessboardCorners(img, (cols, rows), pts, True)
            cv2.circle(img, tuple(np.round(pts[0, 0]).astype(int)), 7, (255, 0, 255), 2)  # Corner 0
        if flash:
            cv2.rectangle(img, (0, 0), (TEX_W - 1, TEX_H - 1), (0, 255, 0), 10)
        cc.put_label(img, text, (6, 6), color=color, scale=0.55)
        cc.put_label(img, f"Pairs: {n_pairs}" + ("  [REC]" if recording else ""), (6, 36),
                     color=(80, 80, 255) if recording else WHITE, scale=0.55)
        tex = to_texture(img, TEX_W, TEX_H)
        with self.lock:
            st.det_tex = tex
            st.det_seq += 1

    def _add_pair(self, pair: sc.StereoPair) -> None:  # Lock held
        self.pairs.append(pair)
        self._add_coverage(pair)
        self.pairs_version += 1
        t = time.time() + 0.4
        self.cams["left"].flash_until = self.cams["right"].flash_until = t

    def _add_coverage(self, pair: sc.StereoPair) -> None:  # Lock held
        w, h = self.image_size or (1280, 720)
        scale = np.array([TEX_W / w, TEX_H / h], np.float32)
        cc.add_to_coverage(self.coverage["left"], pair.left * scale)
        cc.add_to_coverage(self.coverage["right"], pair.right * scale)

    def _pairs_changed(self) -> None:  # Lock held
        self.pairs_version += 1
        self.coverage = {r: np.zeros((TEX_H, TEX_W), np.uint16) for r in ROLES}
        for p in self.pairs:
            self._add_coverage(p)

    def _reset_pairs(self) -> None:  # Lock held
        self.pairs = []
        self.pairs_serials = ("", "")
        self.image_size = None
        self.session_dir = None
        self.result = None
        self.result_version += 1
        self.maps = self.small_maps = None
        self._pairs_changed()
        self.pairs_saved_version = self.pairs_version

    # ------------------------------------------------------------------------------------------------------- recording
    def toggle_recording(self, *_):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        with self.lock:
            left, right = self.cams["left"], self.cams["right"]
            if not (left.running and right.running):
                self.log("Cannot record: both cameras must be open.")
                return
            if left.size != right.size:
                self.log(f"Cannot record: different resolutions ({left.size} vs {right.size}).")
                return
            serials = (left.serial, right.serial)
            if self.pairs and self.pairs_serials != serials:
                self.log(f"Cannot record: the current pairs come from {self.pairs_serials}. Clear the pairs first.")
                return
            if (self.cols + self.rows) % 2 == 0:
                self.log(f"Warning: with {self.cols}x{self.rows} inner corners (even sum) the corner order cannot be "
                         "verified between the cameras. Prefer e.g. 9x6 or 8x5.")
            self.pairs_serials = serials
            self.image_size = left.size
            session = self._ensure_session_dir()
            self.collector.reset_timer()
            self.recording = True
        if dpg.get_value("save_raw"):
            self.raw_index += 1
            for role in ROLES:
                path = session / f"{role}_{safe_name(self.cams[role].serial)}_{self.raw_index:03d}.raw"
                self.send(("start_raw", str(path)), (role,))
        self.log("Recording started: keep the rig and the board still ~1-2 s per pose, both cameras must see the "
                 "whole chessboard.")

    def stop_recording(self):
        with self.lock:
            if not self.recording:
                return
            self.recording = False
        self.send(("stop_raw",))
        self.log(f"Recording stopped ({len(self.pairs)} pairs).")
        threading.Thread(target=self._save_pairs_now, daemon=True).start()

    def delete_last_pair(self, *_):
        with self.lock:
            if not self.pairs:
                return
            self.pairs.pop()
            self._pairs_changed()
            n = len(self.pairs)
        self.log(f"Last pair deleted ({n} left).")

    def clear_pairs(self, *_):
        with self.lock:
            if self.recording:
                self.log("Stop the recording before clearing the pairs.")
                return
            self._reset_pairs()
        self.log("All pairs cleared.")

    def load_pairs(self, directory: str):
        if self.recording:
            self.log("Stop the recording before loading pairs.")
            return
        try:
            size, pattern, pairs, sl, sr = sc.load_pairs(Path(directory))
        except Exception as e:
            self.log(f"Failed to load pairs from {directory}: {e}")
            return
        with self.lock:
            self.cols, self.rows = int(pattern["n_cols"]), int(pattern["n_rows"])
            self.square_mm = float(pattern["square_width"]) * 1000
            self.pairs, self.pairs_serials, self.image_size = pairs, (sl, sr), size
            self.session_dir = Path(directory)
            self.result = None
            self.result_version += 1
            self.maps = self.small_maps = None
            self._pairs_changed()
            self.pairs_saved_version = self.pairs_version
            self._intr_dirty = True
        for tag, v in (("cols", self.cols), ("rows", self.rows), ("square_mm", self.square_mm)):
            dpg.set_value(tag, v)
        self.send(("pattern", self.cols, self.rows))
        self.log(f"Loaded {len(pairs)} pairs (left {sl}, right {sr}, {size[0]}x{size[1]}) from {directory}")

    def _ensure_session_dir(self) -> Path:
        with self.lock:
            if self.session_dir is None:
                l_s, r_s = (safe_name(s) for s in self.pairs_serials)
                self.session_dir = self._output_root() / time.strftime(f"stereo_{l_s}_{r_s}_%Y%m%d_%H%M%S")
            self.session_dir.mkdir(parents=True, exist_ok=True)
            return self.session_dir

    def _save_pairs_now(self) -> None:
        with self.lock:
            if not self.pairs:
                return
            pairs, version = list(self.pairs), self.pairs_version
            session = self._ensure_session_dir()
            args = (self.image_size, self.cols, self.rows, self.square_mm / 1000, self.square_mm / 1000,
                    *self.pairs_serials)
        try:
            path = sc.save_pairs(session, pairs, *args)
            with self.lock:
                if self.pairs_version == version:
                    self.pairs_saved_version = version
            self.log(f"{len(pairs)} pairs saved to {path}")
        except Exception as e:
            self.log(f"Failed to save the pairs: {e}")

    def _output_root(self) -> Path:
        value = dpg.get_value("output_dir") if dpg.does_item_exist("output_dir") else self.args.output_dir
        return Path(value or self.args.output_dir)

    # ------------------------------------------------------------------------------------------------------ intrinsics
    def _role_serial(self, role: str) -> str:
        i = 0 if role == "left" else 1
        if self.pairs and self.pairs_serials[i]:
            return self.pairs_serials[i]
        if self.cams[role].serial and self.cams[role].alive():
            return self.cams[role].serial
        src = self.sources.get(dpg.get_value(f"combo_{role}"))
        return src[1] if src and src[0] == "camera" else ""

    def update_intrinsics(self):
        """(Re)loads the intrinsics of both cameras. Main thread only."""
        self._intr_dirty = False
        directory = dpg.get_value("intr_dir")
        for role in ROLES:
            serial = self._role_serial(role)
            path = self.intr_override[role] or (sc.find_intrinsics_file(directory, serial) if serial else None)
            intr, msg = None, ""
            if path is None:
                msg = f"{serial or '?'}: no intrinsics found - run step 1 or select a file"
            else:
                try:
                    intr = sc.load_intrinsics(path)
                    if serial and intr.serial and intr.serial != serial:
                        msg = f"WARNING: file is for {intr.serial}, camera is {serial}\n"
                    msg += f"{intr.serial or serial}: {intr.short()}\n{Path(path).name}"
                except Exception as e:
                    msg = f"Failed to load {path}: {e}"
            changed = (self.intr[role] is None) != (intr is None) or (intr and self.intr[role] and
                                                                      intr.path != self.intr[role].path)
            self.intr[role], self.intr_msg[role] = intr, msg
            dpg.set_value(f"intr_{role}", msg)
            dpg.configure_item(f"intr_{role}", color=(150, 220, 150) if intr else (255, 170, 80))
            if changed and intr:
                self.log(f"Intrinsics ({role}): {intr.short()} [{Path(intr.path).name}]")
        # Pattern from the intrinsics files (same board as step 1) unless given on the command line
        intr = self.intr["left"] or self.intr["right"]
        if intr and intr.pattern and not self.pattern_from_cli and not self.pairs:
            cols, rows = int(intr.pattern.get("n_cols", self.cols)), int(intr.pattern.get("n_rows", self.rows))
            sq = float(intr.pattern.get("square_width", self.square_mm / 1000)) * 1000
            if (cols, rows, round(sq, 3)) != (self.cols, self.rows, round(self.square_mm, 3)):
                self.cols, self.rows, self.square_mm = cols, rows, sq
                for tag, v in (("cols", cols), ("rows", rows), ("square_mm", sq)):
                    dpg.set_value(tag, v)
                self.send(("pattern", cols, rows))
                self.log(f"Pattern taken from the intrinsics: {cols}x{rows}, {sq:.2f} mm squares.")

    def select_intrinsics_file(self, role: str, path: str):
        self.intr_override[role] = path
        self._intr_dirty = True

    # ----------------------------------------------------------------------------------------------------- calibration
    def compute(self, *_):
        with self.lock:
            if self.calc_running:
                return
            if len(self.pairs) < 3:
                self.log(f"At least 3 pairs are needed (got {len(self.pairs)}); 20-40 are recommended.")
                return
            intr_l, intr_r = self.intr["left"], self.intr["right"]
            if intr_l is None or intr_r is None:
                self.log("The intrinsics of both cameras are required (step 1).")
                return
            for intr, serial, role in ((intr_l, self.pairs_serials[0], "left"), (intr_r, self.pairs_serials[1], "right")):
                if intr.serial and serial and intr.serial != serial:
                    self.log(f"The {role} intrinsics are for camera {intr.serial} but the pairs come from {serial}.")
                    return
                if self.image_size and tuple(intr.image_size) != tuple(self.image_size):
                    self.log(f"The {role} intrinsics are for {intr.image_size}, the pairs are {self.image_size}.")
                    return
            ptsL = [p.left for p in self.pairs]
            ptsR = [p.right for p in self.pairs]
            version = self.pairs_version
            cols, rows, sq = self.cols, self.rows, self.square_mm / 1000
            self.calc_running = True
        alpha = float(dpg.get_value("alpha"))
        outlier = float(dpg.get_value("outlier_ths"))
        fix = not dpg.get_value("refine_intr")
        if len(ptsL) < 10:
            self.log(f"Warning: only {len(ptsL)} pairs, the result may be inaccurate (20-40 recommended).")

        def job():
            try:
                res = sc.calibrate_stereo(ptsL, ptsR, intr_l, intr_r, cols, rows, sq, sq, outlier_ths=outlier,
                                          fix_intrinsics=fix, alpha=alpha, log=self.log)
                maps = {s: sc.compute_maps(res, s) for s in ROLES}
                sm = {s: small_maps(maps[s], res.image_size) for s in ROLES}
                with self.lock:
                    self.result, self.result_pts, self.result_pairs_version = res, (ptsL, ptsR), version
                    self.maps, self.small_maps = maps, sm
                    self.result_version += 1
                self.log(f"Done in {res.duration_s:.1f} s: baseline {res.baseline * 1000:.2f} mm, "
                         f"stereo RMS {res.rms:.3f} px, epipolar error {res.rect_stats().get('mean_rms_px', 0):.3f} px")
                for w in res.warnings():
                    self.log("WARNING: " + w)
            except Exception as e:
                self.log(f"Stereo calibration failed: {e}")
                print(traceback.format_exc())
            finally:
                with self.lock:
                    self.calc_running = False

        threading.Thread(target=job, name="stereo_calibration", daemon=True).start()

    def on_alpha(self, sender, value, user_data):
        """Re-rectifies with the new alpha (in a background thread, coalescing quick slider moves)."""
        with self.lock:
            if self.result is None:
                return
            self._alpha_request = float(value)
            if self._alpha_thread is not None and self._alpha_thread.is_alive():
                return
            self._alpha_thread = threading.Thread(target=self._alpha_worker, daemon=True)
            self._alpha_thread.start()

    def _alpha_worker(self):
        while True:
            with self.lock:
                alpha, self._alpha_request = self._alpha_request, None
                base, pts = self.result, self.result_pts
            if alpha is None or base is None:
                return
            res = copy.copy(base)
            sc.compute_rectification(res, alpha)
            res.rect_errors = sc.rectification_errors(res, *pts)
            maps = {s: sc.compute_maps(res, s) for s in ROLES}
            sm = {s: small_maps(maps[s], res.image_size) for s in ROLES}
            with self.lock:
                if self.result is base:
                    self.result, self.maps, self.small_maps = res, maps, sm
                    self.result_version += 1

    def save_parameters(self, *_):
        with self.lock:
            res = self.result
            if res is None:
                self.log("Nothing to save: press 'Compute extrinsics + rectification' first.")
                return
            if self.result_pairs_version != self.pairs_version:
                self.log("The pairs changed since the last computation: compute again before saving.")
                return
            session = self._ensure_session_dir()
            pairs = list(self.pairs)
            pair_args = (self.image_size, self.cols, self.rows, self.square_mm / 1000, self.square_mm / 1000,
                         *self.pairs_serials)
            intr_paths = {r: (self.intr[r].path if self.intr[r] else "") for r in ROLES}
        name = f"stereo_{safe_name(res.serial_left)}_{safe_name(res.serial_right)}"
        maps_name = f"rectification_maps_{safe_name(res.serial_left)}_{safe_name(res.serial_right)}.npz"

        def job():
            try:
                sc.save_pairs(session, pairs, *pair_args)
                meta = {"intrinsics_files": intr_paths, "maps_file": str(session / maps_name),
                        "detection": dict(self.blink.__dict__)}
                json_path, yaml_path = sc.save_stereo(session, res, name, meta)
                sc.save_maps(session / maps_name, res)
                (session / f"{name}_summary.txt").write_text(res.summary() + "\n")
                latest, _ = sc.save_stereo(self._output_root(), res, name, meta)
                self.log(f"Stereo parameters saved to {json_path} (+ {yaml_path.name}, {maps_name})")
                self.log(f"Latest parameters of this pair: {latest} (+ {name}_opencv.yaml)")
            except Exception as e:
                self.log(f"Failed to save the parameters: {e}")
                print(traceback.format_exc())

        threading.Thread(target=job, name="save", daemon=True).start()

    # ------------------------------------------------------------------------------------------------------ misc UI
    def launch_chessboard(self, *_):
        try:
            html = PATTERN_TEMPLATE.read_text(encoding="utf-8")
            html = html.replace("__COLS__", str(self.cols)).replace("__ROWS__", str(self.rows))
            out_dir = self._output_root()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"chessboard_{self.cols}x{self.rows}.html"
            path.write_text(html, encoding="utf-8")
            webbrowser.open(path.resolve().as_uri())
            self.log(f"Chessboard launched in the browser ({self.cols}x{self.rows} inner corners, F = fullscreen).")
        except Exception as e:
            self.log(f"Failed to launch the chessboard: {e}")

    def open_session_folder(self, *_):
        path = self.session_dir or self._output_root()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except AttributeError:
            webbrowser.open(path.resolve().as_uri())

    def on_pattern_changed(self, sender, value, user_data):
        with self.lock:
            if user_data == "square_mm":
                self.square_mm = max(0.1, float(value))
                self.pattern_from_cli = True
                return
            if self.pairs:
                dpg.set_value(sender, self.cols if user_data == "cols" else self.rows)
                self.log("Clear the pairs before changing the pattern geometry.")
                return
            if user_data == "cols":
                self.cols = max(2, int(value))
            else:
                self.rows = max(2, int(value))
            self.pattern_from_cli = True
        self.send(("pattern", self.cols, self.rows))
        self.log(f"Pattern set to {self.cols}x{self.rows} inner corners. Press 'Launch chessboard' to update it.")

    def on_param_changed(self, sender, value, user_data):
        blink_keys = {"acc_ms": ("accumulation_time_us", lambda v: int(max(5, v) * 1000)),
                      "min_px": ("min_blinking_pixels", lambda v: int(max(1, v))),
                      "ratio_on": ("ratio_on", float), "ratio_off": ("ratio_off", float)}
        if user_data in blink_keys:
            key, conv = blink_keys[user_data]
            setattr(self.blink, key, conv(value))
            self.send(("blink", {key: getattr(self.blink, key)}))
        elif user_data == "skip_s":
            self.collector.skip_time_s = max(0.0, float(value))
        elif user_data == "min_shift":
            self.collector.min_shift_px = max(0.0, float(value))
        elif user_data == "sync_s":
            self.collector.sync_window_s = max(0.01, float(value))
        elif user_data == "stability":
            self.collector.stability_px = max(0.1, float(value))

    def on_show_rectified(self, sender, value, user_data):
        with self.lock:
            self.show_rectified = bool(value)

    # ---------------------------------------------------------------------------------------------------------- UI
    def build_ui(self):
        with dpg.texture_registry():
            for role in ROLES:
                dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder(f"{role} camera closed"), TEX_W, TEX_H),
                                    format=dpg.mvFormat_Float_rgb, tag=f"tex_ev_{role}")
                dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder("No detection"), TEX_W, TEX_H),
                                    format=dpg.mvFormat_Float_rgb, tag=f"tex_det_{role}")
            dpg.add_raw_texture(PAIR_TEX_W, PAIR_TEX_H,
                                to_texture(placeholder("No stereo pair yet", 1600, 450), PAIR_TEX_W, PAIR_TEX_H),
                                format=dpg.mvFormat_Float_rgb, tag="tex_pair")

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
        dpg.add_file_dialog(directory_selector=True, tag="dlg_intr_dir", **dlg,
                            callback=lambda s, a: (dpg.set_value("intr_dir", a["file_path_name"]),
                                                   setattr(self, "_intr_dirty", True)))
        dpg.add_file_dialog(directory_selector=True, tag="dlg_load", **dlg,
                            callback=lambda s, a: self.load_pairs(a["file_path_name"]))
        for role in ROLES:
            with dpg.file_dialog(tag=f"dlg_intr_{role}", **dlg, default_path=str(self.args.intrinsics_dir),
                                 callback=lambda s, a, u: self.select_intrinsics_file(u, a["file_path_name"]),
                                 user_data=role):
                dpg.add_file_extension(".json")

        a = self.args
        wrap = LEFT_W - 30
        with dpg.window(tag="main"):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=LEFT_W, tag="left_panel", border=True):
                    with dpg.collapsing_header(label="How to", default_open=False):
                        dpg.add_text(
                            "1. Select LEFT/RIGHT cameras (left = on the left when looking where the cameras look), "
                            "press 'Open cameras' and check the live views.\n"
                            "2. Check that both intrinsics (step 1) are loaded and the square size is right.\n"
                            "3. Start recording. Keep rig and board still ~1-2 s per pose; a pair is captured when "
                            "both cameras see the whole board. Vary distance, tilt and position. 20-40 pairs.\n"
                            "4. Stop, compute, check the epipolar error and the rectified views, then save.",
                            wrap=wrap)

                    with dpg.collapsing_header(label="Cameras", default_open=True):
                        dpg.add_combo([], label="Left", tag="combo_left", width=LEFT_W - 170,
                                      callback=lambda: setattr(self, "_intr_dirty", True))
                        dpg.add_combo([], label="Right", tag="combo_right", width=LEFT_W - 170,
                                      callback=lambda: setattr(self, "_intr_dirty", True))
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Refresh", tag="btn_refresh", callback=lambda: self.refresh_cameras())
                            dpg.add_button(label="Swap left/right", tag="btn_swap", callback=self.swap_cameras)
                        btn = dpg.add_button(label="Open cameras", tag="btn_cameras", width=-1, height=30,
                                             callback=self.toggle_cameras)
                        dpg.bind_item_theme(btn, "theme_action")
                        for role in ROLES:
                            dpg.add_text("", tag=f"cam_status_{role}")

                    with dpg.collapsing_header(label="Intrinsics (step 1)", default_open=True):
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="intr_dir", default_value=str(a.intrinsics_dir), width=LEFT_W - 160,
                                               callback=lambda: setattr(self, "_intr_dirty", True), on_enter=True)
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dlg_intr_dir"))
                            dpg.add_button(label="Reload", callback=lambda: setattr(self, "_intr_dirty", True))
                        for role in ROLES:
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"{role.capitalize():5s}:")
                                dpg.add_button(label="Select file...", callback=lambda s, ap, u: dpg.show_item(u),
                                               user_data=f"dlg_intr_{role}")
                            dpg.add_text("", tag=f"intr_{role}", wrap=wrap)

                    with dpg.collapsing_header(label="Chessboard pattern", default_open=True):
                        dpg.add_input_int(label="Columns (inner corners)", tag="cols", default_value=self.cols,
                                          width=120, callback=self.on_pattern_changed, user_data="cols", on_enter=True)
                        dpg.add_input_int(label="Rows (inner corners)", tag="rows", default_value=self.rows,
                                          width=120, callback=self.on_pattern_changed, user_data="rows", on_enter=True)
                        dpg.add_input_float(label="Square size (mm)", tag="square_mm", default_value=self.square_mm,
                                            width=120, format="%.2f", step=0.5, callback=self.on_pattern_changed,
                                            user_data="square_mm")
                        btn = dpg.add_button(label="Launch chessboard", width=-1, callback=self.launch_chessboard)
                        dpg.bind_item_theme(btn, "theme_action")

                    with dpg.collapsing_header(label="Detection parameters", default_open=False):
                        for label, default, fmt, step, key in (
                                ("Accumulation time (ms)", a.accumulation_ms, "%.0f", 10, "acc_ms"),
                                ("Ratio ON", a.ratio_on, "%.2f", 0.05, "ratio_on"),
                                ("Ratio OFF", a.ratio_off, "%.2f", 0.05, "ratio_off"),
                                ("Skip time between pairs (s)", a.skip_time_s, "%.1f", 0.5, "skip_s"),
                                ("Min. pose change (px)", a.min_shift_px, "%.1f", 1, "min_shift"),
                                ("Sync window (s)", a.sync_window_s, "%.2f", 0.05, "sync_s"),
                                ("Stability threshold (px)", a.stability_px, "%.1f", 0.5, "stability")):
                            dpg.add_input_float(label=label, default_value=default, width=120, format=fmt, step=step,
                                                callback=self.on_param_changed, user_data=key)
                        dpg.add_input_int(label="Min. blinking pixels", default_value=a.min_blink_pixels, width=120,
                                          step=50, callback=self.on_param_changed, user_data="min_px")
                        dpg.add_text("A pair needs a stable detection in both cameras within the sync window.",
                                     wrap=wrap, color=(170, 170, 170))

                    with dpg.collapsing_header(label="Recording", default_open=True):
                        dpg.add_text("Output folder:")
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="output_dir", default_value=str(a.output_dir), width=LEFT_W - 110)
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dlg_output"))
                        dpg.add_checkbox(label="Also save RAW events of both cameras (large files!)", tag="save_raw",
                                         default_value=a.save_raw)
                        dpg.add_button(label="Start recording", tag="btn_record", width=-1, height=42,
                                       callback=self.toggle_recording)
                        dpg.bind_item_theme("btn_record", "theme_rec_idle")
                        dpg.add_text("Stereo pairs: 0", tag="pairs_count")
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Delete last pair", callback=self.delete_last_pair)
                            dpg.add_button(label="Clear all pairs", callback=self.clear_pairs)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Load pairs...", callback=lambda: dpg.show_item("dlg_load"))
                            dpg.add_button(label="Open folder", callback=self.open_session_folder)
                        dpg.add_text("", tag="session_label", wrap=wrap, color=(170, 170, 170))

                    with dpg.collapsing_header(label="Extrinsics + rectification", default_open=True):
                        dpg.add_slider_float(label="Alpha (0 crop, 1 full)", tag="alpha", default_value=a.alpha,
                                             min_value=0.0, max_value=1.0, width=170, format="%.2f",
                                             callback=self.on_alpha)
                        dpg.add_input_float(label="Outlier threshold (std)", tag="outlier_ths",
                                            default_value=a.outlier_ths, width=120, format="%.1f", step=0.5)
                        dpg.add_checkbox(label="Also refine the intrinsics (not recommended)", tag="refine_intr",
                                         default_value=False)
                        with dpg.group(horizontal=True):
                            b1 = dpg.add_button(label="Compute extrinsics + rectification", tag="btn_compute",
                                                width=(LEFT_W - 40) * 3 // 5, height=34, callback=self.compute)
                            b2 = dpg.add_button(label="Save parameters", width=(LEFT_W - 40) * 2 // 5, height=34,
                                                callback=self.save_parameters)
                            dpg.bind_item_theme(b1, "theme_action")
                            dpg.bind_item_theme(b2, "theme_action")
                        dpg.add_input_text(tag="results", multiline=True, readonly=True, width=-1, height=220,
                                           default_value="Not computed yet.")

                with dpg.child_window(tag="right_panel", border=False):
                    with dpg.tab_bar():
                        with dpg.tab(label="Live"):
                            dpg.add_checkbox(label="Rectified view with epipolar lines (after computing)",
                                             tag="show_rect", callback=self.on_show_rectified)
                            for kind, title in (("ev", "events"), ("det", "detection (green = coverage of the pairs)")):
                                with dpg.group(horizontal=True):
                                    for role in ROLES:
                                        with dpg.group():
                                            dpg.add_text(f"{role.capitalize()} {title}", tag=f"lbl_{kind}_{role}")
                                            dpg.add_image(f"tex_{kind}_{role}", tag=f"img_{kind}_{role}",
                                                          width=TEX_W // 2, height=TEX_H // 2)
                        with dpg.tab(label="Pairs"):
                            with dpg.group(horizontal=True):
                                dpg.add_slider_int(label="Pair", tag="pair_slider", min_value=1, max_value=1,
                                                   default_value=1, width=300, callback=lambda: self.render_pair())
                                dpg.add_text("Magenta circle = corner 0 (must be the same physical corner in both)",
                                             color=(170, 170, 170))
                            dpg.add_image("tex_pair", tag="img_pair", width=PAIR_TEX_W // 2, height=PAIR_TEX_H // 2)
                    dpg.add_text("Log")
                    with dpg.child_window(tag="log_child", height=LOG_H, border=True):
                        dpg.add_text("", tag="log_text", wrap=0)

    def layout(self, *_):
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        dpg.configure_item("left_panel", height=vh - 16)
        dpg.configure_item("right_panel", height=vh - 16)
        right_w = max(300, vw - LEFT_W - 36)
        avail_h = max(200, vh - 90 - LOG_H - 40)
        aspect = TEX_W / TEX_H
        iw = int(min((right_w - 16) / 2, (avail_h - 50) / 2 * aspect))
        for kind in ("ev", "det"):
            for role in ROLES:
                dpg.configure_item(f"img_{kind}_{role}", width=iw, height=int(iw / aspect))
        pasp = PAIR_TEX_W / PAIR_TEX_H
        pw = int(min(right_w - 16, (avail_h - 10) * pasp))
        dpg.configure_item("img_pair", width=pw, height=int(pw / pasp))

    def render_pair(self):
        with self.lock:
            n = len(self.pairs)
            if n == 0:
                img = placeholder("No stereo pair yet", 1600, 450)
            else:
                i = min(max(1, dpg.get_value("pair_slider")), n) - 1
                pair = self.pairs[i]
                valid = self.result is not None and self.result_pairs_version == self.pairs_version
                res, maps = (self.result, self.maps) if valid else (None, None)
                size = self.image_size or (pair.frame_left.shape[1], pair.frame_left.shape[0])
                img = sc.draw_pair(pair, self.cols, self.rows, size, res, maps, i)
        dpg.set_value("tex_pair", to_texture(img, PAIR_TEX_W, PAIR_TEX_H))

    def update_ui(self):
        """Called every frame from the main thread."""
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

        if self._intr_dirty:
            self.update_intrinsics()

        # Textures produced by the pump threads
        updates = []
        with self.lock:
            for role, st in self.cams.items():
                for kind, tex, seq in (("ev", st.ev_tex, st.ev_seq), ("det", st.det_tex, st.det_seq)):
                    key = f"{kind}_{role}"
                    if tex is not None and self._seq_shown.get(key) != seq:
                        self._seq_shown[key] = seq
                        updates.append((f"tex_{key}", tex))
            recording, n_pairs = self.recording, len(self.pairs)
            pairs_version, result_version, calc = self.pairs_version, self.result_version, self.calc_running
            session, serials = self.session_dir, self.pairs_serials
        for tag, tex in updates:
            dpg.set_value(tag, tex)

        # Cameras
        alive = {r: self.cams[r].alive() for r in ROLES}
        for role in ROLES:
            st = self.cams[role]
            if st.running:
                txt = (f"{role.capitalize()}: {st.label}  {st.rate / 1e6:.2f} Mev/s  latency {st.lag * 1000:.0f} ms"
                       + (f"  skipping {st.skipped * 100:.0f}%" if st.skipped > 0 else ""))
            else:
                txt = f"{role.capitalize()}: " + ("opening..." if alive[role] else "closed")
            dpg.set_value(f"cam_status_{role}", txt)
        cam_state = tuple(alive.values()) + tuple(self.cams[r].running for r in ROLES)
        if cam_state != self._shown["cams"]:
            self._shown["cams"] = cam_state
            any_alive = any(alive.values())
            dpg.configure_item("btn_cameras", label="Close cameras" if any_alive else "Open cameras")
            for tag in ("combo_left", "combo_right", "btn_refresh", "btn_swap"):
                dpg.configure_item(tag, enabled=not any_alive)
            for role in ROLES:
                if not alive[role]:
                    dpg.set_value(f"tex_ev_{role}", to_texture(placeholder(f"{role} camera closed"), TEX_W, TEX_H))
                lbl = self.cams[role].serial if self.cams[role].running else ""
                dpg.set_value(f"lbl_ev_{role}", f"{role.capitalize()} events {lbl}")

        if recording != self._shown["rec"]:
            self._shown["rec"] = recording
            dpg.configure_item("btn_record", label="Stop recording" if recording else "Start recording")
            dpg.bind_item_theme("btn_record", "theme_rec_active" if recording else "theme_rec_idle")
        locked = n_pairs > 0
        if locked != self._shown["locked"]:
            self._shown["locked"] = locked
            dpg.configure_item("cols", enabled=not locked)
            dpg.configure_item("rows", enabled=not locked)
        dpg.configure_item("btn_compute", label="Computing..." if calc else "Compute extrinsics + rectification")
        dpg.set_value("session_label", f"Session: {session}" if session else "")

        if pairs_version != self._shown["pairs"] or result_version != self._shown["result"]:
            self._shown["pairs"], self._shown["result"] = pairs_version, result_version
            dpg.set_value("pairs_count", f"Stereo pairs: {n_pairs}"
                          + (f"  (left {serials[0]}, right {serials[1]})" if n_pairs and serials[0] else ""))
            dpg.configure_item("pair_slider", max_value=max(1, n_pairs))
            if dpg.get_value("pair_slider") > n_pairs:
                dpg.set_value("pair_slider", max(1, n_pairs))
            with self.lock:
                res = self.result
                valid = res is not None and self.result_pairs_version == self.pairs_version
            if res is None:
                dpg.set_value("results", "Not computed yet.")
            else:
                dpg.set_value("results", ("" if valid else "(OUTDATED: pairs changed since this computation)\n\n")
                              + res.summary())
            self.render_pair()

    def run(self):
        dpg.create_context()
        self.build_ui()
        dpg.create_viewport(title="Stereo extrinsics + rectification (blinking chessboard)", width=1700, height=980)
        dpg.setup_dearpygui()
        dpg.set_global_font_scale(self.args.ui_scale)
        dpg.set_viewport_resize_callback(self.layout)
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        self.layout()

        sources = self.refresh_cameras()
        n_cams = sum(k == "camera" for k, _ in sources.values())
        left_given = self.args.left_input or self.args.left
        right_given = self.args.right_input or self.args.right
        if (left_given and right_given) or (n_cams == 2 and not (left_given or right_given)):
            self.open_cameras()
            if n_cams == 2 and not (self.args.left and self.args.right):
                self.log("Cameras assigned by serial number: check that the LEFT view really is the left camera "
                         "(otherwise close, 'Swap left/right' and open again).")
        elif n_cams > 2:
            self.log("More than 2 cameras: select the left and right ones and press 'Open cameras'.")
        if not self.args.no_browser:
            self.launch_chessboard()

        t_start = time.time()
        try:
            while dpg.is_dearpygui_running():
                self.update_ui()
                dpg.render_dearpygui_frame()
                if self.args.exit_after and time.time() - t_start > self.args.exit_after:
                    break
        finally:
            self.close_cameras()
            for st in self.cams.values():
                if st.proc is not None and st.proc.pid is not None:
                    st.proc.join(timeout=10)
                    if st.proc.is_alive():
                        st.proc.terminate()
            dpg.destroy_context()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo extrinsics and rectification of two event cameras.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--left", default=None, help="Serial number of the left camera.")
    p.add_argument("--right", default=None, help="Serial number of the right camera.")
    p.add_argument("--left-input", default=None, help="RAW file to replay as the left camera.")
    p.add_argument("--right-input", default=None, help="RAW file to replay as the right camera.")
    p.add_argument("--intrinsics-dir", default=str(sc.STEP1_DIR / "output"),
                   help="Output folder of step 1 (intrinsics_<serial>.json).")
    p.add_argument("-o", "--output-dir", default=str(SCRIPT_DIR / "output"), help="Root folder of the sessions.")
    p.add_argument("-b", "--biases", default=None, help="Metavision .bias file applied to both cameras.")
    p.add_argument("--erc-mev", type=float, default=20.0, help="Sensor event rate limit per camera (Mev/s), 0 = off.")
    p.add_argument("--no-hw-filters", action="store_true", help="Do not enable the sensor trail filter / ERC.")
    p.add_argument("--cols", type=int, default=None, help="Inner corners along the width (default: from step 1).")
    p.add_argument("--rows", type=int, default=None, help="Inner corners along the height (default: from step 1).")
    p.add_argument("--square-size-mm", type=float, default=None, help="Square side in mm (default: from step 1).")
    p.add_argument("--accumulation-ms", type=float, default=100.0, help="Blinking detection window (ms).")
    p.add_argument("--min-blink-pixels", type=int, default=100, help="Min. number of blinking pixels.")
    p.add_argument("--ratio-on", type=float, default=0.15, help="Max ratio of ON-only pixels / blinking pixels.")
    p.add_argument("--ratio-off", type=float, default=0.15, help="Max ratio of OFF-only pixels / blinking pixels.")
    p.add_argument("--skip-time-s", type=float, default=2.0, help="Min. time between two captured pairs (s).")
    p.add_argument("--min-shift-px", type=float, default=10.0, help="Min. pose change w.r.t. captured pairs (px).")
    p.add_argument("--sync-window-s", type=float, default=0.3,
                   help="Max. time between the left and right detections of a pair (s).")
    p.add_argument("--stability-px", type=float, default=1.5,
                   help="Max. corner motion between consecutive detections for a view to be considered static (px).")
    p.add_argument("--alpha", type=float, default=0.0, help="Rectification scaling: 0 = crop to valid pixels, 1 = all.")
    p.add_argument("--outlier-ths", type=float, default=2.0, help="Reject pairs with error > mean + ths*std.")
    p.add_argument("--save-raw", action="store_true", help="Enable 'Also save RAW events' by default.")
    p.add_argument("--no-browser", action="store_true", help="Do not launch the chessboard automatically.")
    p.add_argument("--ui-scale", type=float, default=1.0, help="Global font scale of the UI.")
    p.add_argument("--exit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # Automated smoke tests
    return p.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    App(parse_args()).run()
