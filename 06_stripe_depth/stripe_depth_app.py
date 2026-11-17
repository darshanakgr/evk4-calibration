#!/usr/bin/env python3
"""
Step 6 - Laser-stripe stereo depth with two hardware-synchronised event cameras (method C: per-event time matching).

Record tab (same as step 4)
  * Two cameras, each in its own process, hardware timestamp sync (master/slave through the SYNC cable) or host clock.
  * Live preview (not rectified). Raw events are saved to one compressed HDF5 file per camera; with hardware sync both
    files cover exactly the same time interval.

Replay tab (depth)
  * Load a recording and the stereo calibration of the pair (step 2, found automatically by serial numbers).
  * Sweep segmentation: the laser sweeps are the bursts of ON events in the left camera (1 ms bins above the background
    by n_sigma robust sigma, merged and padded). They are listed and drawn on the timeline; click the timeline or the
    list to select one. Only the time range between the green and red lines is searched.
  * Depth (method C): for the selected sweep (or all sweeps), in a separate process:
      - first ON event of each pixel that passes the neighbour filter (= arrival of the stripe), rectified;
      - for every left event, the right stripe position at the same instant on the same row, from a local linear fit
        of x_R against t_R over +-tau (px of stripe motion) and rows +-h; disparity d = x_L - x_R;
      - depth image: robust k x k aggregation (median, mean of the values within +-max_dev, >= min_support values).
    Results (depth .npz, point cloud .ply coloured by depth, stats .json) are written to the output folder and reused
    when the same sweep is selected again with the same parameters.
  * Two views (choose: arrival time left/right, depth per event, aggregated depth, distance from a fitted plane,
    support), hover for the values at a pixel, statistics, and "Open in Open3D" for the point cloud (separate
    process: cloud_viewer.py; needs the open3d package in the Python given by --viewer-python).

Usage
  python stripe_depth_app.py                                   # two connected cameras, hardware sync, left = master
  python stripe_depth_app.py --replay ../04_dual_cam_recorder/recordings/cw/20us
  python stripe_depth_app.py --viewer-python C:/path/to/python_with_open3d.exe
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import recorder_camera_process
import recorder_core as rc  # Puts the step 1-3 folders on sys.path
import stereo_core as sc
import stripe_depth_core as sd
from stereo_rectify import safe_name, to_texture

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
STEP2_DIR = ROOT / "02_stereo_rectification"
STEP4_REC = ROOT / "04_dual_cam_recorder" / "recordings"
TEX_W, TEX_H = 640, 360
IMG_W, IMG_H = 1280, 720  # Depth view textures (EVK4 resolution; other sizes are letterboxed)
LEG_W, LEG_H = 640, 22
LEFT_W = 450
LOG_H = 110
TIMELINE_H = 190
ROLES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
SYNC_MODES = ("Hardware sync (cable)", "Host clock (no cable)")
SERIES_COLORS = {"left": (255, 160, 40), "right": (60, 200, 255), "on": (255, 255, 255)}  # RGB
VIEWS = ("Arrival time (left)", "Arrival time (right)", "Depth per event (before aggregation)", "Depth (aggregated)",
         "Distance from fitted plane", "Support (agreeing values)")
CLOUD_COLORS = ("depth (file colours)", "depth", "height", "plane")
BG = 35  # Background gray of empty pixels


def placeholder(text: str, w: int = TEX_W, h: int = TEX_H) -> np.ndarray:
    img = np.full((h, w, 3), BG, np.uint8)
    sc.cc.put_label(img, text, (10, 10), color=(200, 200, 200), bg=(60, 60, 60), scale=0.7)
    return img


def fmt_count(n: float) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.1f} k" if n >= 1e3 else f"{n:.0f}"


# ----------------------------------------------------------------------------------------------------------------------
# Colour maps (BGR)
# ----------------------------------------------------------------------------------------------------------------------
def _lut_from_anchors(anchors_rgb) -> np.ndarray:
    a = np.asarray(anchors_rgb, np.float64)
    u = np.linspace(0, len(a) - 1, 256)
    i = np.minimum(u.astype(int), len(a) - 2)
    f = (u - i)[:, None]
    rgb = a[i] * (1 - f) + a[i + 1] * f
    return (rgb[:, ::-1] * 255).astype(np.uint8)  # BGR


_CV_LUT = {name: cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cm).reshape(256, 3)
           for name, cm in (("turbo", cv2.COLORMAP_TURBO), ("viridis", cv2.COLORMAP_VIRIDIS),
                            ("inferno", cv2.COLORMAP_INFERNO))}
_CV_LUT["diverging"] = _lut_from_anchors([[0.02, 0.19, 0.38], [0.26, 0.58, 0.76], [0.97, 0.97, 0.97],
                                          [0.84, 0.38, 0.30], [0.40, 0.0, 0.12]])


def colorize(img: np.ndarray, lo: float, hi: float, cmap: str) -> np.ndarray:
    """BGR uint8 image of a float image (NaN = background). lo maps to the start of the colour map."""
    ok = np.isfinite(img)
    u = np.zeros(img.shape, np.uint8)
    u[ok] = np.clip((img[ok] - lo) / (hi - lo if hi != lo else 1e-9) * 255, 0, 255).astype(np.uint8)
    out = _CV_LUT[cmap][u]
    out[~ok] = BG
    return out


def legend(lo: float, hi: float, cmap: str, unit: str, left_note: str = "", right_note: str = "",
           width: int = LEG_W) -> np.ndarray:
    img = np.full((LEG_H, width, 3), BG, np.uint8)
    margin = 150 if width <= LEG_W else 200  # Room for "577.3 mm (far)" / "544.8 mm (near)"
    x0, x1 = margin, width - margin
    bar = _CV_LUT[cmap][np.linspace(0, 255, x1 - x0).astype(np.uint8)]
    img[4:LEG_H - 4, x0:x1] = bar[None]
    f, s = cv2.FONT_HERSHEY_SIMPLEX, 0.42
    a = f"{lo:.4g} {unit}" + (f" {left_note}" if left_note else "")
    b = f"{hi:.4g} {unit}" + (f" {right_note}" if right_note else "")
    (wa, _), _ = cv2.getTextSize(a, f, s, 1)
    cv2.putText(img, a, (x0 - 6 - wa, LEG_H - 7), f, s, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, b, (x1 + 6, LEG_H - 7), f, s, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def img_texture(img_bgr: np.ndarray) -> np.ndarray:
    return to_texture(img_bgr, IMG_W, IMG_H)


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
    sync_role: str = ""
    status: dict = field(default_factory=dict)
    tex: Optional[np.ndarray] = None
    tex_seq: int = 0

    def alive(self) -> bool:
        return self.pending or (self.proc is not None and self.proc.is_alive()) or \
            (self.pump is not None and self.pump.is_alive())


@dataclass
class Shown:
    """The sweep result currently displayed."""
    k: int
    res: sd.SweepResult
    rig: dict
    pts: np.ndarray
    pix: tuple
    plane: dict = field(default_factory=dict)
    plane_img: Optional[np.ndarray] = None


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.ctx = mp.get_context("spawn")
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._log_lines: List[str] = []
        self._scroll_log = 0

        # Record
        self.cams = {r: CamState(r) for r in ROLES}
        self.sources: dict = {}
        self.sync_hw = args.sync == "hardware"
        self.master_role = args.master
        self.session_sync = None
        self._reaper: Optional[threading.Thread] = None
        self.recording = False
        self.session: Optional[dict] = None  # Current/last recording session
        self.rec_waiting: set = set()  # Roles whose file is being finalized
        self.rec_t0 = 0.0
        self._shown = {"cams": None, "rec": None}
        self._seq_shown = {}

        # Replay / depth
        self.rec_dir: Optional[Path] = None
        self.rec_files: dict = {}
        self.rec_info: dict = {}
        self.t_origin = 0  # Event time (us) of x = 0 on the timeline
        self.on_counts: Optional[np.ndarray] = None  # Left camera ON events per ms
        self.on_t_start = 0
        self.stereo_json = ""
        self.rig_desc = ""
        self.sweeps: List[tuple] = []  # [(t0_us, t1_us)] absolute
        self.sel: int = -1
        self.results: Dict[int, dict] = {}  # sweep index -> {"files", "stats", "key"} or {"error"}
        self.shown: Optional[Shown] = None
        self.job = None
        self.job_q = None
        self.job_cancel = None
        self.job_progress = (0, 1, "")
        self.job_keys: Dict[int, str] = {}
        self._view_dirty = True
        self._y_max = 1.0
        self._pending_select = None  # (k, estimate) for the automated test hooks
        self._screenshot_due = None
        self.viewers: List[subprocess.Popen] = []

    # ------------------------------------------------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_queue.put(line)

    # ==================================================================================================== RECORD TAB
    def refresh_cameras(self, *_):
        sources = {}
        try:
            from metavision_hal import DeviceDiscovery
            for uid in DeviceDiscovery.list():
                sources[uid.split(":")[-1]] = uid.split(":")[-1]
        except Exception as e:
            self.log(f"Camera discovery failed: {e}")
        self.sources = sources
        cams = list(sources)
        wanted = {"left": self.args.left, "right": self.args.right}
        for i, role in enumerate(ROLES):
            dpg.configure_item(f"combo_{role}", items=cams)
            if wanted[role] in sources:
                dpg.set_value(f"combo_{role}", wanted[role])
            elif dpg.get_value(f"combo_{role}") not in sources:
                dpg.set_value(f"combo_{role}", cams[i] if i < len(cams) else "")
        self.log(f"{len(cams)} camera(s) found" + (": " + ", ".join(cams) if cams else "."))
        return sources

    def swap_cameras(self, *_):
        if self.any_camera_alive():
            return
        left, right = dpg.get_value("combo_left"), dpg.get_value("combo_right")
        dpg.set_value("combo_left", right)
        dpg.set_value("combo_right", left)
        self.master_role = OTHER[self.master_role]  # Keep the same physical master
        dpg.set_value("combo_master", self.master_role)

    def any_camera_alive(self) -> bool:
        return any(c.alive() for c in self.cams.values())

    def toggle_cameras(self, *_):
        if self.any_camera_alive():
            self.close_cameras()
        else:
            self.open_cameras()

    def open_cameras(self):
        choice = {r: dpg.get_value(f"combo_{r}") for r in ROLES}
        if not all(choice[r] in self.sources for r in ROLES) or choice["left"] == choice["right"]:
            self.log("Select two different cameras (press 'Refresh' if one is missing).")
            return
        hw = self.sync_hw
        sync_modes = ({r: ("master" if r == self.master_role else "slave") for r in ROLES} if hw
                      else {r: "standalone" for r in ROLES})
        self.session_sync = (hw, self.master_role)
        erc = float(dpg.get_value("erc_mev"))
        base = {"bias_file": self.args.biases, "hw_filters": True, "trail_filter": bool(dpg.get_value("trail")),
                "erc_mev": erc}
        for role in ROLES:
            st = self.cams[role]
            st.cmd_q, st.out_q, st.stop_evt = self.ctx.Queue(), self.ctx.Queue(maxsize=64), self.ctx.Event()
            st.running, st.serial, st.status, st.sync_role = False, "", {}, sync_modes[role] if hw else ""
            st.proc = self.ctx.Process(target=recorder_camera_process.run_camera, name=f"rec_cam_{role}", daemon=True,
                                       args=(role, choice[role], dict(base, sync_mode=sync_modes[role]), st.cmd_q,
                                             st.out_q, st.stop_evt))
            st.pump = threading.Thread(target=self._pump, args=(role,), name=f"pump_{role}", daemon=True)
            st.pending = True
        # One camera at a time; with hardware sync the slave first (it waits for the master clock)
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
        was_recording = self.recording
        if was_recording:
            self.stop_recording()
        hw, master = self.session_sync or (False, self.master_role)
        states = [self.cams[OTHER[master]], self.cams[master]] if hw else list(self.cams.values())
        for st in states:
            st.pending = False

        def reaper():
            t0 = time.time()
            while self.rec_waiting and time.time() - t0 < 60:  # Let both files reach t_stop and be finalized
                time.sleep(0.1)
            if not hw:
                for st in states:
                    if st.stop_evt is not None:
                        st.stop_evt.set()
            for st in states:  # With hardware sync: slave first, while the master still provides the clock
                if st.stop_evt is not None:
                    st.stop_evt.set()
                if st.proc is not None and st.proc.pid is not None:
                    st.proc.join(timeout=120)  # Includes flushing the writer queue
                    if st.proc.is_alive():
                        self.log(f"[{st.role}] camera process did not stop, terminating it.")
                        st.proc.terminate()
        self._reaper = threading.Thread(target=reaper, name="reaper", daemon=True)
        self._reaper.start()

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
                if kind == "preview":
                    tex = to_texture(msg[3], TEX_W, TEX_H)
                    with self.lock:
                        st.tex, st.tex_seq = tex, st.tex_seq + 1
                elif kind == "status":
                    st.status = msg[2]
                elif kind == "rec_done":
                    self._on_rec_done(role, msg[2])
                elif kind == "log":
                    self.log(f"[{role}] {msg[2]}")
                elif kind == "started":
                    with self.lock:
                        st.running, st.serial, st.size, st.label = True, msg[2], (msg[3], msg[4]), msg[5]
                    self.log(f"[{role}] {msg[5]} started ({msg[3]}x{msg[4]}).")
                elif kind == "stopped":
                    break
            except Exception as e:
                self.log(f"[{role}] UI error: {e}")
                print(traceback.format_exc())
        with self.lock:
            st.running, st.status = False, {}
        self.rec_waiting.discard(role)
        self.log(f"[{role}] camera closed.")
        if self.recording:
            self.stop_recording()

    def toggle_recording(self, *_):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.rec_waiting:
            self.log("The previous recording is still being written, please wait.")
            return
        left, right = self.cams["left"], self.cams["right"]
        if not (left.running and right.running):
            self.log("Cannot record: both cameras must be open.")
            return
        hw = bool(self.session_sync and self.session_sync[0])
        t_last = [st.status.get("t_last", 0) for st in (left, right)]
        if hw and min(t_last) <= 0:
            self.log("Cannot record yet: no events from both cameras (hardware sync: is the master streaming?).")
            return
        root = Path(dpg.get_value("output_dir") or self.args.output_dir)
        session_dir = root / time.strftime("rec_%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        # Shared time base: both cameras start (and later stop) at the same event timestamp
        t_start = int(max(t_last)) + 300_000 if hw else None
        comp, comp_opts = rc.COMPRESSIONS[dpg.get_value("compression")]
        hw_master = self.session_sync[1] if hw else ""
        files = {}
        for role, st in (("left", left), ("right", right)):
            path = session_dir / f"{role}_{safe_name(st.serial)}.h5"
            files[role] = str(path)
            attrs = {"role": role, "sync_mode": st.sync_role or "standalone", "master_role": hw_master,
                     "time_base": "shared (hardware sync)" if hw else "independent camera clock",
                     "created": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session_dir.name,
                     "trail_filter": bool(dpg.get_value("trail")), "erc_mev": float(dpg.get_value("erc_mev")),
                     "biases": self.args.biases or "", "recorder": "06_stripe_depth"}
            st.cmd_q.put(("rec_start", str(path), attrs, t_start, comp, comp_opts))
        self.session = {"dir": str(session_dir), "files": files, "serials": {r: self.cams[r].serial for r in ROLES},
                        "sync": "hardware" if hw else "host", "master": hw_master, "t_start_us": t_start,
                        "compression": dpg.get_value("compression"), "results": {}}
        self._write_session()
        self.recording, self.rec_t0 = True, time.time()
        self.log(f"Recording to {session_dir}" + (f" (synchronized start at t = {t_start} us)" if hw else ""))

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        hw = bool(self.session_sync and self.session_sync[0])
        t_last = [self.cams[r].status.get("t_last", 0) for r in ROLES]
        t_stop = int(max(t_last)) + 300_000 if hw else None  # Margin for the status age; the writers cut exactly there
        self.rec_waiting = {r for r in ROLES if self.cams[r].running}
        for role in ROLES:
            st = self.cams[role]
            if st.cmd_q is not None and st.alive():
                st.cmd_q.put(("rec_stop", t_stop))
        if self.session is not None:
            self.session["t_stop_us"] = t_stop
        self.log("Stopping: writing the remaining queued events..." +
                 (f" (synchronized stop at t = {t_stop} us)" if hw else ""))

    def _on_rec_done(self, role: str, info: dict):
        if self.session is not None:
            self.session["results"][role] = info
        self.rec_waiting.discard(role)
        err = f" ERROR: {info['error']}" if info.get("error") else ""
        self.log(f"[{role}] saved {fmt_count(info['n_events'])} events, {info['bytes'] / 2 ** 20:.1f} MB "
                 f"-> {Path(info['path']).name}{err}")
        if not self.rec_waiting and self.session is not None:
            self._write_session()
            self.log(f"Recording complete: {self.session['dir']} (open it in the Replay tab).")

    def _write_session(self):
        try:
            Path(self.session["dir"], "session.json").write_text(json.dumps(self.session, indent=2))
        except Exception as e:
            self.log(f"Failed to write session.json: {e}")

    def open_output_folder(self, *_):
        path = Path(self.session["dir"]) if self.session else Path(dpg.get_value("output_dir") or self.args.output_dir)
        open_folder(path)

    def on_sync_mode(self, sender, value, user_data):
        self.sync_hw = value == SYNC_MODES[0]
        dpg.configure_item("combo_master", enabled=self.sync_hw)

    # ==================================================================================================== REPLAY TAB
    # --------------------------------------------------------------------------------------------------- recording
    def load_recording(self, directory: str):
        try:
            files = rc.find_recording_files(directory)
            info, series = {}, {}
            readers = {r: rc.EventFileReader(p) for r, p in files.items()}
            try:
                t0 = min(rd.t_start for rd in readers.values())
                t1 = max(rd.t_stop for rd in readers.values())
                bin_us = max(1000, int((t1 - t0) / 1500))  # ~1500 points on the timeline
                for role, rd in readers.items():
                    info[role] = {"serial": rd.serial, "n": rd.n, "t_start": rd.t_start, "t_stop": rd.t_stop,
                                  "size": (rd.width, rd.height), "sync": rd.attrs.get("sync_mode", "?"),
                                  "complete": bool(rd.attrs.get("complete", True)), "path": str(rd.path)}
                    ts, rate = rd.rates(bin_us)
                    series[role] = ((ts + (rd.t_start - t0) * 1e-6).tolist(), (rate / 1e6).tolist())
                t_c = time.perf_counter()
                on = sd.on_counts_per_ms(readers["left"])  # Sweep segmentation input
                t_on = time.perf_counter() - t_c
                on_t_start = readers["left"].t_start
            finally:
                for rd in readers.values():
                    rd.close()
        except Exception as e:
            self.log(f"Failed to load {directory}: {e}")
            return
        self.cancel_job()
        self.rec_dir, self.rec_files, self.rec_info, self.t_origin = Path(directory), files, info, t0
        self.on_counts, self.on_t_start = on, on_t_start
        self.sweeps, self.sel, self.results, self.shown = [], -1, {}, None
        for role in ROLES:
            dpg.set_value(f"tl_{role}", list(series[role]))
            dpg.configure_item(f"tl_{role}", label=f"{role} {info[role]['serial']} (all events)")
        # Left ON-event rate at the timeline resolution
        b = max(1, bin_us // 1000)
        n = len(on) // b * b
        on_rate = on[:n].reshape(-1, b).sum(1) / (b * 1e-3) / 1e6
        dpg.set_value("tl_on", [((np.arange(on_rate.size) * b * 1e-3) + (on_t_start - t0) * 1e-6).tolist(),
                                on_rate.tolist()])
        dpg.fit_axis_data("tl_x")
        self._y_max = max(max(series[r][1]) if series[r][1] else 0.0 for r in ROLES)
        dpg.set_axis_limits("tl_y", 0.0, max(1e-3, self._y_max * 1.15))
        dpg.set_value("tl_pos", 0.0)
        self.set_trim(start=0.0, end=(t1 - t0) * 1e-6)
        lines = [f"{Path(directory).name}   duration {(t1 - t0) * 1e-6:.2f} s"]
        for role in ROLES:
            i = info[role]
            lines.append(f"{role}: {i['serial']}  {fmt_count(i['n'])} events  ({i['sync']})"
                         + ("" if i["complete"] else "  [INCOMPLETE FILE]"))
        if len({tuple(i["size"]) for i in info.values()}) > 1:
            lines.append("WARNING: different resolutions")
        if info["left"]["sync"] == "standalone" or info["right"]["sync"] == "standalone":
            lines.append("WARNING: not hardware-synchronised: method C needs a shared clock")
        dpg.set_value("rec_info", "\n".join(lines))
        dpg.set_value("depth_out", str(self.default_out_dir()))
        found = sc.find_stereo_file(self.args.calib_dir, info["left"]["serial"], info["right"]["serial"]) or \
            sc.find_stereo_file(self.args.calib_dir, info["right"]["serial"], info["left"]["serial"])
        if found and not self.stereo_json:
            self.select_stereo_json(str(found))
        self.log(f"Loaded recording {directory} (left ON-event timeline in {t_on:.1f} s)")
        self.detect_sweeps()

    def load_last_recording(self, *_):
        if self.session and not self.rec_waiting:
            self.load_recording(self.session["dir"])
        else:
            self.log("No finished recording in this session yet.")

    def default_out_dir(self) -> Path:
        d = self.rec_dir
        name = d.name if d.parent.name == "recordings" else f"{d.parent.name}_{d.name}"
        return Path(self.args.depth_dir) / name

    def recording_duration_us(self) -> int:
        if not self.rec_info:
            return 0
        return max(i["t_stop"] for i in self.rec_info.values()) - min(i["t_start"] for i in self.rec_info.values())

    # --- time range (searched for sweeps)
    def set_trim(self, start=None, end=None):
        dur = self.recording_duration_us() * 1e-6
        start = float(dpg.get_value("trim_start") if start is None else start)
        end = float(dpg.get_value("trim_end") if end is None else end)
        start = float(np.clip(start, 0.0, dur))
        end = float(np.clip(end, 0.0, dur))
        if end <= start:
            end = min(dur, start + 1e-6)
            start = min(start, end - 1e-6) if end >= 1e-6 else 0.0
        for tag, v in (("trim_start", start), ("tl_start", start), ("trim_end", end), ("tl_end", end)):
            dpg.set_value(tag, v)

    def on_trim_input(self, *_):
        self.set_trim()

    def on_trim_drag(self, sender, *_):
        if sender == "tl_start":
            self.set_trim(start=dpg.get_value("tl_start"))
        else:
            self.set_trim(end=dpg.get_value("tl_end"))

    def trim_to_cursor(self, which: str):
        pos = float(dpg.get_value("tl_pos"))
        self.set_trim(start=pos) if which == "start" else self.set_trim(end=pos)

    # --- calibration
    def select_stereo_json(self, path: str):
        try:
            rig = sd.load_rig(path)
        except Exception as e:
            self.log(f"Calibration {path} rejected: {e}")
            return
        self.stereo_json, self.rig_desc = path, rig.describe()
        dpg.set_value("stereo_path", f"{path}\n{self.rig_desc}\nserials L/R: {rig.serial['left']} / "
                                     f"{rig.serial['right']}")
        if self.rec_info:
            serials = {self.rec_info[r]["serial"] for r in ROLES}
            if serials != set(rig.serial.values()):
                self.log(f"WARNING: the calibration is for cameras {rig.serial['left']}/{rig.serial['right']}, the "
                         f"recording has {'/'.join(sorted(serials))}: the recording roles will be used.")
        self.refresh_sweep_list()

    # ---------------------------------------------------------------------------------------------------- sweeps
    def sweep_params(self) -> sd.SweepParams:
        return sd.SweepParams(n_sigma=float(dpg.get_value("sw_sigma")), merge_gap_ms=int(dpg.get_value("sw_gap")),
                              min_len_ms=int(dpg.get_value("sw_minlen")), pad_ms=int(dpg.get_value("sw_pad")))

    def detect_sweeps(self, *_):
        if self.on_counts is None:
            self.log("Load a recording first.")
            return
        rng = (self.t_origin + int(dpg.get_value("trim_start") * 1e6), self.t_origin + int(dpg.get_value("trim_end") * 1e6))
        self.sweeps = sd.detect_sweeps(self.on_counts, self.on_t_start, self.sweep_params(), rng)
        self.results, self.shown = {}, None
        self.sel = -1
        self._view_dirty = True
        self.refresh_sweep_list()
        self.update_sweep_series()
        if self.sweeps:
            durs = [(b - a) / 1e3 for a, b in self.sweeps]
            dpg.set_value("sweep_info", f"{len(self.sweeps)} sweeps, {min(durs):.0f}-{max(durs):.0f} ms long "
                                        "(incl. padding)")
            self.log(f"{len(self.sweeps)} sweeps detected.")
            self.select_sweep(0, estimate=False)
        else:
            dpg.set_value("sweep_info", "No sweep found: lower the threshold or check the time range.")

    def sweep_stem(self, k: int) -> Path:
        a, b = self.sweeps[k]
        return Path(dpg.get_value("depth_out")) / f"sweep_{(a - self.t_origin) / 1e3:.0f}-{(b - self.t_origin) / 1e3:.0f}ms"

    def sweep_label(self, k: int) -> str:
        a, b = self.sweeps[k]
        s = f"#{k:<3d} {(a - self.t_origin) * 1e-6:8.3f} - {(b - self.t_origin) * 1e-6:8.3f} s  {(b - a) / 1e3:4.0f} ms"
        r = self.results.get(k)
        if r is None:
            r = self.find_existing(k)
        if r:
            if "error" in r:
                s += "   [error]"
            else:
                st = r["stats"]
                s += f"   {'L->R' if st['direction'] > 0 else 'R->L'}  {fmt_count(st['depth_px'])} px" + \
                     ("" if r.get("key") == self.current_key() else "  (old parameters)")
        return s

    def refresh_sweep_list(self):
        items = [self.sweep_label(k) for k in range(len(self.sweeps))]
        dpg.configure_item("sweep_list", items=items)
        if 0 <= self.sel < len(items):
            dpg.set_value("sweep_list", items[self.sel])

    def update_sweep_series(self):
        h = max(1e-3, self._y_max * 1.08)
        xs, ys = [], []
        for a, b in self.sweeps:
            a, b = (a - self.t_origin) * 1e-6, (b - self.t_origin) * 1e-6
            xs += [a, a, b, b]
            ys += [0.0, h, h, 0.0]
        dpg.set_value("tl_sweeps", [xs, ys])
        if 0 <= self.sel < len(self.sweeps):
            a, b = [(v - self.t_origin) * 1e-6 for v in self.sweeps[self.sel]]
            dpg.set_value("tl_sel", [[a, a, b, b], [0.0, h, h, 0.0]])
        else:
            dpg.set_value("tl_sel", [[], []])

    def on_sweep_list(self, sender, value):
        items = dpg.get_item_configuration("sweep_list")["items"]
        if value in items:
            self.select_sweep(items.index(value))

    def select_sweep(self, k: int, estimate: Optional[bool] = None):
        if not self.sweeps:
            return
        k = int(np.clip(k, 0, len(self.sweeps) - 1))
        self.sel = k
        a, b = self.sweeps[k]
        dpg.set_value("tl_pos", (a - self.t_origin) * 1e-6)
        self.update_sweep_series()
        self.refresh_sweep_list()
        r = self.results.get(k) or self.find_existing(k)
        if r and "files" in r and r.get("key") == self.current_key():
            self.results[k] = r
            self.show_result(k)
            return
        self.shown = None
        self._view_dirty = True
        if estimate is None:
            estimate = bool(dpg.get_value("auto_estimate"))
        if estimate:
            self.estimate([k])

    def on_timeline_click(self, *_):
        x, _ = dpg.get_plot_mouse_pos()
        dpg.set_value("tl_pos", float(x))
        t = self.t_origin + x * 1e6
        for k, (a, b) in enumerate(self.sweeps):
            if a <= t < b:
                self.select_sweep(k)
                return

    # ----------------------------------------------------------------------------------------------------- depth
    def depth_params(self) -> sd.DepthParams:
        g = dpg.get_value
        return sd.DepthParams(tau_px=float(g("p_tau")), row_halfwidth=int(g("p_rows")), min_events=int(g("p_min")),
                              nb_delta_us=float(g("p_nbdt")), nb_min=int(g("p_nbk")), agg_k=int(g("p_aggk")),
                              agg_max_dev_px=float(g("p_dev")), agg_min_support=int(g("p_sup")),
                              d_min=float(g("p_dmin")), d_max=float(g("p_dmax")))

    def current_key(self) -> str:
        return sd.params_key(self.depth_params(), self.stereo_json)

    def find_existing(self, k: int) -> Optional[dict]:
        """A result of this sweep saved earlier (any parameters): {"files", "stats", "key"}."""
        stem = self.sweep_stem(k)
        js, npz = Path(f"{stem}_info.json"), Path(f"{stem}_depth.npz")
        if not (js.exists() and npz.exists()):
            return None
        try:
            st = json.loads(js.read_text())
            key = sd.params_key(sd.DepthParams(**st["params"]), st.get("calib", ""))
        except Exception:
            return None
        return {"files": {"npz": str(npz), "ply": f"{stem}_cloud.ply", "json": str(js)}, "stats": st, "key": key}

    def estimate(self, ks: List[int]):
        if self.job is not None and self.job.is_alive():
            self.log("Busy: wait for the current estimation or press 'Cancel'.")
            return
        if not self.rec_files:
            self.log("Load a recording first.")
            return
        if not self.stereo_json:
            self.log("Select the stereo calibration (step 2) first.")
            return
        if not ks:
            self.log("Nothing to estimate.")
            return
        p = self.depth_params()
        key = self.current_key()
        job = [(k, int(self.sweeps[k][0]), int(self.sweeps[k][1]), str(self.sweep_stem(k))) for k in ks]
        self.job_keys = {k: key for k in ks}
        self.job_q, self.job_cancel = self.ctx.Queue(), self.ctx.Event()
        self.job_progress = (0, len(job), "starting")
        self.job = self.ctx.Process(target=sd.run_depth_job, daemon=True, name="depth_job",
                                    args=({r: str(f) for r, f in self.rec_files.items()}, self.stereo_json, job,
                                          asdict(p), self.job_q, self.job_cancel))
        self.job.start()
        self.log(f"Estimating depth for {len(job)} sweep(s) (tau {p.tau_px:g} px, rows +-{p.row_halfwidth}, "
                 f"window {p.agg_k}x{p.agg_k}) -> {dpg.get_value('depth_out')}")

    def estimate_selected(self, *_):
        if self.job is not None and self.job.is_alive():
            self.cancel_job()
            return
        if 0 <= self.sel < len(self.sweeps):
            self.estimate([self.sel])
        else:
            self.log("Select a sweep first.")

    def estimate_all(self, *_):
        if self.job is not None and self.job.is_alive():
            self.cancel_job()
            return
        key = self.current_key()
        todo = []
        for k in range(len(self.sweeps)):
            r = self.results.get(k) or self.find_existing(k)
            if not (r and "files" in r and r.get("key") == key):
                todo.append(k)
        if not todo:
            self.log("All sweeps are already estimated with these parameters.")
        self.estimate(todo)

    def cancel_job(self):
        if self.job is not None and self.job.is_alive():
            self.job_cancel.set()
            self.log("Cancelling after the current sweep...")

    def _poll_job(self):
        while self.job_q is not None:
            try:
                msg = self.job_q.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == "progress":
                self.job_progress = (msg[1], msg[2], msg[3])
            elif kind == "log":
                self.log(msg[1])
            elif kind == "sweep_done":
                k, files, stats = msg[1], msg[2], msg[3]
                self.results[k] = {"files": files, "stats": stats, "key": self.job_keys.get(k)}
                t = stats["timing_s"]
                self.log(f"Sweep #{k}: {fmt_count(stats['matched'])} matches, {fmt_count(stats['depth_px'])} depth px, "
                         f"Z median {stats['z_median_m'] * 1e3:.0f} mm, {stats['total_s']:.1f} s "
                         f"(filter {t['read_filter_left_s'] + t['read_filter_right_s']:.1f}, "
                         f"match {t['match_s']:.2f}, aggregate {t['aggregate_s']:.2f})")
                self.refresh_sweep_list()
                if k == self.sel:
                    self.show_result(k)
            elif kind == "sweep_error":
                self.results[msg[1]] = {"error": msg[2]}
                self.log(f"Sweep #{msg[1]} failed: {msg[2]}")
                self.refresh_sweep_list()
            elif kind in ("done", "cancelled", "error"):
                if kind == "error":
                    self.log(f"Depth estimation failed: {msg[1]}")
                elif kind == "cancelled":
                    self.log("Depth estimation cancelled.")
                self.job_q = None

    # --------------------------------------------------------------------------------------------------- display
    def show_result(self, k: int):
        r = self.results.get(k)
        try:
            res, rig = sd.load_result(r["files"]["npz"])
        except Exception as e:
            self.log(f"Cannot read the result of sweep #{k}: {e}")
            return
        pts, pix = sd.cloud_from_result(res, rig)
        self.shown = Shown(k, res, rig, pts, pix)
        self.update_plane()
        self._view_dirty = True

    def update_plane(self, *_):
        s = self.shown
        if s is None:
            return
        if dpg.get_value("flat_check") and s.plane_img is None and len(s.pts) > 100:
            s.plane = sd.plane_summary(s.pts)
            img = np.full(s.res.disp.shape, np.nan, np.float32)
            img[s.pix] = s.plane["residual_mm"]
            s.plane_img = img
        self._view_dirty = True

    def view_image(self, view: str):
        """(BGR image, legend BGR, title) of one view of the shown result."""
        s = self.shown
        res = s.res
        fB = s.rig["fB"]
        if view.startswith("Arrival"):
            side = "left" if "left" in view else "right"
            both = np.concatenate([res.arrival_ms[x][np.isfinite(res.arrival_ms[x])] for x in ROLES])
            lo, hi = (np.percentile(both, [0.5, 99.5]) if both.size else (0, 1))
            return (colorize(res.arrival_ms[side], lo, hi, "turbo"), legend(lo, hi, "turbo", "ms"),
                    f"Stripe arrival time, {side} camera (ms, same scale in both)")
        if view.startswith("Depth"):
            lo, hi = self.depth_range()
            with np.errstate(divide="ignore", invalid="ignore"):
                z = fB / (res.raw_disp if "before" in view else res.disp) * 1e3
            # Near = yellow: colour map reversed by swapping the range
            img = colorize(z, hi, lo, "viridis")
            return img, legend(hi, lo, "viridis", "mm", "(far)", "(near)"), view + " (mm)"
        if view.startswith("Distance"):
            if s.plane_img is None:
                return (placeholder("Tick 'Flat-target check' to fit a plane", IMG_W, IMG_H), legend(0, 1, "diverging", ""),
                        view)
            r = float(dpg.get_value("plane_range"))
            return (colorize(s.plane_img, -r, r, "diverging"), legend(-r, r, "diverging", "mm", "(in front)", "(behind)"),
                    f"{view} (mm): blue = in front of the plane, red = behind")
        k = int(s.res.stats["params"]["agg_k"]) ** 2
        sup = np.where(res.support > 0, res.support.astype(np.float32), np.nan)
        return colorize(sup, 0, k, "inferno"), legend(0, k, "inferno", "values"), view

    def depth_range(self):
        s = self.shown
        if not dpg.get_value("z_auto"):
            return float(dpg.get_value("z_min")), float(dpg.get_value("z_max"))
        lo, hi = s.res.stats.get("z_p1_p99_m", [np.nan, np.nan])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return 0.0, 1000.0
        dpg.set_value("z_min", lo * 1e3)
        dpg.set_value("z_max", hi * 1e3)
        return lo * 1e3, hi * 1e3

    def redraw_views(self):
        self._view_dirty = False
        for pane in ("a", "b"):
            view = dpg.get_value(f"view_{pane}")
            if self.shown is None:
                msg = ("Load a recording" if not self.rec_files else "Select the calibration" if not self.stereo_json
                       else "Select a sweep" if self.sel < 0 else "Press 'Estimate selected sweep'")
                if self.job is not None and self.job.is_alive():
                    msg = "Estimating depth..."
                img, leg, title = placeholder(msg, IMG_W, IMG_H), np.full((LEG_H, LEG_W, 3), BG, np.uint8), view
            else:
                img, leg, title = self.view_image(view)
            dpg.set_value(f"tex_{pane}", img_texture(img))
            dpg.set_value(f"tex_leg_{pane}", to_texture(leg, LEG_W, LEG_H))
            dpg.configure_item(f"plot_{pane}", label=title)
        self.update_stats()

    def update_stats(self):
        s = self.shown
        if s is None:
            dpg.set_value("stats", "")
            return
        st = s.res.stats
        t = st["timing_s"]
        lines = [f"Sweep #{s.k}: {(st['t0_us'] - self.t_origin) * 1e-6:.3f} - {(st['t1_us'] - self.t_origin) * 1e-6:.3f} s, "
                 f"stripe {'left -> right' if st['direction'] > 0 else 'right -> left'} at {st['speed_px_ms']:.1f} px/ms "
                 f"(crosses the image in {st['cross_ms']:.0f} ms), tau = {st['tau_us']:.0f} us",
                 f"ON events L/R: {fmt_count(st['on_events']['left'])} / {fmt_count(st['on_events']['right'])}   "
                 f"first filtered events: {fmt_count(st['first_events']['left'])} / "
                 f"{fmt_count(st['first_events']['right'])}   matched: {fmt_count(st['matched'])}   "
                 f"depth pixels: {fmt_count(st['depth_px'])}",
                 f"Z median {st['z_median_m'] * 1e3:.1f} mm (1-99 %: {st['z_p1_p99_m'][0] * 1e3:.0f} - "
                 f"{st['z_p1_p99_m'][1] * 1e3:.0f} mm)   processing {st['total_s']:.1f} s (filter "
                 f"{t['read_filter_left_s'] + t['read_filter_right_s']:.1f} s, match {t['match_s']:.2f} s, "
                 f"aggregation {t['aggregate_s']:.2f} s)"]
        if s.plane:
            p = s.plane
            lines.append(f"Plane fit: {p['distance_m'] * 1e3:.0f} mm away, tilted {p['tilt_deg']:.1f} deg, spread "
                         f"(1.48 MAD) {p['mad_mm']:.2f} mm, {p['within_10mm'] * 100:.1f} % within 10 mm")
        dpg.set_value("stats", "\n".join(lines))

    def update_hover(self):
        s = self.shown
        for pane in ("a", "b"):
            if dpg.is_item_hovered(f"plot_{pane}"):
                break
        else:
            return
        if s is None:
            return
        x, y = dpg.get_plot_mouse_pos()
        h, w = s.res.disp.shape
        c, r = int(np.floor(x)), int(np.floor(y))
        if not (0 <= c < w and 0 <= r < h):
            dpg.set_value("hover", "")
            return
        fB = s.rig["fB"]
        d, dr = s.res.disp[r, c], s.res.raw_disp[r, c]
        parts = [f"pixel x {c}, row {r}:"]
        parts.append(f"Z {fB / d * 1e3:.1f} mm (d {d:.2f} px, support {s.res.support[r, c]})" if np.isfinite(d)
                     else "no depth")
        if np.isfinite(dr):
            parts.append(f"event: Z {fB / dr * 1e3:.1f} mm, d {dr:.2f} px")
        for side in ROLES:
            v = s.res.arrival_ms[side][r, c]
            if np.isfinite(v):
                parts.append(f"t_{side[0].upper()} {v:.3f} ms")
        if s.plane_img is not None and np.isfinite(s.plane_img[r, c]):
            parts.append(f"plane {s.plane_img[r, c]:+.1f} mm")
        dpg.set_value("hover", "   ".join(parts))

    # ----------------------------------------------------------------------------------------------- point cloud
    def open_cloud(self, all_sweeps: bool = False):
        key = self.current_key()
        if all_sweeps:
            ks = [k for k in range(len(self.sweeps)) if (self.results.get(k) or {}).get("files")]
        else:
            ks = [self.sel] if (self.results.get(self.sel) or {}).get("files") else []
        files = [self.results[k]["files"]["ply"] for k in ks if Path(self.results[k]["files"]["ply"]).exists()]
        if not files:
            self.log("No point cloud yet: estimate the depth first.")
            return
        mode = {CLOUD_COLORS[0]: "file", CLOUD_COLORS[1]: "depth", CLOUD_COLORS[2]: "height",
                CLOUD_COLORS[3]: "plane"}[dpg.get_value("cloud_color")]
        py = self.args.viewer_python or sys.executable
        cmd = [py, "-u", str(SCRIPT_DIR / "cloud_viewer.py"), *files, "--color", mode]
        if dpg.get_value("cloud_overlay") and len(files) > 1:
            cmd.append("--overlay")
        if self.args.test_viewer_shot:
            cmd += ["--screenshot", self.args.test_viewer_shot]
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags)
        except Exception as e:
            self.log(f"Cannot start the point cloud viewer ({py}): {e}")
            return
        self.viewers.append(p)
        self.log(f"Opening {len(files)} point cloud(s) in Open3D" + ("" if key == self.results[ks[0]].get("key")
                                                                       else " (estimated with other parameters)"))

        def reader():
            for line in p.stdout:
                line = line.rstrip()
                if line:
                    self.log(f"[viewer] {line}")
            if p.wait() != 0:
                self.log(f"[viewer] exited with code {p.returncode}. Open3D missing? Install it in {py}: "
                         f"pip install open3d (or uv pip install open3d --python \"{py}\"), "
                         "or start the app with --viewer-python <python with open3d>.")
        threading.Thread(target=reader, daemon=True).start()

    def save_depth_png(self, *_):
        """<stem>_depth_color.png (as displayed, with a colour bar) and <stem>_depth_mm.png (16-bit, mm, 0 = none)."""
        s = self.shown
        if s is None:
            self.log("No depth image yet: estimate the depth of a sweep first.")
            return
        npz = self.results.get(s.k, {}).get("files", {}).get("npz", "")
        stem = npz[:-len("_depth.npz")] if npz.endswith("_depth.npz") else str(self.sweep_stem(s.k))
        lo, hi = self.depth_range()
        z_mm = s.res.depth * 1e3
        color = colorize(z_mm, hi, lo, "viridis")  # Near = yellow, as in the view
        h, w = color.shape[:2]
        band = np.full((LEG_H + 10, w, 3), BG, np.uint8)
        band[5:5 + LEG_H] = legend(hi, lo, "viridis", "mm", "(far)", "(near)", width=w)
        files = {"color": f"{stem}_depth_color.png", "mm": f"{stem}_depth_mm.png"}
        try:
            sd.write_png(files["color"], np.vstack([color, band]))
            sd.write_depth_png16(files["mm"], s.res.depth)
        except Exception as e:
            self.log(f"Saving the depth PNG failed: {e}")
            return
        self.log(f"Saved {Path(files['color']).name} (colours {hi:.0f} - {lo:.0f} mm, far - near) and "
                 f"{Path(files['mm']).name} (16-bit, depth in mm, 0 = none) in {Path(stem).parent}")

    def open_depth_folder(self, *_):
        open_folder(Path(dpg.get_value("depth_out")))

    # ============================================================================================================ UI
    def build_ui(self):
        with dpg.texture_registry():
            for role in ROLES:
                dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder(f"{role} camera closed"), TEX_W, TEX_H),
                                    format=dpg.mvFormat_Float_rgb, tag=f"tex_live_{role}")
            for pane in ("a", "b"):
                dpg.add_raw_texture(IMG_W, IMG_H, img_texture(placeholder("Load a recording", IMG_W, IMG_H)),
                                    format=dpg.mvFormat_Float_rgb, tag=f"tex_{pane}")
                dpg.add_raw_texture(LEG_W, LEG_H, np.zeros(LEG_W * LEG_H * 3, np.float32),
                                    format=dpg.mvFormat_Float_rgb, tag=f"tex_leg_{pane}")
        for tag, colors in (("theme_rec_idle", ((30, 120, 50), (40, 150, 65), (50, 170, 80))),
                            ("theme_rec_active", ((170, 35, 35), (200, 45, 45), (220, 60, 60))),
                            ("theme_action", ((40, 80, 150), (55, 100, 180), (70, 120, 200)))):
            with dpg.theme(tag=tag):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, colors[0])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colors[1])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colors[2])
        for name, rgb in SERIES_COLORS.items():
            with dpg.theme(tag=f"theme_series_{name}"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, rgb, category=dpg.mvThemeCat_Plots)
        for tag, rgb, wt in (("theme_sweeps", (120, 230, 120), 1.0), ("theme_sel", (255, 80, 220), 2.5)):
            with dpg.theme(tag=tag):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, rgb, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, wt, category=dpg.mvThemeCat_Plots)

        dlg = dict(show=False, width=750, height=450)
        dpg.add_file_dialog(directory_selector=True, tag="dlg_output", **dlg,
                            callback=lambda s, a: dpg.set_value("output_dir", a["file_path_name"]))
        dpg.add_file_dialog(directory_selector=True, tag="dlg_load", **dlg, default_path=str(self.args.recordings_dir),
                            callback=lambda s, a: self.load_recording(a["file_path_name"]))
        dpg.add_file_dialog(directory_selector=True, tag="dlg_depth_out", **dlg, default_path=str(self.args.depth_dir),
                            callback=lambda s, a: dpg.set_value("depth_out", a["file_path_name"]))
        with dpg.file_dialog(tag="dlg_stereo", **dlg, default_path=str(self.args.calib_dir),
                             callback=lambda s, a: self.select_stereo_json(a["file_path_name"])):
            dpg.add_file_extension(".json")

        a = self.args
        wrap = LEFT_W - 30
        dp = sd.DepthParams()
        spar = sd.SweepParams()
        with dpg.window(tag="main"):
            with dpg.tab_bar(tag="tabs"):
                # ---------------------------------------------------------------------------------------- Record
                with dpg.tab(label="Record"):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=LEFT_W, tag="rec_left", border=True):
                            with dpg.collapsing_header(label="Cameras", default_open=True):
                                dpg.add_combo([], label="Left", tag="combo_left", width=LEFT_W - 170)
                                dpg.add_combo([], label="Right", tag="combo_right", width=LEFT_W - 170)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Refresh", tag="btn_refresh",
                                                   callback=lambda: self.refresh_cameras())
                                    dpg.add_button(label="Swap left/right", tag="btn_swap", callback=self.swap_cameras)
                                dpg.add_combo(SYNC_MODES, label="Timestamp sync", tag="combo_sync",
                                              default_value=SYNC_MODES[0] if self.sync_hw else SYNC_MODES[1],
                                              width=LEFT_W - 170, callback=self.on_sync_mode)
                                dpg.add_combo(ROLES, label="Master camera", tag="combo_master",
                                              default_value=self.master_role, enabled=self.sync_hw,
                                              width=LEFT_W - 170, callback=lambda s, v: setattr(self, "master_role", v))
                                dpg.add_checkbox(label="Sensor trail filter (off = raw events)", tag="trail",
                                                 default_value=a.trail_filter)
                                dpg.add_input_float(label="Event rate limit (Mev/s, 0 = off)", tag="erc_mev",
                                                    default_value=a.erc_mev, width=110, format="%.0f", step=5)
                                b = dpg.add_button(label="Open cameras", tag="btn_cameras", width=-1, height=30,
                                                   callback=self.toggle_cameras)
                                dpg.bind_item_theme(b, "theme_action")
                                for role in ROLES:
                                    dpg.add_text("", tag=f"cam_status_{role}", wrap=wrap)
                                    dpg.add_text("", tag=f"cam_rec_{role}", wrap=wrap, color=(170, 170, 170))
                            with dpg.collapsing_header(label="Recording", default_open=True):
                                dpg.add_text("Output folder:")
                                with dpg.group(horizontal=True):
                                    dpg.add_input_text(tag="output_dir", default_value=str(a.output_dir),
                                                       width=LEFT_W - 110)
                                    dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dlg_output"))
                                dpg.add_combo(list(rc.COMPRESSIONS), label="Compression", tag="compression",
                                              default_value=list(rc.COMPRESSIONS)[0], width=200)
                                dpg.add_button(label="Start recording", tag="btn_record", width=-1, height=42,
                                               callback=self.toggle_recording)
                                dpg.bind_item_theme("btn_record", "theme_rec_idle")
                                dpg.add_text("", tag="rec_status", wrap=wrap)
                                dpg.add_button(label="Open folder", callback=self.open_output_folder)
                        with dpg.child_window(tag="rec_right", border=False):
                            with dpg.group(horizontal=True):
                                for role in ROLES:
                                    with dpg.group():
                                        dpg.add_text(f"{role.capitalize()} camera (live, not rectified)",
                                                     tag=f"lbl_live_{role}")
                                        dpg.add_image(f"tex_live_{role}", tag=f"img_live_{role}", width=TEX_W // 2,
                                                      height=TEX_H // 2)
                # ------------------------------------------------------------------------------- Replay / depth
                with dpg.tab(label="Replay: depth (method C)"):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=LEFT_W, tag="rp_left", border=True):
                            with dpg.collapsing_header(label="Recording", default_open=True):
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Load recording...",
                                                   callback=lambda: dpg.show_item("dlg_load"))
                                    dpg.add_button(label="Load last recording", callback=self.load_last_recording)
                                dpg.add_text("No recording loaded.", tag="rec_info", wrap=wrap)
                                dpg.add_button(label="Select calibration...", callback=lambda: dpg.show_item("dlg_stereo"))
                                dpg.add_text("(no stereo calibration selected)", tag="stereo_path", wrap=wrap,
                                             color=(170, 170, 170))
                            with dpg.collapsing_header(label="1. Sweep segmentation", default_open=True):
                                dpg.add_text("Time range searched (s from the recording start):")
                                with dpg.group(horizontal=True):
                                    dpg.add_input_float(label="Start", tag="trim_start", default_value=0.0, width=120,
                                                        format="%.3f", step=0.1, callback=self.on_trim_input,
                                                        on_enter=True)
                                    dpg.add_input_float(label="End", tag="trim_end", default_value=0.0, width=120,
                                                        format="%.3f", step=0.1, callback=self.on_trim_input,
                                                        on_enter=True)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Start = cursor", callback=lambda: self.trim_to_cursor("start"))
                                    dpg.add_button(label="End = cursor", callback=lambda: self.trim_to_cursor("end"))
                                    dpg.add_button(label="Full range",
                                                   callback=lambda: self.set_trim(0.0, self.recording_duration_us() * 1e-6))
                                dpg.add_input_float(label="Threshold (robust sigmas)", tag="sw_sigma",
                                                    default_value=spar.n_sigma, width=110, step=1, format="%.1f")
                                dpg.add_input_int(label="Merge gaps shorter than (ms)", tag="sw_gap",
                                                  default_value=spar.merge_gap_ms, width=110, min_value=0,
                                                  min_clamped=True)
                                dpg.add_input_int(label="Minimum sweep length (ms)", tag="sw_minlen",
                                                  default_value=spar.min_len_ms, width=110, min_value=1,
                                                  min_clamped=True)
                                dpg.add_input_int(label="Padding on both sides (ms)", tag="sw_pad",
                                                  default_value=spar.pad_ms, width=110, min_value=0, min_clamped=True)
                                b = dpg.add_button(label="Detect sweeps", width=-1, height=28, callback=self.detect_sweeps)
                                dpg.bind_item_theme(b, "theme_action")
                                dpg.add_text("", tag="sweep_info", wrap=wrap, color=(150, 220, 150))
                                dpg.add_listbox([], tag="sweep_list", width=-1, num_items=7,
                                                callback=self.on_sweep_list)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="< Previous", callback=lambda: self.select_sweep(self.sel - 1))
                                    dpg.add_button(label="Next >", callback=lambda: self.select_sweep(self.sel + 1))
                                    dpg.add_checkbox(label="Estimate on selection", tag="auto_estimate",
                                                     default_value=True)
                            with dpg.collapsing_header(label="2. Depth (method C)", default_open=True):
                                w = 110
                                dpg.add_input_float(label="Match window tau (px of stripe motion)", tag="p_tau",
                                                    default_value=dp.tau_px, width=w, step=1, format="%.1f",
                                                    min_value=0.5, min_clamped=True)
                                dpg.add_input_int(label="Rows used: row +-", tag="p_rows", default_value=dp.row_halfwidth,
                                                  width=w, min_value=0, max_value=5, min_clamped=True, max_clamped=True)
                                dpg.add_input_int(label="Min. right events per fit", tag="p_min",
                                                  default_value=dp.min_events, width=w, min_value=2, min_clamped=True)
                                dpg.add_input_float(label="Neighbour filter window (us)", tag="p_nbdt",
                                                    default_value=dp.nb_delta_us, width=w, step=50, format="%.0f",
                                                    min_value=1, min_clamped=True)
                                dpg.add_input_int(label="Neighbour filter: min. neighbours (0 = off)", tag="p_nbk",
                                                  default_value=dp.nb_min, width=w, min_value=0, max_value=8,
                                                  min_clamped=True, max_clamped=True)
                                dpg.add_input_int(label="Aggregation window (px, odd)", tag="p_aggk",
                                                  default_value=dp.agg_k, width=w, step=2, min_value=1, max_value=15,
                                                  min_clamped=True, max_clamped=True)
                                dpg.add_input_float(label="Aggregation: max. deviation (px)", tag="p_dev",
                                                    default_value=dp.agg_max_dev_px, width=w, step=0.25, format="%.2f")
                                dpg.add_input_int(label="Aggregation: min. agreeing values", tag="p_sup",
                                                  default_value=dp.agg_min_support, width=w, min_value=1,
                                                  min_clamped=True)
                                with dpg.group(horizontal=True):
                                    dpg.add_input_float(label="", tag="p_dmin", default_value=dp.d_min, width=80,
                                                        step=0, format="%.0f")
                                    dpg.add_input_float(label="Disparity range (px)", tag="p_dmax",
                                                        default_value=dp.d_max, width=80, step=0, format="%.0f")
                                b = dpg.add_button(label="Estimate selected sweep", tag="btn_est", width=-1, height=32,
                                                   callback=self.estimate_selected)
                                dpg.bind_item_theme(b, "theme_action")
                                dpg.add_button(label="Estimate all sweeps", tag="btn_est_all", width=-1,
                                               callback=self.estimate_all)
                                dpg.add_progress_bar(tag="job_progress", default_value=0.0, width=-1, overlay="")
                                dpg.add_checkbox(label="Flat-target check (fit a plane, show distances)",
                                                 tag="flat_check", default_value=False, callback=self.update_plane)
                            with dpg.collapsing_header(label="3. Export and point cloud", default_open=True):
                                dpg.add_text("Output folder (depth .npz/.png, cloud .ply, stats .json):")
                                with dpg.group(horizontal=True):
                                    dpg.add_input_text(tag="depth_out", default_value=str(a.depth_dir),
                                                       width=LEFT_W - 110)
                                    dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dlg_depth_out"))
                                dpg.add_button(label="Save depth image (PNG)", width=-1, height=28,
                                               callback=self.save_depth_png)
                                dpg.add_text("Saves <sweep>_depth_color.png (as displayed, with colour bar) and "
                                             "<sweep>_depth_mm.png (16-bit, depth in mm, 0 = no depth).",
                                             wrap=wrap, color=(170, 170, 170))
                                dpg.add_combo(CLOUD_COLORS, label="Colours", tag="cloud_color",
                                              default_value=CLOUD_COLORS[0], width=200)
                                b = dpg.add_button(label="Open in Open3D (selected sweep)", width=-1, height=32,
                                                   callback=lambda: self.open_cloud(False))
                                dpg.bind_item_theme(b, "theme_action")
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Open all estimated sweeps", callback=lambda: self.open_cloud(True))
                                    dpg.add_checkbox(label="overlaid", tag="cloud_overlay", default_value=False)
                                dpg.add_button(label="Open output folder", callback=self.open_depth_folder)
                        with dpg.child_window(tag="rp_right", border=False):
                            with dpg.group(horizontal=True):
                                for pane, default in (("a", VIEWS[0]), ("b", VIEWS[3])):
                                    with dpg.group(tag=f"pane_{pane}"):
                                        dpg.add_combo(VIEWS, tag=f"view_{pane}", default_value=default, width=330,
                                                      callback=lambda: setattr(self, "_view_dirty", True))
                                        with dpg.plot(tag=f"plot_{pane}", label=default, width=600, height=360,
                                                      equal_aspects=True, no_mouse_pos=True):
                                            dpg.add_plot_axis(dpg.mvXAxis, label="rectified x (px)", tag=f"ax_x_{pane}")
                                            with dpg.plot_axis(dpg.mvYAxis, label="rectified row", tag=f"ax_y_{pane}",
                                                               invert=True):
                                                # Inverted y axis (row 0 at the top): flip the texture to match
                                                dpg.add_image_series(f"tex_{pane}", [0, 0], [IMG_W, IMG_H],
                                                                     uv_min=(0, 1), uv_max=(1, 0), tag=f"img_{pane}")
                                        dpg.add_image(f"tex_leg_{pane}", width=LEG_W // 2 + 150, height=LEG_H)
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Depth colour range: auto (1-99 %)", tag="z_auto",
                                                 default_value=True, callback=lambda: setattr(self, "_view_dirty", True))
                                dpg.add_input_float(label="min (mm)", tag="z_min", default_value=0.0, width=90, step=0,
                                                    format="%.0f", on_enter=True,
                                                    callback=lambda: (dpg.set_value("z_auto", False),
                                                                      setattr(self, "_view_dirty", True)))
                                dpg.add_input_float(label="max (mm)", tag="z_max", default_value=1000.0, width=90,
                                                    step=0, format="%.0f", on_enter=True,
                                                    callback=lambda: (dpg.set_value("z_auto", False),
                                                                      setattr(self, "_view_dirty", True)))
                                dpg.add_input_float(label="plane distance range +- (mm)", tag="plane_range",
                                                    default_value=10.0, width=90, step=0, format="%.0f", on_enter=True,
                                                    callback=lambda: setattr(self, "_view_dirty", True))
                            dpg.add_text("", tag="hover", color=(255, 230, 120))
                            dpg.add_text("", tag="stats")
                            with dpg.plot(label="Event rate: click a sweep (green) to select it; green/red lines = "
                                                "time range searched", tag="timeline_plot", height=TIMELINE_H, width=-1):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="time (s)", tag="tl_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="event rate (Mev/s)", tag="tl_y"):
                                    for role in ROLES:
                                        dpg.add_line_series([], [], label=role, tag=f"tl_{role}")
                                        dpg.bind_item_theme(f"tl_{role}", f"theme_series_{role}")
                                    dpg.add_line_series([], [], label="left ON events", tag="tl_on")
                                    dpg.bind_item_theme("tl_on", "theme_series_on")
                                    dpg.add_line_series([], [], label="sweeps", tag="tl_sweeps")
                                    dpg.bind_item_theme("tl_sweeps", "theme_sweeps")
                                    dpg.add_line_series([], [], label="selected", tag="tl_sel")
                                    dpg.bind_item_theme("tl_sel", "theme_sel")
                                dpg.add_drag_line(label="start", tag="tl_start", color=[60, 220, 90, 255],
                                                  default_value=0.0, callback=self.on_trim_drag)
                                dpg.add_drag_line(label="end", tag="tl_end", color=[235, 70, 70, 255],
                                                  default_value=0.0, callback=self.on_trim_drag)
                                dpg.add_drag_line(label="cursor", tag="tl_pos", color=[255, 230, 0, 255],
                                                  default_value=0.0)
            dpg.add_text("Log")
            with dpg.child_window(tag="log_child", height=LOG_H, border=True):
                dpg.add_text("", tag="log_text", wrap=0)
        with dpg.item_handler_registry(tag="tl_handlers"):
            dpg.add_item_clicked_handler(callback=self.on_timeline_click)
        dpg.bind_item_handler_registry("timeline_plot", "tl_handlers")

    def layout(self, *_):
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        panel_h = max(200, vh - LOG_H - 80)
        for tag in ("rec_left", "rec_right", "rp_left", "rp_right"):
            dpg.configure_item(tag, height=panel_h)
        right_w = max(300, vw - LEFT_W - 40)
        aspect = TEX_W / TEX_H
        iw = int(min((right_w - 16) / 2, (panel_h - 40) * aspect))
        for role in ROLES:
            dpg.configure_item(f"img_live_{role}", width=iw, height=int(iw / aspect))
        # Depth views: two plots side by side above the stats and the timeline
        pw = int((right_w - 24) / 2)
        ph = int(min(pw * IMG_H / IMG_W + 60, panel_h - TIMELINE_H - 190))
        for pane in ("a", "b"):
            dpg.configure_item(f"plot_{pane}", width=pw, height=max(150, ph))

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

        # --- Record tab
        with self.lock:
            updates = []
            for role in ROLES:
                st = self.cams[role]
                if st.tex is not None and self._seq_shown.get(role) != st.tex_seq:
                    self._seq_shown[role] = st.tex_seq
                    updates.append((f"tex_live_{role}", st.tex))
        for tag, tex in updates:
            dpg.set_value(tag, tex)
        alive = {r: self.cams[r].alive() for r in ROLES}
        for role in ROLES:
            st, s = self.cams[role], self.cams[role].status
            if st.running:
                dpg.set_value(f"cam_status_{role}", f"{role.capitalize()}: {st.label}"
                              + (f" [{st.sync_role}]" if st.sync_role else "")
                              + f"  {s.get('rate', 0) / 1e6:.2f} Mev/s  latency {s.get('lag', 0) * 1000:.0f} ms")
                rec = (f"  writing: {fmt_count(s.get('written', 0))} events, file {s.get('file_mb', 0):.1f} MB, "
                       f"queue {s.get('backlog_mb', 0):.1f} MB" if s.get("recording") else "")
                dpg.set_value(f"cam_rec_{role}", rec)
            else:
                dpg.set_value(f"cam_status_{role}", f"{role.capitalize()}: " + ("opening..." if alive[role] else "closed"))
                dpg.set_value(f"cam_rec_{role}", "")
        state = tuple(alive.values())
        if state != self._shown["cams"]:
            self._shown["cams"] = state
            any_alive = any(state)
            dpg.configure_item("btn_cameras", label="Close cameras" if any_alive else "Open cameras")
            for tag in ("combo_left", "combo_right", "btn_refresh", "btn_swap", "combo_sync", "trail", "erc_mev"):
                dpg.configure_item(tag, enabled=not any_alive)
            dpg.configure_item("combo_master", enabled=not any_alive and self.sync_hw)
            for role in ROLES:
                if not alive[role]:
                    dpg.set_value(f"tex_live_{role}", to_texture(placeholder(f"{role} camera closed"), TEX_W, TEX_H))
        if self.recording != self._shown["rec"]:
            self._shown["rec"] = self.recording
            dpg.configure_item("btn_record", label="Stop recording" if self.recording else "Start recording")
            dpg.bind_item_theme("btn_record", "theme_rec_active" if self.recording else "theme_rec_idle")
        if self.recording:
            dpg.set_value("rec_status", f"RECORDING {time.time() - self.rec_t0:.1f} s -> {self.session['dir']}")
        elif self.rec_waiting:
            backlog = sum(self.cams[r].status.get("backlog_mb", 0) for r in self.rec_waiting)
            dpg.set_value("rec_status", f"Finishing the files... (queue {backlog:.1f} MB)")
        elif self.session:
            dpg.set_value("rec_status", f"Last recording: {self.session['dir']}")

        # --- Replay / depth tab
        was_running = self.job_q is not None
        self._poll_job()
        running = self.job is not None and self.job.is_alive() and self.job_q is not None
        if was_running and not running:
            self._view_dirty = True
        dpg.configure_item("btn_est", label="Cancel" if running else "Estimate selected sweep")
        dpg.configure_item("btn_est_all", label="Cancel" if running else "Estimate all sweeps")
        i, n, what = self.job_progress
        if running:
            dpg.set_value("job_progress", i / max(1, n))
            dpg.configure_item("job_progress", overlay=f"{what} ({i}/{n})")
        elif self.shown is not None:
            dpg.set_value("job_progress", 1.0)
            dpg.configure_item("job_progress", overlay=f"sweep #{self.shown.k} ready")
        else:
            dpg.set_value("job_progress", 0.0)
            dpg.configure_item("job_progress", overlay="")
        if self._view_dirty:
            self.redraw_views()
        self.update_hover()
        self._test_hooks(running)

    def _test_hooks(self, running: bool):
        """--select-sweep / --screenshot (automated checks)."""
        if self._pending_select is not None and self.sweeps and not running:
            k, est = self._pending_select
            self._pending_select = None
            self.select_sweep(k, estimate=est)
        if not self.args.screenshot or self._screenshot_due == "done":
            return
        if self.shown is None or running:
            self._screenshot_due = None
        elif self._screenshot_due is None:
            self._screenshot_due = time.time() + 1.5  # Let the new textures render first
        elif time.time() > self._screenshot_due:
            dpg.output_frame_buffer(file=self.args.screenshot)
            self.log(f"Screenshot saved to {self.args.screenshot}")
            self._screenshot_due = "done"
            if self.args.test_viewer_shot:
                self.open_cloud(False)
            if self.args.test_save_png:
                self.save_depth_png()

    def run(self):
        dpg.create_context()
        self.build_ui()
        dpg.create_viewport(title="Laser-stripe stereo depth (method C)", width=1750, height=1000)
        dpg.setup_dearpygui()
        dpg.set_global_font_scale(self.args.ui_scale)
        dpg.set_viewport_resize_callback(self.layout)
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        self.layout()
        sources = self.refresh_cameras()
        if self.args.replay:
            dpg.set_value("tabs", dpg.get_item_children("tabs", 1)[1])
            self.load_recording(self.args.replay)
            if self.args.select_sweep is not None:
                self._pending_select = (self.args.select_sweep, True)
            if self.args.test_views:
                for pane, i in zip(("a", "b"), self.args.test_views.split(",")):
                    dpg.set_value(f"view_{pane}", VIEWS[int(i)])
            if self.args.test_flat:
                dpg.set_value("flat_check", True)
        elif len(sources) == 2 or (self.args.left and self.args.right):
            self.open_cameras()
        t_start = time.time()
        try:
            while dpg.is_dearpygui_running():
                self.update_ui()
                dpg.render_dearpygui_frame()
                if self.args.exit_after and time.time() - t_start > self.args.exit_after:
                    break
        finally:
            if self.any_camera_alive():
                self.close_cameras()
            if self._reaper is not None:
                self._reaper.join(timeout=150)
            if self.job is not None and self.job.is_alive():
                self.job_cancel.set()
                self.job.join(timeout=30)
                if self.job.is_alive():
                    self.job.terminate()
            dpg.destroy_context()


def open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(path))
    except AttributeError:
        webbrowser.open(path.resolve().as_uri())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Laser-stripe stereo depth (method C) with two synchronised event cameras.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--left", default=None, help="Serial number of the left camera.")
    p.add_argument("--right", default=None, help="Serial number of the right camera.")
    p.add_argument("--sync", choices=("hardware", "host"), default="hardware", help="Timestamp synchronization.")
    p.add_argument("--master", choices=ROLES, default="left", help="Master camera with hardware sync.")
    p.add_argument("--trail-filter", action="store_true", help="Enable the sensor trail filter (default: raw events).")
    p.add_argument("--erc-mev", type=float, default=20.0,
                   help="Sensor event rate limit per camera (Mev/s), 0 = off. Protects the recording from overload.")
    p.add_argument("-b", "--biases", default=None, help="Metavision .bias file applied to both cameras.")
    p.add_argument("-o", "--output-dir", default=str(SCRIPT_DIR / "recordings"), help="Folder of new recordings.")
    p.add_argument("--recordings-dir", default=str(STEP4_REC if STEP4_REC.exists() else SCRIPT_DIR / "recordings"),
                   help="Folder where 'Load recording...' starts.")
    p.add_argument("--depth-dir", default=str(SCRIPT_DIR / "output"), help="Root folder of the depth results.")
    p.add_argument("--calib-dir", default=str(STEP2_DIR / "output"), help="Step 2 output folder (stereo calibration).")
    p.add_argument("--viewer-python", default=None,
                   help="Python interpreter with open3d for the point cloud viewer (default: this one).")
    p.add_argument("--replay", default=None, help="Recording folder to open in the Replay tab.")
    p.add_argument("--ui-scale", type=float, default=1.0, help="Global font scale of the UI.")
    p.add_argument("--exit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # Automated tests
    p.add_argument("--select-sweep", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    p.add_argument("--test-views", default="", help=argparse.SUPPRESS)  # e.g. "4,2": view indices of the two panes
    p.add_argument("--test-flat", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--test-viewer-shot", default="", help=argparse.SUPPRESS)
    p.add_argument("--test-save-png", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    App(parse_args()).run()
