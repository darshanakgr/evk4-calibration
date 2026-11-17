#!/usr/bin/env python3
"""
Step 4 - Dual event camera recorder (hardware-synchronized) and replay with pre-processing.

Record tab
  * Two cameras, each in its own process, hardware timestamp sync (master/slave through the SYNC cable) or host clock.
  * Live preview (not rectified). Raw events are saved to one compressed HDF5 file per camera with a producer-consumer
    design: the acquisition loop queues the events, a writer thread compresses and writes them. Saving and preview may
    lag behind; no event is dropped (the queue absorbs the delay, its size is shown).
  * With hardware sync, Start/Stop are given as event timestamps, so both files cover exactly the same time interval.

Replay tab
  * Load a recording, set the pre-processing (accumulation time 1 us - 1 s, ON/OFF/BOTH events, optional
    rectification with a stereo_<left>_<right>.json from step 2) and press "Process": a separate process renders the
    event frames into a temporary HDF5 file. Frames are consecutive windows of the accumulation time:
    N = duration / accumulation (e.g. 5 s at 1000 us = 5000 frames).
  * Time range (start/end, seconds from the recording start): only that part is processed, e.g. to stay within the
    frame limit. Set it with the inputs, the green/red lines on the timeline, or the buttons ("Fit frame limit").
  * "Color by time bins": each frame window is split into 2-10 equal temporal bins and every event takes the color of
    its bin (tab10 palette, same colors in both cameras, polarity ignored), so segments of the two views from the same
    time range have the same color.
  * The frame rate (1-50 fps) is the playback rate: N frames last N / fps seconds (5000 frames at 10 fps = 500 s,
    i.e. 0.01x real time). It can be changed while playing, without processing again.
  * Timeline: event rate of both cameras vs time (two colors); click or drag the yellow line to choose the position,
    then play/pause.

Usage
  python dual_cam_recorder.py                                # two connected cameras, hardware sync, left = master
  python dual_cam_recorder.py --left 00050496 --right 00051788 --master right
  python dual_cam_recorder.py --sync host                    # no sync cable
  python dual_cam_recorder.py --replay path/to/rec_folder    # open a recording in the Replay tab
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import tempfile
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import recorder_camera_process
import recorder_core as rc  # Puts the step 1-3 folders on sys.path
import stereo_core as sc
from stereo_rectify import safe_name, to_texture

SCRIPT_DIR = Path(__file__).resolve().parent
STEP2_DIR = SCRIPT_DIR.parent / "02_stereo_rectification"
TEX_W, TEX_H = 640, 360
LEFT_W = 450
LOG_H = 110
TIMELINE_H = 230
LEGEND_W, LEGEND_H = 960, 26
ROLES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
SYNC_MODES = ("Hardware sync (cable)", "Host clock (no cable)")
SERIES_COLORS = {"left": (255, 160, 40), "right": (60, 200, 255)}  # RGB


def placeholder(text: str, w: int = TEX_W, h: int = TEX_H) -> np.ndarray:
    img = np.full((h, w, 3), 35, np.uint8)
    sc.cc.put_label(img, text, (10, 10), color=(200, 200, 200), bg=(60, 60, 60), scale=0.7)
    return img


def fmt_count(n: float) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.1f} k" if n >= 1e3 else f"{n:.0f}"


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

        # Replay
        self.rec_dir: Optional[Path] = None
        self.rec_files: dict = {}
        self.rec_info: dict = {}
        self.t_origin = 0  # Event time (us) of x = 0 on the timeline
        self.stereo_json = ""
        self.proc = None
        self.proc_q = None
        self.proc_cancel = None
        self.proc_progress = (0, 1)
        self.proc_out = ""
        self.frames: Optional[rc.FrameFile] = None
        self.k = 0
        self.playing = False
        self._play_t0 = 0.0
        self._play_k0 = 0
        self._play_fps = 0.0
        self._shown = {"cams": None, "rec": None}
        self._seq_shown = {}

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
                     "biases": self.args.biases or "", "recorder": "04_dual_cam_recorder"}
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
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except AttributeError:
            webbrowser.open(path.resolve().as_uri())

    def on_sync_mode(self, sender, value, user_data):
        self.sync_hw = value == SYNC_MODES[0]
        dpg.configure_item("combo_master", enabled=self.sync_hw)

    # ==================================================================================================== REPLAY TAB
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
            finally:
                for rd in readers.values():
                    rd.close()
        except Exception as e:
            self.log(f"Failed to load {directory}: {e}")
            return
        self._close_frames()
        self.rec_dir, self.rec_files, self.rec_info, self.t_origin = Path(directory), files, info, t0
        for role in ROLES:
            dpg.set_value(f"tl_{role}", list(series[role]))
            dpg.configure_item(f"tl_{role}", label=f"{role} {info[role]['serial']}")
        dpg.fit_axis_data("tl_x")
        # Explicit y range with headroom: an auto-fit puts a nearly constant rate exactly on the plot border
        y_max = max(max(series[r][1]) if series[r][1] else 0.0 for r in ROLES)
        dpg.set_axis_limits("tl_y", 0.0, max(1e-3, y_max * 1.15))
        dpg.set_value("tl_pos", 0.0)
        self.set_trim(start=0.0, end=(t1 - t0) * 1e-6)  # Full recording by default
        lines = [f"{Path(directory).name}   duration {(t1 - t0) * 1e-6:.2f} s"]
        for role in ROLES:
            i = info[role]
            lines.append(f"{role}: {i['serial']}  {fmt_count(i['n'])} events  ({i['sync']})"
                         + ("" if i["complete"] else "  [INCOMPLETE FILE]"))
        if len({tuple(i["size"]) for i in info.values()}) > 1:
            lines.append("WARNING: different resolutions")
        dpg.set_value("rec_info", "\n".join(lines))
        # Stereo calibration of this camera pair (step 2), in either order
        found = sc.find_stereo_file(self.args.calib_dir, info["left"]["serial"], info["right"]["serial"]) or \
            sc.find_stereo_file(self.args.calib_dir, info["right"]["serial"], info["left"]["serial"])
        if found and not self.stereo_json:
            self.stereo_json = str(found)
        dpg.set_value("stereo_path", self.stereo_json or "(none selected)")
        for role in ROLES:
            dpg.set_value(f"tex_rp_{role}", to_texture(placeholder("Press 'Process' to render the frames"), TEX_W, TEX_H))
        self.log(f"Loaded recording {directory}")

    def load_last_recording(self, *_):
        if self.session and not self.rec_waiting:
            self.load_recording(self.session["dir"])
        else:
            self.log("No finished recording in this session yet.")

    def current_params(self) -> rc.PreprocessParams:
        return rc.PreprocessParams(accumulation_us=int(np.clip(dpg.get_value("acc_us"), 1, 1_000_000)),
                                   polarity=dpg.get_value("polarity"), rectify=bool(dpg.get_value("rectify")),
                                   stereo_json=self.stereo_json if dpg.get_value("rectify") else "",
                                   time_bins=bool(dpg.get_value("time_bins")),
                                   n_bins=int(np.clip(dpg.get_value("n_bins"), rc.MIN_BINS, rc.MAX_BINS)),
                                   start_us=int(round(dpg.get_value("trim_start") * 1e6)),
                                   end_us=int(round(dpg.get_value("trim_end") * 1e6)))

    def playback_fps(self) -> float:
        return float(np.clip(dpg.get_value("fps"), 1, 50))

    def recording_duration_us(self) -> int:
        if not self.rec_info:
            return 0
        return max(i["t_stop"] for i in self.rec_info.values()) - min(i["t_start"] for i in self.rec_info.values())

    def trimmed_duration_us(self) -> int:
        dur = self.recording_duration_us()
        t0, t1 = rc.trim_range(0, dur, int(round(dpg.get_value("trim_start") * 1e6)),
                               int(round(dpg.get_value("trim_end") * 1e6)))
        return max(0, t1 - t0)

    def frame_plan(self):
        """(frames per camera, replay duration s, speed vs real time) for the current settings and time range."""
        return rc.frame_plan(max(1, self.trimmed_duration_us()), dpg.get_value("acc_us"), self.playback_fps())

    # --- time range
    def set_trim(self, start=None, end=None):
        """Updates the time range (seconds from the recording start), keeping 0 <= start < end <= duration."""
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

    def fit_frame_limit(self, *_):
        start = float(dpg.get_value("trim_start"))
        acc = int(np.clip(dpg.get_value("acc_us"), 1, 1_000_000))
        self.set_trim(end=start + rc.MAX_FRAMES * acc * 1e-6)

    def on_process(self, *_):
        if self.proc is not None and self.proc.is_alive():
            self.proc_cancel.set()
            return
        if not self.rec_files:
            self.log("Load a recording first.")
            return
        params = self.current_params()
        n_frames, _, _ = self.frame_plan()
        if n_frames > rc.MAX_FRAMES:
            self.log(f"{n_frames} frames per camera: more than {rc.MAX_FRAMES}. Increase the accumulation time.")
            return
        if params.rectify and not params.stereo_json:
            self.log("Rectification needs a stereo calibration file (step 2): press 'Select calibration...'.")
            return
        self._close_frames()
        self.proc_out = str(Path(tempfile.gettempdir()) / f"dualcam_frames_{os.getpid()}_{int(time.time())}.h5")
        self.proc_q, self.proc_cancel = self.ctx.Queue(), self.ctx.Event()
        self.proc_progress = (0, 1)
        self.proc = self.ctx.Process(target=rc.run_preprocess, daemon=True,
                                     args=({r: str(p) for r, p in self.rec_files.items()}, self.proc_out,
                                           asdict(params), self.proc_q, self.proc_cancel))
        self.proc.start()
        self.log(f"Processing [{params.start_us * 1e-6:.6f}, {params.end_us * 1e-6:.6f}) s: "
                 f"{n_frames} frames of {params.accumulation_us} us, {params.polarity} events"
                 + (f", {params.n_bins} time bins" if params.time_bins else "")
                 + (", rectified" if params.rectify else "") + f" -> {self.proc_out}")

    def _poll_process(self):
        while self.proc_q is not None:  # Cleared when the processing ends
            try:
                msg = self.proc_q.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "progress":
                self.proc_progress = (msg[1], msg[2])
            elif msg[0] == "log":
                self.log(msg[1])
            elif msg[0] == "done":
                s = msg[1]
                self.log(f"Processed {s['frames']} frames in {s['seconds']:.1f} s.")
                self.frames = rc.FrameFile(self.proc_out)
                self.proc_q = None
                fp = self.frames.params
                if fp.time_bins:
                    dpg.set_value("tex_legend", to_texture(rc.bin_legend(fp.accumulation_us, fp.n_bins),
                                                           LEGEND_W, LEGEND_H))
                dpg.configure_item("legend_group", show=fp.time_bins)
                self.seek_time(float(dpg.get_value("tl_pos")))
            elif msg[0] in ("cancelled", "error"):
                self.log("Processing cancelled." if msg[0] == "cancelled" else f"Processing failed: {msg[1]}")
                self.proc_q = None
                self._delete_temp(self.proc_out)

    def _close_frames(self):
        self.playing = False
        if self.frames is not None:
            path = self.frames.path
            self.frames.close()
            self.frames = None
            self._delete_temp(path)

    def _delete_temp(self, path):
        try:
            if path and Path(path).exists():
                Path(path).unlink()
        except OSError:
            pass

    def select_stereo_json(self, path: str):
        self.stereo_json = path
        dpg.set_value("stereo_path", path)

    # --- playback
    def show_frame(self, k: int):
        if self.frames is None:
            return
        ff = self.frames
        self.k = int(np.clip(k, 0, ff.n - 1))
        epi = dpg.get_value("epilines") and ff.params.rectify
        for role in ROLES:
            if role not in ff.roles:
                continue
            img = rc.colorize(ff.frame(role, self.k), ff.params.time_bins)
            if epi:
                sc.draw_epipolar_lines(img, True, step=40)
            dpg.set_value(f"tex_rp_{role}", to_texture(img, TEX_W, TEX_H))
        t0 = (ff.start_time(self.k) - self.t_origin) * 1e-6
        t1 = (ff.times[self.k] - self.t_origin) * 1e-6
        dpg.set_value("tl_pos", t0)  # The yellow line marks the start of the frame window
        counts = " / ".join(fmt_count(ff.counts[r][self.k]) for r in ff.roles)
        dpg.set_value("play_info", f"window [{t0:.6f}, {t1:.6f}) s   frame {self.k + 1}/{ff.n}   "
                                   f"events in frame (L/R): {counts}")

    def seek_time(self, t_rel: float):
        if self.frames is None:
            return
        self.show_frame(self.frames.index_for_time(self.t_origin + t_rel * 1e6))
        self._play_t0, self._play_k0 = time.time(), self.k

    def on_timeline_drag(self, *_):
        self.seek_time(float(dpg.get_value("tl_pos")))

    def on_timeline_click(self, *_):
        x, _ = dpg.get_plot_mouse_pos()
        dpg.set_value("tl_pos", float(x))
        self.seek_time(float(x))

    def toggle_play(self, *_):
        if self.frames is None:
            self.log("Press 'Process' first.")
            return
        self.playing = not self.playing
        if self.playing and self.k >= self.frames.n - 1:
            self.show_frame(0)
        self._play_t0, self._play_k0 = time.time(), self.k

    def step(self, delta: int):
        self.playing = False
        self.show_frame(self.k + delta)

    def _advance_playback(self):
        if not self.playing or self.frames is None:
            return
        fps = self.playback_fps()
        if fps != self._play_fps:  # Rate changed while playing: continue from the current frame
            self._play_fps, self._play_t0, self._play_k0 = fps, time.time(), self.k
        target = self._play_k0 + int((time.time() - self._play_t0) * fps)  # fps frames per second
        if target >= self.frames.n:
            if dpg.get_value("loop"):
                self._play_t0, self._play_k0 = time.time(), 0
                target = 0
            else:
                self.playing = False
                target = self.frames.n - 1
        if target != self.k:
            self.show_frame(target)  # Frames are skipped if rendering is slower than the frame rate

    # ============================================================================================================ UI
    def build_ui(self):
        with dpg.texture_registry():
            for role in ROLES:
                dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder(f"{role} camera closed"), TEX_W, TEX_H),
                                    format=dpg.mvFormat_Float_rgb, tag=f"tex_live_{role}")
                dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder("Load a recording"), TEX_W, TEX_H),
                                    format=dpg.mvFormat_Float_rgb, tag=f"tex_rp_{role}")
            dpg.add_raw_texture(LEGEND_W, LEGEND_H, np.zeros(LEGEND_W * LEGEND_H * 3, np.float32),
                                format=dpg.mvFormat_Float_rgb, tag="tex_legend")
        for tag, colors in (("theme_rec_idle", ((30, 120, 50), (40, 150, 65), (50, 170, 80))),
                            ("theme_rec_active", ((170, 35, 35), (200, 45, 45), (220, 60, 60))),
                            ("theme_action", ((40, 80, 150), (55, 100, 180), (70, 120, 200)))):
            with dpg.theme(tag=tag):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, colors[0])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colors[1])
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colors[2])
        for role in ROLES:
            with dpg.theme(tag=f"theme_series_{role}"):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, SERIES_COLORS[role], category=dpg.mvThemeCat_Plots)

        dlg = dict(show=False, width=750, height=450)
        dpg.add_file_dialog(directory_selector=True, tag="dlg_output", **dlg,
                            callback=lambda s, a: dpg.set_value("output_dir", a["file_path_name"]))
        dpg.add_file_dialog(directory_selector=True, tag="dlg_load", **dlg, default_path=str(self.args.output_dir),
                            callback=lambda s, a: self.load_recording(a["file_path_name"]))
        with dpg.file_dialog(tag="dlg_stereo", **dlg, default_path=str(self.args.calib_dir),
                             callback=lambda s, a: self.select_stereo_json(a["file_path_name"])):
            dpg.add_file_extension(".json")

        a = self.args
        wrap = LEFT_W - 30
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
                # ---------------------------------------------------------------------------------------- Replay
                with dpg.tab(label="Replay"):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=LEFT_W, tag="rp_left", border=True):
                            with dpg.collapsing_header(label="Recording", default_open=True):
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Load recording...",
                                                   callback=lambda: dpg.show_item("dlg_load"))
                                    dpg.add_button(label="Load last recording", callback=self.load_last_recording)
                                dpg.add_text("No recording loaded.", tag="rec_info", wrap=wrap)
                            with dpg.collapsing_header(label="Pre-processing", default_open=True):
                                dpg.add_input_int(label="Accumulation per frame (us, 1-1000000)", tag="acc_us",
                                                  default_value=a.accumulation_us, min_value=1, max_value=1_000_000,
                                                  min_clamped=True, max_clamped=True, step=1000, width=140)
                                dpg.add_text("Time range (s from the recording start):")
                                with dpg.group(horizontal=True):
                                    dpg.add_input_float(label="Start", tag="trim_start", default_value=0.0, width=130,
                                                        format="%.6f", step=0.1, callback=self.on_trim_input,
                                                        on_enter=True)
                                    dpg.add_input_float(label="End", tag="trim_end", default_value=0.0, width=130,
                                                        format="%.6f", step=0.1, callback=self.on_trim_input,
                                                        on_enter=True)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Start = cursor", callback=lambda: self.trim_to_cursor("start"))
                                    dpg.add_button(label="End = cursor", callback=lambda: self.trim_to_cursor("end"))
                                    dpg.add_button(label="Full range",
                                                   callback=lambda: self.set_trim(0.0, self.recording_duration_us() * 1e-6))
                                    dpg.add_button(label="Fit frame limit", callback=self.fit_frame_limit)
                                dpg.add_text("", tag="plan_info", wrap=wrap, color=(150, 220, 150))
                                dpg.add_combo(rc.POLARITIES, label="Events", tag="polarity", default_value="BOTH",
                                              width=140)
                                dpg.add_checkbox(label="Color by time bins (polarity ignored)", tag="time_bins",
                                                 default_value=False,
                                                 callback=lambda s, v: dpg.configure_item("n_bins", enabled=v))
                                dpg.add_input_int(label=f"Bins per frame ({rc.MIN_BINS}-{rc.MAX_BINS})", tag="n_bins",
                                                  default_value=4, min_value=rc.MIN_BINS, max_value=rc.MAX_BINS,
                                                  min_clamped=True, max_clamped=True, width=140, enabled=False)
                                dpg.add_checkbox(label="Rectify (step 2 calibration)", tag="rectify",
                                                 default_value=False)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="Select calibration...",
                                                   callback=lambda: dpg.show_item("dlg_stereo"))
                                dpg.add_text("(none selected)", tag="stereo_path", wrap=wrap, color=(170, 170, 170))
                                b = dpg.add_button(label="Process", tag="btn_process", width=-1, height=34,
                                                   callback=self.on_process)
                                dpg.bind_item_theme(b, "theme_action")
                                dpg.add_progress_bar(tag="proc_progress", default_value=0.0, width=-1, overlay="")
                                dpg.add_text("", tag="proc_status", wrap=wrap, color=(170, 170, 170))
                            with dpg.collapsing_header(label="Playback", default_open=True):
                                dpg.add_input_int(label="Playback rate (fps, 1-50)", tag="fps", default_value=a.fps,
                                                  min_value=1, max_value=50, min_clamped=True, max_clamped=True,
                                                  width=140)
                                dpg.add_text("", tag="replay_info", wrap=wrap, color=(150, 220, 150))
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label="|<", callback=lambda: (self.step(-10 ** 9)))
                                    dpg.add_button(label="<", callback=lambda: self.step(-1))
                                    b = dpg.add_button(label="Play", tag="btn_play", width=80, callback=self.toggle_play)
                                    dpg.bind_item_theme(b, "theme_action")
                                    dpg.add_button(label=">", callback=lambda: self.step(1))
                                    dpg.add_button(label=">|", callback=lambda: self.step(10 ** 9))
                                dpg.add_checkbox(label="Loop", tag="loop", default_value=True)
                                dpg.add_checkbox(label="Epipolar lines (rectified frames)", tag="epilines",
                                                 callback=lambda: self.show_frame(self.k))
                        with dpg.child_window(tag="rp_right", border=False):
                            with dpg.group(horizontal=True):
                                for role in ROLES:
                                    with dpg.group():
                                        dpg.add_text(f"{role.capitalize()} camera", tag=f"lbl_rp_{role}")
                                        dpg.add_image(f"tex_rp_{role}", tag=f"img_rp_{role}", width=TEX_W // 2,
                                                      height=TEX_H // 2)
                            dpg.add_text("", tag="play_info")
                            with dpg.group(horizontal=True, tag="legend_group", show=False):
                                dpg.add_text("Time bins:")
                                dpg.add_image("tex_legend", width=LEGEND_W, height=LEGEND_H)
                            with dpg.plot(label="Event rate (click or drag the yellow line to choose the position)",
                                          tag="timeline_plot", height=TIMELINE_H, width=-1):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="time (s)", tag="tl_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="event rate (Mev/s)", tag="tl_y"):
                                    for role in ROLES:
                                        dpg.add_line_series([], [], label=role, tag=f"tl_{role}")
                                        dpg.bind_item_theme(f"tl_{role}", f"theme_series_{role}")
                                dpg.add_drag_line(label="start", tag="tl_start", color=[60, 220, 90, 255],
                                                  default_value=0.0, callback=self.on_trim_drag)
                                dpg.add_drag_line(label="end", tag="tl_end", color=[235, 70, 70, 255],
                                                  default_value=0.0, callback=self.on_trim_drag)
                                dpg.add_drag_line(label="position", tag="tl_pos", color=[255, 230, 0, 255],
                                                  default_value=0.0, callback=self.on_timeline_drag)
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
        iw = int(min((right_w - 16) / 2, (panel_h - TIMELINE_H - 70 - LEGEND_H - 8) * aspect))
        for role in ROLES:
            dpg.configure_item(f"img_rp_{role}", width=iw, height=int(iw / aspect))

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

        # --- Replay tab
        self._poll_process()
        running = self.proc is not None and self.proc.is_alive() and self.proc_q is not None
        dpg.configure_item("btn_process", label="Cancel" if running else "Process")
        k, n = self.proc_progress
        if running:
            dpg.set_value("proc_progress", k / max(1, n))
            dpg.configure_item("proc_progress", overlay=f"{k}/{n} frames")
        elif self.frames is not None:
            dpg.set_value("proc_progress", 1.0)
            dpg.configure_item("proc_progress", overlay=f"{self.frames.n} frames ready")
            outdated = asdict(self.current_params()) != asdict(self.frames.params)
            dpg.set_value("proc_status", ("Parameters changed: press 'Process' again.\n" if outdated else "")
                          + f"Temporary file: {self.frames.path}")
        if self.rec_info:
            n_frames, _, _ = self.frame_plan()
            warn = f"  - TOO MANY (max {rc.MAX_FRAMES})" if n_frames > rc.MAX_FRAMES else ""
            if n_frames > rc.MAX_FRAMES:
                warn += ": shorten the time range ('Fit frame limit') or increase the accumulation"
            dpg.set_value("plan_info", f"{n_frames} frames per camera for {self.trimmed_duration_us() * 1e-6:.6f} s "
                                       f"(range / accumulation){warn}")
            # Playback: the processed frames if any, otherwise what Process would produce
            fps = self.playback_fps()
            if self.frames is not None:
                n_play, acc = self.frames.n, self.frames.params.accumulation_us
            else:
                n_play, acc = n_frames, int(dpg.get_value("acc_us"))
            dpg.set_value("replay_info", f"Replay {n_play / fps:.1f} s for {n_play} frames "
                                         f"({acc * 1e-6 * fps:.4g}x real time)")
        dpg.configure_item("btn_play", label="Pause" if self.playing else "Play")
        self._advance_playback()

    def run(self):
        dpg.create_context()
        self.build_ui()
        dpg.create_viewport(title="Dual event camera recorder", width=1700, height=980)
        dpg.setup_dearpygui()
        dpg.set_global_font_scale(self.args.ui_scale)
        dpg.set_viewport_resize_callback(self.layout)
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        self.layout()
        sources = self.refresh_cameras()
        if self.args.replay:
            self.load_recording(self.args.replay)
            dpg.set_value("tabs", dpg.get_item_children("tabs", 1)[1])
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
            if self.proc is not None and self.proc.is_alive():
                self.proc_cancel.set()
                self.proc.join(timeout=10)
            self._close_frames()
            dpg.destroy_context()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual event camera recorder (hardware sync) with replay.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--left", default=None, help="Serial number of the left camera.")
    p.add_argument("--right", default=None, help="Serial number of the right camera.")
    p.add_argument("--sync", choices=("hardware", "host"), default="hardware", help="Timestamp synchronization.")
    p.add_argument("--master", choices=ROLES, default="left", help="Master camera with hardware sync.")
    p.add_argument("--trail-filter", action="store_true", help="Enable the sensor trail filter (default: raw events).")
    p.add_argument("--erc-mev", type=float, default=20.0,
                   help="Sensor event rate limit per camera (Mev/s), 0 = off. Protects the recording from overload.")
    p.add_argument("-b", "--biases", default=None, help="Metavision .bias file applied to both cameras.")
    p.add_argument("-o", "--output-dir", default=str(SCRIPT_DIR / "recordings"), help="Folder of the recordings.")
    p.add_argument("--calib-dir", default=str(STEP2_DIR / "output"), help="Step 2 output folder (stereo calibration).")
    p.add_argument("--replay", default=None, help="Recording folder to open in the Replay tab.")
    p.add_argument("--fps", type=int, default=25, help="Default playback rate in frames per second (1-50).")
    p.add_argument("--accumulation-us", type=int, default=20_000, help="Default accumulation time (us).")
    p.add_argument("--ui-scale", type=float, default=1.0, help="Global font scale of the UI.")
    p.add_argument("--exit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # Automated tests
    return p.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    App(parse_args()).run()
