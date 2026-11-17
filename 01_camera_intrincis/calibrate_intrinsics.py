#!/usr/bin/env python3
"""
Intrinsic calibration of a Prophesee EVK4 (or any Metavision camera) with a blinking chessboard.

Python / DearPyGui port of the Metavision SDK C++ samples ``metavision_mono_calibration_recording`` (detection of a
blinking chessboard in the event stream) and ``metavision_mono_calibration`` (OpenCV calibration).

Workflow (repeat for each camera)
  0. Select the camera in the "Camera" dropdown and press "Open camera" (it is opened automatically when only one
     camera is connected). The intrinsics are saved as intrinsics_<serial>.json / intrinsics_<serial>_opencv.yaml,
     in the session folder and (latest calibration of each camera) directly in the output folder.
  1. The script launches the blinking chessboard in the web browser (press F there for
     fullscreen). Measure the size of one square on the screen and enter it in "Square size" (only affects the
     extrinsics/translation scale, not the intrinsics).
  2. Press "Start recording". Point the camera at the screen and keep it still (ideally on a tripod) ~1-2 s per pose:
     a view is captured when the blinking chessboard is detected (green flash). Then move to a new pose: vary the
     distance, tilt the camera (up to ~30-45 deg) and place the board in every region of the image, especially in the
     corners (watch the green coverage overlay).
  3. Press "Stop recording" after ~30-50 views. The detections are saved automatically to the session folder.
  4. Press "Calculate intrinsics", check the per-view errors in the "Views" tab, then press "Save intrinsics".

Usage
  python calibrate_intrinsics.py                          # first available camera
  python calibrate_intrinsics.py --serial 00051788 --biases my_biases.bias
  python calibrate_intrinsics.py --input recording.raw    # replay a RAW file instead of a live camera
  python calibrate_intrinsics.py --cols 9 --rows 6 --square-size-mm 25
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from typing import List, Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

import calib_core as cc
from camera_utils import EVENT_PALETTE, RealTimeGuard, apply_bias_file, configure_hw_filters

try:  # Metavision SDK (4.x) Python bindings
    from metavision_core.event_io import EventsIterator
    from metavision_core.event_io.live_replay import LiveReplayEventsIterator
    from metavision_core.event_io.raw_reader import initiate_device
    from metavision_hal import DeviceDiscovery
    from metavision_sdk_cv import TrailFilterAlgorithm
    METAVISION_IMPORT_ERROR = None
except ImportError as _e:  # The UI can still be used offline (load detections + calibrate)
    METAVISION_IMPORT_ERROR = _e

SCRIPT_DIR = Path(__file__).resolve().parent
PATTERN_TEMPLATE = SCRIPT_DIR / "chessboard_pattern.html"

BUFFER_US = 10_000  # Duration of the event buffers read from the camera
TRAIL_FILTER_US = 1_000_000  # Software trail filter threshold (same as the C++ sample), used for RAW file replay
EVENT_DISPLAY_PERIOD_US = 40_000  # Accumulation time of the event display (25 fps)
TEX_W, TEX_H = 800, 450  # Live display textures (16:9, letterboxed for other sensor ratios)
VIEW_TEX_W, VIEW_TEX_H = 1024, 576  # "Views" tab texture
LEFT_W = 440
LOG_H = 150  # Height of the log panel (below the images)

# BGR colors
COVERAGE_COLOR = np.array([60, 200, 60], np.float32)
ORANGE, GREEN, WHITE = (0, 165, 255), (0, 220, 0), (255, 255, 255)


def to_texture(img_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Letterboxes a BGR image into a (th, tw) RGB float32 buffer, as expected by dearpygui raw textures."""
    h, w = img_bgr.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), np.uint8)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)).ravel()


def placeholder(text: str, w: int = 1280, h: int = 720) -> np.ndarray:
    img = np.full((h, w, 3), 35, np.uint8)
    cc.put_label(img, text, (20, 20), color=(200, 200, 200), bg=(60, 60, 60), scale=1.2)
    return img


# ----------------------------------------------------------------------------------------------------------------------
# Camera thread
# ----------------------------------------------------------------------------------------------------------------------
class CameraWorker(threading.Thread):
    """Reads events, runs the blinking chessboard detector and produces the display images."""

    def __init__(self, app: "App", input_path: Optional[str], serial: Optional[str], bias_file: Optional[str],
                 hw_filters: bool = True, erc_rate_mev: float = 20.0):
        super().__init__(name="camera", daemon=True)
        self.app = app
        self.input_path = input_path
        self.serial = serial
        self.bias_file = bias_file
        self.hw_filters = hw_filters
        self.erc_rate_mev = erc_rate_mev
        self.is_live = not input_path
        self._sw_trail = True  # Software trail filter, disabled when the sensor one is active
        self._stop_evt = threading.Event()
        self._commands: "queue.Queue[tuple]" = queue.Queue()
        self._raw_logging = False

    def stop(self) -> None:
        self._stop_evt.set()

    def send(self, *cmd) -> None:
        self._commands.put(cmd)

    def run(self) -> None:
        app = self.app
        device = mv_it = events_stream = None
        try:
            if self.is_live:
                app.log("Opening camera" + (f" {self.serial}" if self.serial else "") + "...")
                device = initiate_device(self.serial or "")
                serial = device.get_i_hw_identification().get_serial()
                if self.bias_file:
                    app.log(f"Loading biases from {self.bias_file}")
                    apply_bias_file(device, self.bias_file, app.log)
                if self.hw_filters:
                    self._sw_trail = not configure_hw_filters(device, self.erc_rate_mev, app.log)
                events_stream = device.get_i_events_stream()
                mv_it = EventsIterator.from_device(device, delta_t=BUFFER_US)
                source = f"Camera {serial}"
            else:
                if self.input_path.lower().endswith(".raw"):  # RAW files know the serial of the recording camera
                    device = initiate_device(self.input_path)
                    hw_id = device.get_i_hw_identification()
                    serial = hw_id.get_serial() if hw_id is not None else ""
                    events = EventsIterator.from_device(device, delta_t=BUFFER_US)
                else:
                    serial = ""
                    events = EventsIterator(self.input_path, delta_t=BUFFER_US)
                mv_it = LiveReplayEventsIterator(events)
                source = f"File {Path(self.input_path).name}"
            height, width = mv_it.get_size()
            app.on_camera_started(width, height, serial, source)
            self._loop(mv_it, width, height, events_stream)
        except Exception as e:
            app.log(f"[camera] ERROR: {e}")
            print(traceback.format_exc())
        finally:
            if self._raw_logging and events_stream is not None:
                try:
                    events_stream.stop_log_raw_data()
                except Exception:
                    pass
            mv_it = events_stream = device = None  # Releases the camera
            app.on_camera_stopped(finished=not self._stop_evt.is_set())

    def _process_commands(self, events_stream) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            if cmd[0] == "start_raw":
                if events_stream is None:
                    self.app.log("RAW saving is only available with a live camera.")
                elif events_stream.log_raw_data(cmd[1]):
                    self._raw_logging = True
                    self.app.log(f"Saving RAW events to {cmd[1]}")
                else:
                    self.app.log(f"Failed to start RAW logging to {cmd[1]}")
            elif cmd[0] == "stop_raw" and self._raw_logging:
                events_stream.stop_log_raw_data()
                self._raw_logging = False
                self.app.log("RAW logging stopped.")

    def _loop(self, mv_it, width: int, height: int, events_stream) -> None:
        app = self.app
        trail = TrailFilterAlgorithm(width, height, TRAIL_FILTER_US) if self._sw_trail else None
        filtered_buf = TrailFilterAlgorithm.get_empty_output_buffer() if self._sw_trail else None
        detector, det_key = None, None
        ev_state = np.zeros(width * height, np.uint8)
        next_display_ts = None
        rate_count, rate_t0 = 0, time.time()
        guard = RealTimeGuard()
        last_skip_log = 0.0

        for evs in mv_it:
            if self._stop_evt.is_set():
                break
            self._process_commands(events_stream)
            rate_count += evs.size
            now = time.time()
            if now - rate_t0 >= 1.0:
                app.event_rate = rate_count / (now - rate_t0)
                app.skipped_ratio = guard.pop_skipped_ratio()
                rate_count, rate_t0 = 0, now
            if evs.size == 0:
                continue

            # Real-time guard: never let a backlog build up (it shows as an ever-growing display delay)
            t_last = int(evs["t"][-1])
            skip = guard.update(t_last)
            app.processing_lag = guard.lag
            if guard.resumed and detector is not None:
                detector.generator.reset()  # The current window is missing events: start a fresh one
            if skip:
                if now - last_skip_log > 10:
                    last_skip_log = now
                    app.log(f"Processing is {guard.lag:.2f} s behind the camera ({app.event_rate / 1e6:.1f} Mev/s): "
                            "skipping buffers to stay real-time. Lower --erc-mev or tune the biases if it persists.")
                continue

            if trail is not None:
                trail.process_events(evs, filtered_buf)
                evs = filtered_buf.numpy()
                if evs.size == 0:
                    continue
            idx = evs["y"].astype(np.int64) * width + evs["x"].astype(np.int64)
            pol = evs["p"] > 0

            key = (app.cols, app.rows)
            if key != det_key:  # Pattern geometry changed: new detector (it reads app.blink_config live)
                detector, det_key = cc.BlinkingChessboardDetector(width, height, key[0], key[1], app.blink_config), key
            for window in detector.process_events(evs, idx, pol):
                app.on_window(window, key)

            ev_state[idx] = pol.view(np.uint8) + 1  # 1 = OFF, 2 = ON
            if next_display_ts is None or t_last >= next_display_ts:
                app.publish_events(EVENT_PALETTE[ev_state].reshape(height, width, 3))
                ev_state[:] = 0
                next_display_ts = t_last + EVENT_DISPLAY_PERIOD_US


# ----------------------------------------------------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------------------------------------------------
class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._log_lines: List[str] = []
        self._scroll_log = 0

        # Pattern
        self.cols, self.rows = args.cols, args.rows
        self.square_mm = args.square_size_mm

        # Detection
        self.blink_config = cc.BlinkingConfig(accumulation_time_us=int(args.accumulation_ms * 1000),
                                              min_blinking_pixels=args.min_blink_pixels,
                                              ratio_on=args.ratio_on, ratio_off=args.ratio_off)
        self.collector = cc.ViewCollector(skip_time_us=int(args.skip_time_s * 1e6), min_shift_px=args.min_shift_px)

        # Camera
        self.camera_sources: dict = {}  # Dropdown label -> ("camera", serial) or ("file", path)
        self.worker: Optional[CameraWorker] = None
        self.camera_running = False
        self.cam_size: Optional[tuple] = None  # (width, height)
        self.cam_serial = ""
        self.cam_source = ""
        self.event_rate = 0.0
        self.processing_lag = 0.0  # Delay (s) between the camera and the processing
        self.skipped_ratio = 0.0  # Fraction of buffers skipped by the real-time guard during the last second

        # Recording / views
        self.recording = False
        self.views: List[cc.CalibrationView] = []
        self.views_version = 0
        self.coverage: Optional[np.ndarray] = None
        self.image_size: Optional[tuple] = None  # (width, height) of the collected views
        self.views_serial = ""  # Serial number of the camera the views come from
        self.views_saved_version = 0  # views_version at the last save/load of the detections
        self.session_dir: Optional[Path] = None
        self.raw_index = 0
        self.flash_until = 0.0

        # Calibration
        self.calib_result: Optional[cc.CalibrationResult] = None
        self.calib_views_version = -1
        self.calib_running = False
        self.calib_version = 0  # Incremented on every new result

        # Display buffers produced by the camera thread
        self._ev_tex = None
        self._ev_seq = 0
        self._pat_tex = None
        self._pat_seq = 0

        # Last state pushed to the UI
        self._shown = {"ev": -1, "pat": -1, "rec": None, "views": -1, "calib": -1, "cam": None, "locked": None}

    # ---------------------------------------------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log_queue.put(line)

    # ---------------------------------------------------------------------------------------------- camera callbacks
    def on_camera_started(self, width: int, height: int, serial: str, source: str) -> None:
        with self.lock:
            if self.views and self.views_serial and serial and self.views_serial != serial:
                # Never mix views of different cameras: keep the previous camera's detections on disk and start over
                old_serial, n_old = self.views_serial, len(self.views)
                if self.views_saved_version != self.views_version:
                    self._save_detections_now()
                old_session = self.session_dir
                self._reset_views()
                self.log(f"Camera changed: the {n_old} views of camera {old_serial} were cleared "
                         f"(detections saved in {old_session}).")
            self.camera_running = True
            self.cam_size = (width, height)
            self.cam_serial = serial
            self.cam_source = source
            if self.coverage is None or self.coverage.shape != (height, width):
                self.coverage = cc.coverage_map([v.corners for v in self.views], width, height)
        self.log(f"{source} started ({width}x{height}).")

    def on_camera_stopped(self, finished: bool) -> None:
        was_recording = self.recording
        with self.lock:
            self.camera_running = False
            self.recording = False
            self.event_rate = 0.0
        self.log("Playback finished." if (finished and self.args.input) else "Camera stopped.")
        if was_recording:
            self._save_detections_async()

    def publish_events(self, img: np.ndarray) -> None:
        tex = to_texture(img, TEX_W, TEX_H)
        with self.lock:
            self._ev_tex = tex
            self._ev_seq += 1

    def on_window(self, win: cc.BlinkingWindow, pattern: tuple) -> None:
        """Called by the camera thread for every accumulation window (10 Hz with the default 100 ms)."""
        cols, rows = pattern
        captured = False
        with self.lock:
            if win.frame is None:
                status, color = "No blinking pattern  (keep the camera still and pointed at the screen)", ORANGE
            elif win.corners is None:
                status, color = f"Blinking detected - {cols}x{rows} chessboard not found", ORANGE
            elif self.recording:
                ok, reason = self.collector.accept(win, [v.corners for v in self.views])
                if ok:
                    self._add_view(cc.CalibrationView(win.corners.copy(), win.frame.copy(), win.ts))
                    captured = True
                    status, color = f"View {len(self.views)} captured!", GREEN
                else:
                    status, color = f"Chessboard detected - {reason}", GREEN if "skip" in reason else ORANGE
            else:
                status, color = "Chessboard detected  (press 'Start recording' to capture views)", GREEN
            coverage, n_views, recording = self.coverage, len(self.views), self.recording
            flash = time.time() < self.flash_until
        if captured:
            self.log(f"View {n_views} captured.")

        h, w = coverage.shape if coverage is not None else (720, 1280)
        img = cv2.cvtColor(win.frame, cv2.COLOR_GRAY2BGR) if win.frame is not None else np.full((h, w, 3), 70, np.uint8)
        if coverage is not None and coverage.shape == img.shape[:2] and n_views:
            alpha = (np.minimum(coverage, 4).astype(np.float32) * (0.5 / 4))[..., None]
            img = (img * (1 - alpha) + COVERAGE_COLOR * alpha).astype(np.uint8)
        if win.corners is not None and len(win.corners) == cols * rows:
            cv2.drawChessboardCorners(img, (cols, rows), win.corners.reshape(-1, 1, 2), True)
        if flash:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 255, 0), 18)
        cc.put_label(img, status, (10, 10), color=color)
        cc.put_label(img, f"Views: {n_views}" + ("   [RECORDING]" if recording else ""), (10, 55),
                     color=(80, 80, 255) if recording else WHITE)
        tex = to_texture(img, TEX_W, TEX_H)
        with self.lock:
            self._pat_tex = tex
            self._pat_seq += 1

    def _add_view(self, view: cc.CalibrationView) -> None:  # Called with the lock held
        self.views.append(view)
        h, w = view.frame.shape[:2]
        if self.coverage is None or self.coverage.shape != (h, w):
            self.coverage = np.zeros((h, w), np.uint16)
        cc.add_to_coverage(self.coverage, view.corners)
        self.views_version += 1
        self.flash_until = time.time() + 0.4

    def _views_changed(self) -> None:  # Called with the lock held
        self.views_version += 1
        size = self.image_size or self.cam_size
        if size is not None:
            self.coverage = cc.coverage_map([v.corners for v in self.views], size[0], size[1])

    def _reset_views(self) -> None:  # Called with the lock held
        self.views = []
        self.image_size = None
        self.views_serial = ""
        self.session_dir = None  # The next recording goes to a new session folder
        self.calib_result = None
        self.calib_version += 1
        self._views_changed()
        self.views_saved_version = self.views_version

    # ---------------------------------------------------------------------------------------------------- actions
    def refresh_cameras(self, *_):
        """Lists the connected cameras (and the --input file) in the camera dropdown."""
        sources = {}
        if self.args.input:
            sources[f"File: {Path(self.args.input).name}"] = ("file", self.args.input)
        if METAVISION_IMPORT_ERROR is None:
            try:
                for uid in DeviceDiscovery.list():
                    serial = uid.split(":")[-1]
                    sources[serial] = ("camera", serial)
            except Exception as e:
                self.log(f"Camera discovery failed: {e}")
        # Keep the open camera in the list even if the discovery does not report it while it is in use
        if self.camera_running and self.cam_serial and not self.args.input and self.cam_serial not in sources:
            sources[self.cam_serial] = ("camera", self.cam_serial)
        self.camera_sources = sources
        labels = list(sources)
        current = dpg.get_value("camera_combo")
        dpg.configure_item("camera_combo", items=labels)
        if current not in sources:
            preferred = self.args.serial if self.args.serial in sources else (labels[0] if labels else "")
            dpg.set_value("camera_combo", preferred)
        n_cams = sum(kind == "camera" for kind, _ in sources.values())
        self.log(f"{n_cams} camera(s) found" + (": " + ", ".join(s for k, s in sources.values() if k == "camera")
                                                if n_cams else "."))
        return sources

    def start_camera(self, *_):
        if self.worker is not None and self.worker.is_alive():
            return
        if METAVISION_IMPORT_ERROR is not None:
            self.log(f"Metavision SDK Python modules not available: {METAVISION_IMPORT_ERROR}")
            return
        choice = self.camera_sources.get(dpg.get_value("camera_combo"))
        if choice is None:
            self.log("No camera selected: connect a camera and press 'Refresh'.")
            return
        kind, value = choice
        self.worker = CameraWorker(self, value if kind == "file" else None, value if kind == "camera" else None,
                                   self.args.biases, hw_filters=not self.args.no_hw_filters,
                                   erc_rate_mev=self.args.erc_mev)
        self.worker.start()

    def stop_camera(self, *_):
        if self.recording:
            self.stop_recording()
        if self.worker is not None:
            self.worker.stop()

    def toggle_camera(self, *_):
        if self.worker is not None and self.worker.is_alive():
            self.stop_camera()
        else:
            self.start_camera()

    def launch_chessboard(self, *_):
        try:
            html = PATTERN_TEMPLATE.read_text(encoding="utf-8")
            html = html.replace("__COLS__", str(self.cols)).replace("__ROWS__", str(self.rows))
            out_dir = self._output_root()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"chessboard_{self.cols}x{self.rows}.html"
            path.write_text(html, encoding="utf-8")
            webbrowser.open(path.resolve().as_uri())
            self.log(f"Chessboard launched in the browser ({self.cols}x{self.rows} inner corners). "
                     "Press F in the browser for fullscreen.")
        except Exception as e:
            self.log(f"Failed to launch the chessboard: {e}")

    def toggle_recording(self, *_):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        with self.lock:
            if not self.camera_running:
                self.log("Cannot record: the camera is not running.")
                return
            if self.views and self.views_serial and self.cam_serial and self.views_serial != self.cam_serial:
                self.log(f"Cannot record: the current views come from camera {self.views_serial}, the open camera "
                         f"is {self.cam_serial}. Clear the views first.")
                return
            if self.views and self.image_size not in (None, self.cam_size):
                self.log(f"Cannot record: the loaded views are {self.image_size}, the camera is {self.cam_size}. "
                         "Clear the views first.")
                return
            self.image_size = self.cam_size
            if not self.views:
                self.views_serial = self.cam_serial
                if self.session_dir is not None and not any(self.session_dir.iterdir()):
                    self.session_dir = None  # Empty folder created for another camera: rename for this one
            session = self._ensure_session_dir()
            self.collector.reset_timer()
            self.recording = True
        if dpg.get_value("save_raw") and self.worker is not None and self.worker.is_live:
            self.raw_index += 1
            self.worker.send("start_raw", str(session / f"recording_{self.raw_index:03d}.raw"))
        self.log("Recording started: hold the camera still for ~1-2 s at each pose.")

    def stop_recording(self):
        with self.lock:
            if not self.recording:
                return
            self.recording = False
        if self.worker is not None:
            self.worker.send("stop_raw")
        self.log(f"Recording stopped ({len(self.views)} views).")
        self._save_detections_async()

    def delete_last_view(self, *_):
        with self.lock:
            if not self.views:
                return
            self.views.pop()
            self._views_changed()
            n = len(self.views)
        self.log(f"Last view deleted ({n} views left).")

    def clear_views(self, *_):
        with self.lock:
            if self.recording:
                self.log("Stop the recording before clearing the views.")
                return
            self._reset_views()
        self.log("All views cleared.")

    def calculate(self, *_):
        with self.lock:
            if self.calib_running:
                return
            if len(self.views) < 3:
                self.log(f"At least 3 views are needed (got {len(self.views)}); 30-50 are recommended.")
                return
            views = list(self.views)
            version = self.views_version
            size = self.image_size or self.cam_size
            cols, rows, sq = self.cols, self.rows, self.square_mm / 1000.0
            self.calib_running = True
        model = dpg.get_value("model")
        outlier = float(dpg.get_value("outlier_ths"))
        refine = bool(dpg.get_value("refine"))
        if len(views) < 15:
            self.log(f"Warning: only {len(views)} views, the result may be inaccurate (30-50 recommended).")

        def job():
            try:
                res = cc.run_calibration([v.corners for v in views], size, cols, rows, sq, sq, model=model,
                                         outlier_ths=outlier, refine=refine, frames=[v.frame for v in views],
                                         log=self.log)
                with self.lock:
                    self.calib_result = res
                    self.calib_views_version = version
                    self.calib_version += 1
                self.log(f"Calibration done in {res.duration_s:.1f} s: RMS = {res.rms:.4f} px, "
                         f"fx = {res.K[0, 0]:.2f}, fy = {res.K[1, 1]:.2f}, cx = {res.K[0, 2]:.2f}, "
                         f"cy = {res.K[1, 2]:.2f}")
            except Exception as e:
                self.log(f"Calibration failed: {e}")
                print(traceback.format_exc())
            finally:
                with self.lock:
                    self.calib_running = False

        threading.Thread(target=job, name="calibration", daemon=True).start()

    def save_intrinsics(self, *_):
        with self.lock:
            res = self.calib_result
            if res is None:
                self.log("Nothing to save: press 'Calculate intrinsics' first.")
                return
            if self.calib_views_version != self.views_version:
                self.log("The views changed since the last calculation: press 'Calculate intrinsics' again.")
                return
            session = self._ensure_session_dir()
            views = list(self.views)
            version = self.views_version
            size = self.image_size or self.cam_size
            cols, rows, sq_mm = self.cols, self.rows, self.square_mm
            serial = self._serial_for_files()
        try:
            cc.save_detections(session, size, cols, rows, sq_mm / 1000, sq_mm / 1000, views, self.views_serial)
            with self.lock:
                self.views_saved_version = version
            meta = {
                "camera_serial": serial,
                "source": self.cam_source,
                "units": "square size and translations in meters",
                "detection": {
                    "accumulation_time_us": self.blink_config.accumulation_time_us,
                    "min_blinking_pixels": self.blink_config.min_blinking_pixels,
                    "ratio_on": self.blink_config.ratio_on,
                    "ratio_off": self.blink_config.ratio_off,
                },
            }
            name = f"intrinsics_{serial}"
            json_path, yaml_path = cc.save_intrinsics(session, res, meta, name=name)
            with open(session / f"calibration_summary_{serial}.txt", "w") as f:
                f.write(f"Camera serial: {serial}\n" + res.summary() + "\n")
            # Latest calibration of each camera, directly in the output folder (one file per serial number)
            latest_json, _ = cc.save_intrinsics(self._output_root(), res, meta, name=name)
            self.log(f"Intrinsics of camera {serial} saved to {json_path} and {yaml_path.name}")
            self.log(f"Latest intrinsics of camera {serial}: {latest_json} (+ {name}_opencv.yaml)")
        except Exception as e:
            self.log(f"Failed to save the intrinsics: {e}")

    def load_detections(self, directory: str):
        if self.recording:
            self.log("Stop the recording before loading detections.")
            return
        try:
            size, pattern, views, serial = cc.load_detections(Path(directory))
        except Exception as e:
            self.log(f"Failed to load detections from {directory}: {e}")
            return
        with self.lock:
            self.cols, self.rows = int(pattern["n_cols"]), int(pattern["n_rows"])
            self.square_mm = float(pattern["square_width"]) * 1000
            self.views = views
            self.image_size = size
            self.views_serial = serial
            self.session_dir = Path(directory)
            self.calib_result = None
            self.calib_version += 1
            self._views_changed()
            self.views_saved_version = self.views_version
        if not serial:
            self.log("These detections have no camera serial number: the intrinsics will be saved under the "
                     "serial of the open camera (or 'unknown').")
        dpg.set_value("cols", self.cols)
        dpg.set_value("rows", self.rows)
        dpg.set_value("square_mm", self.square_mm)
        n_img = sum(v.frame is not None for v in views)
        self.log(f"Loaded {len(views)} views ({size[0]}x{size[1]}, {self.cols}x{self.rows}, "
                 f"{n_img} images) from {directory}")

    def open_session_folder(self, *_):
        path = self.session_dir or self._output_root()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # Windows
        except AttributeError:
            webbrowser.open(path.resolve().as_uri())

    # ---------------------------------------------------------------------------------------------------- helpers
    def _output_root(self) -> Path:
        value = dpg.get_value("output_dir") if dpg.does_item_exist("output_dir") else self.args.output_dir
        return Path(value or self.args.output_dir)

    def _serial_for_files(self) -> str:
        """Serial number used in file names: the camera the views come from, else the open camera."""
        serial = self.views_serial or self.cam_serial or "unknown"
        return re.sub(r"[^A-Za-z0-9_-]", "_", serial)

    def _ensure_session_dir(self) -> Path:
        with self.lock:
            if self.session_dir is None:
                name = time.strftime(f"calib_{self._serial_for_files()}_%Y%m%d_%H%M%S")
                self.session_dir = self._output_root() / name
            self.session_dir.mkdir(parents=True, exist_ok=True)
            return self.session_dir

    def _save_detections_now(self) -> None:
        with self.lock:
            if not self.views:
                return
            views, version = list(self.views), self.views_version
            session = self._ensure_session_dir()
            size = self.image_size or self.cam_size
            cols, rows, sq, serial = self.cols, self.rows, self.square_mm / 1000, self.views_serial
        try:
            path = cc.save_detections(session, size, cols, rows, sq, sq, views, serial)
            with self.lock:
                if self.views_version == version:
                    self.views_saved_version = version
            self.log(f"{len(views)} detections saved to {path}")
        except Exception as e:
            self.log(f"Failed to save detections: {e}")

    def _save_detections_async(self):
        threading.Thread(target=self._save_detections_now, name="save", daemon=True).start()

    # ---------------------------------------------------------------------------------------------------- UI callbacks
    def on_pattern_changed(self, sender, value, user_data):
        with self.lock:
            if user_data == "square_mm":
                self.square_mm = max(0.1, float(value))
                return
            if self.views:
                dpg.set_value(sender, self.cols if user_data == "cols" else self.rows)
                self.log("Clear the views before changing the pattern geometry.")
                return
            value = max(2, int(value))
            if user_data == "cols":
                self.cols = value
            else:
                self.rows = value
        self.log(f"Pattern set to {self.cols}x{self.rows} inner corners. Press 'Launch chessboard' to update it.")

    def on_param_changed(self, sender, value, user_data):
        cfg = self.blink_config
        if user_data == "acc_ms":
            cfg.accumulation_time_us = int(max(5, value) * 1000)
        elif user_data == "min_px":
            cfg.min_blinking_pixels = int(max(1, value))
        elif user_data == "ratio_on":
            cfg.ratio_on = float(value)
        elif user_data == "ratio_off":
            cfg.ratio_off = float(value)
        elif user_data == "skip_s":
            self.collector.skip_time_us = int(max(0.0, value) * 1e6)
        elif user_data == "min_shift":
            self.collector.min_shift_px = float(max(0.0, value))

    def on_model_changed(self, sender, value, user_data):
        dpg.configure_item("refine", enabled=(value == cc.PINHOLE))
        if value != cc.PINHOLE:
            dpg.set_value("refine", False)

    # ---------------------------------------------------------------------------------------------------- UI
    def build_ui(self):
        with dpg.texture_registry():
            dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder("Camera not started"), TEX_W, TEX_H),
                                format=dpg.mvFormat_Float_rgb, tag="tex_events")
            dpg.add_raw_texture(TEX_W, TEX_H, to_texture(placeholder("Waiting for events..."), TEX_W, TEX_H),
                                format=dpg.mvFormat_Float_rgb, tag="tex_pattern")
            dpg.add_raw_texture(VIEW_TEX_W, VIEW_TEX_H,
                                to_texture(placeholder("No views captured yet"), VIEW_TEX_W, VIEW_TEX_H),
                                format=dpg.mvFormat_Float_rgb, tag="tex_view")

        with dpg.theme(tag="theme_rec_idle"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 120, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (40, 150, 65))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 170, 80))
        with dpg.theme(tag="theme_rec_active"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 35, 35))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 45, 45))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (220, 60, 60))
        with dpg.theme(tag="theme_action"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 80, 150))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 100, 180))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (70, 120, 200))

        dpg.add_file_dialog(directory_selector=True, show=False, tag="dlg_output", width=750, height=450,
                            callback=lambda s, a: dpg.set_value("output_dir", a["file_path_name"]))
        dpg.add_file_dialog(directory_selector=True, show=False, tag="dlg_load", width=750, height=450,
                            callback=lambda s, a: self.load_detections(a["file_path_name"]))

        a = self.args
        with dpg.window(tag="main"):
            with dpg.group(horizontal=True):
                # ------------------------------------------------------------------ left panel (controls)
                with dpg.child_window(width=LEFT_W, tag="left_panel", border=True):
                    with dpg.collapsing_header(label="How to", default_open=False):
                        dpg.add_text(
                            "0. Select the camera in the dropdown and press 'Open camera'.\n"
                            "1. Put the browser chessboard in fullscreen (F) and measure one square.\n"
                            "2. Press 'Start recording'.\n"
                            "3. Hold the camera still ~1-2 s per pose until 'captured'. Change the pose: "
                            "distance, tilt (up to 30-45 deg), and position so the whole image gets covered "
                            "(green overlay), especially the corners.\n"
                            "4. Stop after 30-50 views, press 'Calculate intrinsics', check the 'Views' tab, "
                            "then 'Save intrinsics'.", wrap=LEFT_W - 30)

                    with dpg.collapsing_header(label="Chessboard pattern", default_open=True):
                        dpg.add_input_int(label="Columns (inner corners)", tag="cols", default_value=self.cols,
                                          width=120, callback=self.on_pattern_changed, user_data="cols",
                                          on_enter=True)
                        dpg.add_input_int(label="Rows (inner corners)", tag="rows", default_value=self.rows,
                                          width=120, callback=self.on_pattern_changed, user_data="rows",
                                          on_enter=True)
                        dpg.add_input_float(label="Square size (mm)", tag="square_mm", default_value=self.square_mm,
                                            width=120, format="%.2f", step=0.5, callback=self.on_pattern_changed,
                                            user_data="square_mm")
                        btn = dpg.add_button(label="Launch chessboard", width=-1, callback=self.launch_chessboard)
                        dpg.bind_item_theme(btn, "theme_action")

                    with dpg.collapsing_header(label="Camera", default_open=True):
                        with dpg.group(horizontal=True):
                            dpg.add_combo([], label="", tag="camera_combo", width=LEFT_W - 120)
                            dpg.add_button(label="Refresh", tag="btn_refresh", width=-1,
                                           callback=lambda: self.refresh_cameras())
                        btn = dpg.add_button(label="Open camera", tag="btn_camera", width=-1, height=30,
                                             callback=self.toggle_camera)
                        dpg.bind_item_theme(btn, "theme_action")
                        dpg.add_text("", tag="cam_status")
                        dpg.add_text("", tag="cam_rate")

                    with dpg.collapsing_header(label="Detection parameters", default_open=False):
                        dpg.add_input_float(label="Accumulation time (ms)", default_value=a.accumulation_ms, width=120,
                                            format="%.0f", step=10, callback=self.on_param_changed, user_data="acc_ms")
                        dpg.add_input_int(label="Min. blinking pixels", default_value=a.min_blink_pixels, width=120,
                                          step=50, callback=self.on_param_changed, user_data="min_px")
                        dpg.add_input_float(label="Ratio ON", default_value=a.ratio_on, width=120, format="%.2f",
                                            step=0.05, callback=self.on_param_changed, user_data="ratio_on")
                        dpg.add_input_float(label="Ratio OFF", default_value=a.ratio_off, width=120, format="%.2f",
                                            step=0.05, callback=self.on_param_changed, user_data="ratio_off")
                        dpg.add_input_float(label="Skip time between views (s)", default_value=a.skip_time_s,
                                            width=120, format="%.1f", step=0.5, callback=self.on_param_changed,
                                            user_data="skip_s")
                        dpg.add_input_float(label="Min. pose change (px)", default_value=a.min_shift_px, width=120,
                                            format="%.1f", step=1, callback=self.on_param_changed,
                                            user_data="min_shift")
                        dpg.add_text("No detection? Increase the accumulation time, decrease the min. "
                                     "blinking pixels or increase the ON/OFF ratios.", wrap=LEFT_W - 30,
                                     color=(170, 170, 170))

                    with dpg.collapsing_header(label="Recording", default_open=True):
                        dpg.add_text("Output folder:")
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="output_dir", default_value=str(a.output_dir), width=LEFT_W - 110)
                            dpg.add_button(label="Browse", callback=lambda: dpg.show_item("dlg_output"))
                        dpg.add_checkbox(label="Also save RAW events (large files!)", tag="save_raw",
                                         default_value=a.save_raw)
                        dpg.add_button(label="Start recording", tag="btn_record", width=-1, height=42,
                                       callback=self.toggle_recording)
                        dpg.bind_item_theme("btn_record", "theme_rec_idle")
                        dpg.add_text("Views captured: 0", tag="views_count")
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Delete last view", callback=self.delete_last_view)
                            dpg.add_button(label="Clear all views", callback=self.clear_views)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Load detections...", callback=lambda: dpg.show_item("dlg_load"))
                            dpg.add_button(label="Open folder", callback=self.open_session_folder)
                        dpg.add_text("", tag="session_label", wrap=LEFT_W - 30, color=(170, 170, 170))

                    with dpg.collapsing_header(label="Calibration", default_open=True):
                        dpg.add_combo([cc.PINHOLE, cc.FISHEYE], label="Camera model", tag="model",
                                      default_value=cc.FISHEYE if a.fisheye else cc.PINHOLE, width=120,
                                      callback=self.on_model_changed)
                        dpg.add_input_float(label="Outlier threshold (std)", tag="outlier_ths",
                                            default_value=a.outlier_ths, width=120, format="%.1f", step=0.5)
                        dpg.add_checkbox(label="Refine corners (2nd pass, pinhole only)", tag="refine",
                                         default_value=not a.fisheye, enabled=not a.fisheye)
                        with dpg.group(horizontal=True):
                            b1 = dpg.add_button(label="Calculate intrinsics", tag="btn_calc", width=(LEFT_W - 40) // 2,
                                                height=34, callback=self.calculate)
                            b2 = dpg.add_button(label="Save intrinsics", width=(LEFT_W - 40) // 2, height=34,
                                                callback=self.save_intrinsics)
                            dpg.bind_item_theme(b1, "theme_action")
                            dpg.bind_item_theme(b2, "theme_action")
                        dpg.add_input_text(tag="results", multiline=True, readonly=True, width=-1, height=200,
                                           default_value="No calibration yet.")

                # ------------------------------------------------------------------ right panel (images)
                with dpg.child_window(tag="right_panel", border=False):
                    with dpg.tab_bar():
                        with dpg.tab(label="Live"):
                            with dpg.group(horizontal=True, tag="live_group"):
                                with dpg.group():
                                    dpg.add_text("Events (trail filtered)")
                                    dpg.add_image("tex_events", tag="img_events", width=TEX_W // 2,
                                                  height=TEX_H // 2)
                                with dpg.group():
                                    dpg.add_text("Blinking chessboard detection  (green = coverage of captured views)")
                                    dpg.add_image("tex_pattern", tag="img_pattern", width=TEX_W // 2,
                                                  height=TEX_H // 2)
                        with dpg.tab(label="Views"):
                            with dpg.group(horizontal=True):
                                dpg.add_slider_int(label="View", tag="view_slider", min_value=1, max_value=1,
                                                   default_value=1, width=300,
                                                   callback=lambda: self.render_selected_view())
                                dpg.add_text("", tag="view_info")
                            dpg.add_image("tex_view", tag="img_view", width=VIEW_TEX_W // 2, height=VIEW_TEX_H // 2)
                    dpg.add_text("Log")
                    with dpg.child_window(tag="log_child", height=LOG_H, border=True):
                        dpg.add_text("", tag="log_text", wrap=0)

    def layout(self, *_):
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        dpg.configure_item("left_panel", height=vh - 16)
        dpg.configure_item("right_panel", height=vh - 16)
        right_w = max(200, vw - LEFT_W - 36)
        avail_h = max(150, vh - 90 - LOG_H - 30)
        aspect = TEX_W / TEX_H
        side_w = min((right_w - 16) / 2, (avail_h - 30) * aspect)
        stack_w = min(right_w - 16, (avail_h - 60) / 2 * aspect)
        horizontal = side_w >= stack_w
        iw = int(side_w if horizontal else stack_w)
        dpg.configure_item("live_group", horizontal=horizontal)
        for tag in ("img_events", "img_pattern"):
            dpg.configure_item(tag, width=iw, height=int(iw / aspect))
        vasp = VIEW_TEX_W / VIEW_TEX_H
        view_w = int(min(right_w - 16, (avail_h - 40) * vasp))
        dpg.configure_item("img_view", width=view_w, height=int(view_w / vasp))

    def render_selected_view(self):
        with self.lock:
            n = len(self.views)
            if n == 0:
                img, info = placeholder("No views"), ""
            else:
                i = min(max(1, dpg.get_value("view_slider")), n) - 1
                view = self.views[i]
                size = self.image_size or self.cam_size or (view.frame.shape[1], view.frame.shape[0])
                res = self.calib_result if self.calib_views_version == self.views_version else None
                pts = res.pts_2d[i] if res is not None else view.corners
                img = cc.draw_view(view.frame, pts, self.cols, self.rows, size, res, i)
                if res is None:
                    info = f"View {i + 1}/{n}" + ("  (calibration outdated)" if self.calib_result else "")
                else:
                    e = res.view_error(i)
                    info = f"View {i + 1}/{n}: " + ("rejected as outlier" if e is None else f"RMS error {e:.3f} px")
        dpg.set_value("tex_view", to_texture(img, VIEW_TEX_W, VIEW_TEX_H))
        dpg.set_value("view_info", info)

    def update_ui(self):
        """Called every frame from the main thread: pushes the state produced by the worker threads to the UI."""
        new_lines = False
        while True:
            try:
                self._log_lines.append(self._log_queue.get_nowait())
                new_lines = True
            except queue.Empty:
                break
        if new_lines:
            self._log_lines = self._log_lines[-300:]
            dpg.set_value("log_text", "\n".join(self._log_lines))
            self._scroll_log = 3
        if self._scroll_log:
            dpg.set_y_scroll("log_child", dpg.get_y_scroll_max("log_child"))
            self._scroll_log -= 1

        with self.lock:
            ev = self._ev_tex if self._ev_seq != self._shown["ev"] else None
            pat = self._pat_tex if self._pat_seq != self._shown["pat"] else None
            self._shown["ev"], self._shown["pat"] = self._ev_seq, self._pat_seq
            recording, n_views, views_version = self.recording, len(self.views), self.views_version
            calib_version, calib_running = self.calib_version, self.calib_running
            cam_running, cam_source, cam_size, rate = self.camera_running, self.cam_source, self.cam_size, \
                self.event_rate
            session = self.session_dir
        if ev is not None:
            dpg.set_value("tex_events", ev)
        if pat is not None:
            dpg.set_value("tex_pattern", pat)

        worker_alive = self.worker is not None and self.worker.is_alive()
        cam_state = (cam_running, worker_alive)
        if cam_state != self._shown["cam"]:
            self._shown["cam"] = cam_state
            if cam_running:
                dpg.set_value("cam_status", f"Open: {cam_source}  -  {cam_size[0]}x{cam_size[1]}")
            else:
                dpg.set_value("cam_status", "Opening..." if worker_alive else "Closed")
                if not worker_alive:
                    dpg.set_value("tex_events", to_texture(placeholder("Camera closed"), TEX_W, TEX_H))
            dpg.configure_item("btn_camera", label="Close camera" if worker_alive else "Open camera")
            # The camera can only be changed while it is closed
            dpg.configure_item("camera_combo", enabled=not worker_alive)
            dpg.configure_item("btn_refresh", enabled=not worker_alive)
        if cam_running:
            text = f"Event rate: {rate / 1e6:.2f} Mev/s   latency: {self.processing_lag * 1000:.0f} ms"
            if self.skipped_ratio > 0:
                text += f"\nSkipping {self.skipped_ratio * 100:.0f}% of the buffers (CPU overloaded)"
            dpg.set_value("cam_rate", text)
        else:
            dpg.set_value("cam_rate", "")

        if recording != self._shown["rec"]:
            self._shown["rec"] = recording
            dpg.configure_item("btn_record", label="Stop recording" if recording else "Start recording")
            dpg.bind_item_theme("btn_record", "theme_rec_active" if recording else "theme_rec_idle")

        locked = n_views > 0
        if locked != self._shown["locked"]:
            self._shown["locked"] = locked
            dpg.configure_item("cols", enabled=not locked)
            dpg.configure_item("rows", enabled=not locked)

        dpg.configure_item("btn_calc", label="Calculating..." if calib_running else "Calculate intrinsics")
        dpg.set_value("session_label", f"Session: {session}" if session else "")

        if views_version != self._shown["views"] or calib_version != self._shown["calib"]:
            self._shown["views"], self._shown["calib"] = views_version, calib_version
            with self.lock:
                views_serial = self.views_serial
            dpg.set_value("views_count", f"Views captured: {n_views}"
                          + (f"  (camera {views_serial})" if n_views and views_serial else ""))
            dpg.configure_item("view_slider", max_value=max(1, n_views))
            if dpg.get_value("view_slider") > n_views:
                dpg.set_value("view_slider", max(1, n_views))
            with self.lock:
                res = self.calib_result
                valid = res is not None and self.calib_views_version == self.views_version
            if res is None:
                dpg.set_value("results", "No calibration yet.")
            else:
                dpg.set_value("results", ("" if valid else "(OUTDATED: views changed since this calculation)\n\n")
                              + res.summary())
            self.render_selected_view()

    def run(self):
        dpg.create_context()
        self.build_ui()
        dpg.create_viewport(title="EVK4 intrinsic calibration (blinking chessboard)", width=1650, height=950)
        dpg.setup_dearpygui()
        dpg.set_global_font_scale(self.args.ui_scale)
        dpg.set_viewport_resize_callback(self.layout)
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        self.layout()

        if METAVISION_IMPORT_ERROR is not None:
            self.log(f"Metavision SDK Python modules not available ({METAVISION_IMPORT_ERROR}). "
                     "Only offline calibration (Load detections) is possible.")
        else:
            sources = self.refresh_cameras()
            n_cams = sum(kind == "camera" for kind, _ in sources.values())
            if self.args.input or self.args.serial in sources or n_cams == 1:
                self.start_camera()  # Unambiguous choice: open it directly
            elif self.args.serial:
                self.log(f"Camera {self.args.serial} not found.")
            elif n_cams > 1:
                self.log("Several cameras found: select one in the dropdown and press 'Open camera'.")
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
            self.stop_camera()
            if self.worker is not None:
                self.worker.join(timeout=3)
            dpg.destroy_context()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intrinsic calibration of an event camera with a blinking chessboard.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--input", default=None, help="RAW file to replay instead of the live camera.")
    p.add_argument("-s", "--serial", default=None,
                   help="Serial number of the camera to select and open at startup. Without it, the camera is "
                        "opened automatically only if exactly one is connected.")
    p.add_argument("-b", "--biases", default=None, help="Metavision .bias file to apply to the live camera.")
    p.add_argument("-o", "--output-dir", default=str(SCRIPT_DIR / "output"),
                   help="Root folder of the calibration sessions.")
    p.add_argument("--erc-mev", type=float, default=20.0,
                   help="Event rate limit applied by the sensor (Mev/s) to keep the processing real-time. 0 = off.")
    p.add_argument("--no-hw-filters", action="store_true",
                   help="Do not enable the sensor trail filter / event rate controller (not recommended).")
    p.add_argument("--cols", type=int, default=9, help="Number of inner corners along the width.")
    p.add_argument("--rows", type=int, default=6, help="Number of inner corners along the height.")
    p.add_argument("--square-size-mm", type=float, default=20.0, help="Side of a chessboard square (mm).")
    p.add_argument("--accumulation-ms", type=float, default=100.0, help="Blinking detection window (ms).")
    p.add_argument("--min-blink-pixels", type=int, default=100, help="Min. number of blinking pixels.")
    p.add_argument("--ratio-on", type=float, default=0.15, help="Max ratio of ON-only pixels / blinking pixels.")
    p.add_argument("--ratio-off", type=float, default=0.15, help="Max ratio of OFF-only pixels / blinking pixels.")
    p.add_argument("--skip-time-s", type=float, default=2.0, help="Min. time between two captured views (s).")
    p.add_argument("--min-shift-px", type=float, default=10.0,
                   help="Min. mean corner displacement w.r.t. every captured view to accept a new view (px).")
    p.add_argument("--outlier-ths", type=float, default=2.0,
                   help="Reject views with error > mean + ths*std (negative: keep all).")
    p.add_argument("--fisheye", action="store_true", help="Select the fisheye model by default.")
    p.add_argument("--save-raw", action="store_true", help="Enable 'Also save RAW events' by default.")
    p.add_argument("--no-browser", action="store_true", help="Do not launch the chessboard automatically.")
    p.add_argument("--ui-scale", type=float, default=1.0, help="Global font scale of the UI.")
    p.add_argument("--exit-after", type=float, default=0.0, help=argparse.SUPPRESS)  # For automated smoke tests
    return p.parse_args()


if __name__ == "__main__":
    App(parse_args()).run()
