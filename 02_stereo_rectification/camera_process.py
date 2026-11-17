"""
Acquisition + blinking chessboard detection for ONE camera, running in its own process.

Each camera gets a separate process (not a thread) so that the two event streams are processed in parallel instead of
competing for the Python GIL. Inside the process the same measures as in step 1 keep the stream real-time:
sensor trail filter + event rate controller, a RealTimeGuard that skips buffers if the processing falls behind, and
pixel indices computed once per buffer.

Messages sent to the UI process (out_q), as tuples whose first element is the type:
  ("started", role, serial, width, height, source_label)
  ("events", role, bgr_small)                                   ~25 Hz, dropped if the UI is busy
  ("window", role, wall_time, status, corners, frame_small, frame_full)
        status: "none" (no blinking), "blink" (blinking but no chessboard), "board" (chessboard found)
        frame_full (uint8 HxW) is only sent when the chessboard was found
  ("stats", role, event_rate, lag_s, skipped_ratio)              1 Hz
  ("log", role, text)
  ("stopped", role, finished)
Commands received (cmd_q): ("pattern", cols, rows), ("blink", dict of BlinkingConfig fields),
  ("start_raw", path), ("stop_raw",)
"""
from __future__ import annotations

import queue
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

STEP1_DIR = Path(__file__).resolve().parent.parent / "01_camera_intrincis"
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))
import calib_core as cc  # noqa: E402
from camera_utils import EVENT_PALETTE, RealTimeGuard, open_event_source  # noqa: E402

BUFFER_US = 10_000
TRAIL_FILTER_US = 1_000_000  # Software trail filter (RAW replay only)
EVENT_DISPLAY_PERIOD_US = 40_000
SMALL_W, SMALL_H = 640, 360  # Size of the display images sent to the UI


def _small(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)


def run_camera(role: str, serial: str, input_path: str, opts: dict, cmd_q, out_q, stop_evt) -> None:
    """Process entry point. Either ``serial`` (live camera) or ``input_path`` (RAW file replay) is given."""

    def send(msg, important: bool = False) -> None:
        try:
            if important:
                out_q.put(msg, timeout=2.0)
            else:
                out_q.put_nowait(msg)
        except queue.Full:
            pass

    def log(text: str) -> None:
        send(("log", role, text), important=True)

    device = mv_it = events_stream = None
    raw_logging = False
    try:
        import multiprocessing as mp
        from metavision_sdk_cv import TrailFilterAlgorithm

        parent = mp.parent_process()  # To close the camera cleanly if the UI process dies
        src = open_event_source(serial, input_path, dict(opts, buffer_us=BUFFER_US), log, stop_evt)
        device, mv_it, events_stream = src["device"], src["iterator"], src["events_stream"]
        serial, label, sw_trail = src["serial"], src["label"], src["sw_trail"]
        height, width = mv_it.get_size()
        send(("started", role, serial, width, height, label), important=True)

        trail = TrailFilterAlgorithm(width, height, TRAIL_FILTER_US) if sw_trail else None
        trail_buf = TrailFilterAlgorithm.get_empty_output_buffer() if sw_trail else None
        config = cc.BlinkingConfig(**opts.get("blink", {}))
        cols, rows = opts.get("cols", 9), opts.get("rows", 6)
        detector = cc.BlinkingChessboardDetector(width, height, cols, rows, config)
        guard = RealTimeGuard()
        ev_state = np.zeros(width * height, np.uint8)
        next_display_ts = None
        rate_count, rate_t0, last_skip_log = 0, time.time(), 0.0

        for evs in mv_it:
            if stop_evt.is_set():
                break
            # Commands from the UI
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                except queue.Empty:
                    break
                if cmd[0] == "pattern" and (cmd[1], cmd[2]) != (cols, rows):
                    cols, rows = cmd[1], cmd[2]
                    detector = cc.BlinkingChessboardDetector(width, height, cols, rows, config)
                elif cmd[0] == "blink":
                    for k, v in cmd[1].items():
                        setattr(config, k, v)
                elif cmd[0] == "start_raw":
                    if events_stream is None:
                        log("RAW saving is only available with a live camera.")
                    elif events_stream.log_raw_data(cmd[1]):
                        raw_logging = True
                        log(f"Saving RAW events to {cmd[1]}")
                    else:
                        log(f"Failed to start RAW logging to {cmd[1]}")
                elif cmd[0] == "stop_raw" and raw_logging:
                    events_stream.stop_log_raw_data()
                    raw_logging = False

            rate_count += evs.size
            now = time.time()
            if now - rate_t0 >= 1.0:
                send(("stats", role, rate_count / (now - rate_t0), guard.lag, guard.pop_skipped_ratio()))
                rate_count, rate_t0 = 0, now
                if parent is not None and not parent.is_alive():
                    break  # Orphaned: release the camera instead of streaming forever
            if evs.size == 0:
                continue

            t_last = int(evs["t"][-1])
            skip = guard.update(t_last)
            if guard.resumed:
                detector.generator.reset()
            if skip:
                if now - last_skip_log > 10:
                    last_skip_log = now
                    log(f"Processing {guard.lag:.2f} s behind the camera: skipping buffers to stay real-time "
                        "(lower --erc-mev if it persists).")
                continue

            if trail is not None:
                trail.process_events(evs, trail_buf)
                evs = trail_buf.numpy()
                if evs.size == 0:
                    continue
            idx = evs["y"].astype(np.int64) * width + evs["x"].astype(np.int64)
            pol = evs["p"] > 0

            for win in detector.process_events(evs, idx, pol):
                if win.frame is None:
                    send(("window", role, time.time(), "none", None, None, None))
                elif win.corners is None:
                    send(("window", role, time.time(), "blink", None, _small(win.frame), None))
                else:  # Important: needed to form stereo pairs
                    send(("window", role, time.time(), "board", win.corners, _small(win.frame), win.frame),
                         important=True)

            ev_state[idx] = pol.view(np.uint8) + 1
            if next_display_ts is None or t_last >= next_display_ts:
                send(("events", role, _small(EVENT_PALETTE[ev_state].reshape(height, width, 3))))
                ev_state[:] = 0
                next_display_ts = t_last + EVENT_DISPLAY_PERIOD_US
        finished = not stop_evt.is_set()
    except Exception as e:
        log(f"ERROR: {e}")
        send(("log", role, traceback.format_exc(limit=4)), important=True)
        finished = False
    finally:
        if raw_logging and events_stream is not None:
            try:
                events_stream.stop_log_raw_data()
            except Exception:
                pass
        mv_it = events_stream = device = None  # Releases the camera
    send(("stopped", role, finished), important=True)
