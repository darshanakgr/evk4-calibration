"""
Acquisition + on-the-fly rectification for ONE camera of the depth tool, running in its own process.

Same real-time measures as steps 1-2 (sensor trail filter + event rate controller, RealTimeGuard, one process per
camera). Every event is moved to its rectified pixel (at the processing scale) with an index LUT and written into a
time surface. Two ways of aligning the left and right frames:
  * host clock (default, no cable): frames are published on host-clock ticks (int(time.time() * fps)); both processes
    publish at the same moments within a few ms.
  * hardware sync (opts["aligned"], cameras in master/slave mode with the sync cable): both cameras share the same
    event time base, so frames are cut at exact event-time boundaries (multiples of 1/fps) and the frame key is that
    boundary: identical keys in both cameras = exactly the same instant.

Messages to the UI (out_q):
  ("started", role, serial, width, height, label)
  ("frame", role, host_time, key_us, image_uint8)   rectified time surface at the processing scale; key_us = event
                                                    time of the frame (boundary time when aligned)
  ("stats", role, event_rate, lag_s, skipped_ratio)
  ("log", role, text)
  ("stopped", role, finished)
Commands (cmd_q): ("lut", index_lut_or_None, (w_s, h_s), scale), ("tau", tau_ms), ("fps", fps),
  ("start_raw", path), ("stop_raw",)
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

STEP1_DIR = Path(__file__).resolve().parent.parent / "01_camera_intrincis"
if str(STEP1_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1_DIR))
from camera_utils import RealTimeGuard, open_event_source  # noqa: E402

import depth_core as dc  # noqa: E402

BUFFER_US = 5_000  # Short buffers: finer alignment of the published frames
TRAIL_FILTER_US = 1_000_000  # Software trail filter (RAW replay only)


def run_camera(role: str, serial: str, input_path: str, opts: dict, cmd_q, out_q, stop_evt) -> None:
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
    finished = False
    try:
        import multiprocessing as mp
        from metavision_sdk_cv import TrailFilterAlgorithm

        parent = mp.parent_process()
        sync_mode = opts.get("sync_mode") if not input_path else None
        src = open_event_source(serial, input_path, dict(opts, buffer_us=BUFFER_US, sync_mode=sync_mode), log,
                                stop_evt)
        device, mv_it, events_stream = src["device"], src["iterator"], src["events_stream"]
        height, width = mv_it.get_size()
        send(("started", role, src["serial"], width, height, src["label"]), important=True)

        # Watchdog: the read loop blocks when no data arrives (a slave camera without its master clock). Stopping the
        # stream from this thread unblocks it, so the camera is always released cleanly (never killed during a USB
        # transfer). It also warns when a slave receives nothing.
        state = {"last_buffer": time.time(), "loop_done": False}

        def watchdog():
            warned = False
            while not state["loop_done"]:
                if stop_evt.wait(0.5) or (parent is not None and not parent.is_alive()):
                    time.sleep(0.5)  # Grace period: the loop normally exits by itself
                    if not state["loop_done"] and events_stream is not None:
                        events_stream.stop()
                    return
                idle = time.time() - state["last_buffer"]
                if sync_mode == "slave" and idle > 5 and not warned:
                    warned = True
                    log("No events from this SLAVE camera for 5 s: is the master camera streaming and the sync cable "
                        "connected (master SYNC OUT -> slave SYNC IN)?")
                elif idle < 1:
                    warned = False

        threading.Thread(target=watchdog, name="watchdog", daemon=True).start()
        aligned = bool(opts.get("aligned")) and not input_path

        trail = TrailFilterAlgorithm(width, height, TRAIL_FILTER_US) if src["sw_trail"] else None
        trail_buf = TrailFilterAlgorithm.get_empty_output_buffer() if src["sw_trail"] else None

        def setup(lut, scale):
            if lut is None or lut.size != width * height:  # No (matching) calibration: downscaled raw view
                lut = dc.make_index_lut(None, role, (width, height), scale)
            ws, hs = dc.scaled_size((width, height), scale)
            return lut, dc.TimeSurface(ws, hs)

        scale = float(opts.get("scale", 0.5))
        lut, ts = setup(opts.get("lut"), scale)
        tau_us = float(opts.get("tau_ms", 30.0)) * 1000
        fps = float(opts.get("fps", 20.0))
        guard = RealTimeGuard()
        last_tick = None
        framer = dc.EventTimeFramer(fps)  # Hardware-synchronized mode
        rate_count, rate_t0, last_skip_log = 0, time.time(), 0.0

        for evs in mv_it:
            state["last_buffer"] = time.time()
            if stop_evt.is_set():
                break
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                except queue.Empty:
                    break
                if cmd[0] == "lut":
                    scale = float(cmd[3])
                    lut, ts = setup(cmd[1], scale)
                elif cmd[0] == "tau":
                    tau_us = float(cmd[1]) * 1000
                elif cmd[0] == "fps":
                    fps = max(1.0, float(cmd[1]))
                    framer.set_fps(fps)
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
                    break  # Orphaned: release the camera
            if evs.size == 0:
                continue
            t_last = int(evs["t"][-1])
            if guard.update(t_last):
                if now - last_skip_log > 10:
                    last_skip_log = now
                    log(f"Processing {guard.lag:.2f} s behind the camera: skipping buffers to stay real-time.")
                continue

            if trail is not None:
                trail.process_events(evs, trail_buf)
                evs = trail_buf.numpy()
            rect_idx = lut[evs["y"].astype(np.int64) * width + evs["x"].astype(np.int64)] if evs.size else None

            if aligned:  # Shared event time base: frames cut exactly at the same event-time boundaries
                t = evs["t"] if evs.size else np.zeros(0, np.int64)
                for key, img in framer.process(ts, rect_idx, t, t_last, tau_us):
                    send(("frame", role, time.time(), key, img))
            else:
                if rect_idx is not None:
                    ts.add(rect_idx, evs["t"])
                tick = int(now * fps)  # Host-clock tick shared by both camera processes
                if tick != last_tick:
                    last_tick = tick
                    send(("frame", role, now, t_last, ts.render(t_last, tau_us)))
        finished = not stop_evt.is_set()
    except Exception as e:
        log(f"ERROR: {e}")
        send(("log", role, traceback.format_exc(limit=4)), important=True)
    finally:
        if "state" in locals():
            state["loop_done"] = True
        if raw_logging and events_stream is not None:
            try:
                events_stream.stop_log_raw_data()
            except Exception:
                pass
        mv_it = events_stream = device = None
    send(("stopped", role, finished), important=True)
