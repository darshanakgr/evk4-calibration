"""
One camera of the dual recorder, running in its own process (no GIL contention between the two cameras).

Producer-consumer inside the process:
  * producer = the acquisition loop: reads event buffers and, while recording, queues the events of the recording
    window [t_start, t_stop) to the EventFileWriter. It never waits for the disk.
  * consumer = the EventFileWriter thread: compresses and appends the events to the HDF5 file.
The live preview is the only optional work: it is skipped when the process falls behind (RealTimeGuard), the
recording is not.

Messages to the UI (out_q):
  ("started", role, serial, width, height, label)
  ("preview", role, t_last_us, bgr_small)
  ("status", role, dict)          ~4 Hz: rate, lag, t_last, recording state, written events, backlog, file size
  ("rec_done", role, dict)        file finalized (path, n_events, t_start, t_stop, error)
  ("log", role, text), ("stopped", role, finished)
Commands (cmd_q):
  ("rec_start", path, attrs, t_start_us or None, compression, compression_opts)   None = own latest time + 200 ms
  ("rec_stop", t_stop_us or None)                                                  None = own latest time
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

import recorder_core as rc  # Also puts the step 1-3 folders on sys.path
from camera_utils import EVENT_PALETTE, RealTimeGuard, open_event_source

BUFFER_US = 10_000
PREVIEW_PERIOD_US = 40_000
SMALL_W, SMALL_H = 640, 360
START_MARGIN_US = 200_000


def run_camera(role: str, serial: str, opts: dict, cmd_q, out_q, stop_evt) -> None:
    def send(msg, important: bool = False) -> None:
        try:
            if important:
                out_q.put(msg, timeout=5.0)
            else:
                out_q.put_nowait(msg)
        except queue.Full:
            pass

    def log(text: str) -> None:
        send(("log", role, text), important=True)

    device = mv_it = events_stream = None
    writer = None
    finishing = []
    finished = False
    state = {"last_buffer": time.time(), "loop_done": False}
    try:
        import multiprocessing as mp
        parent = mp.parent_process()
        sync_mode = opts.get("sync_mode")
        src = open_event_source(serial, "", dict(opts, buffer_us=BUFFER_US), log, stop_evt)
        device, mv_it, events_stream = src["device"], src["iterator"], src["events_stream"]
        serial = src["serial"]
        height, width = mv_it.get_size()
        state["last_buffer"] = time.time()  # Idle time counts from the moment the camera is open
        send(("started", role, serial, width, height, src["label"]), important=True)

        def watchdog():  # Unblocks the read loop (slave without master clock) so the camera is always released
            warned = False
            while not state["loop_done"]:
                if stop_evt.wait(0.5) or (parent is not None and not parent.is_alive()):
                    time.sleep(0.5)
                    if not state["loop_done"] and events_stream is not None:
                        events_stream.stop()
                    return
                idle = time.time() - state["last_buffer"]
                if sync_mode == "slave" and idle > 5 and not warned:
                    warned = True
                    log("No events from this SLAVE camera for 5 s: is the master streaming and the sync cable "
                        "connected (master SYNC OUT -> slave SYNC IN)?")
                elif idle < 1:
                    warned = False

        threading.Thread(target=watchdog, name="watchdog", daemon=True).start()

        def on_writer_done(w):
            send(("rec_done", role, {"path": str(w.path), "n_events": w.n_written, "t_start_us": w.t_start,
                                     "t_stop_us": w.t_stop, "bytes": w.file_size(),
                                     "error": repr(w.error) if w.error else ""}), important=True)

        guard = RealTimeGuard()
        ev_state = np.zeros(width * height, np.uint8)
        next_preview = None
        t_last = 0
        t_start = t_stop = None
        rate_count, rate_t0, last_status = 0, time.time(), 0.0

        for evs in mv_it:
            state["last_buffer"] = time.time()
            if stop_evt.is_set():
                break
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                except queue.Empty:
                    break
                if cmd[0] == "rec_start" and writer is None:
                    _, path, attrs, t_req, comp, comp_opts = cmd
                    t_start = int(t_req) if t_req is not None else t_last + START_MARGIN_US
                    t_stop = None
                    attrs = dict(attrs, serial=serial, width=width, height=height)
                    writer = rc.EventFileWriter(path, width, height, t_start, attrs, compression=comp,
                                                compression_opts=comp_opts, on_done=on_writer_done).start()
                    log(f"Recording to {Path(path).name} from t = {t_start} us")
                elif cmd[0] == "rec_stop" and writer is not None and t_stop is None:
                    t_stop = int(cmd[1]) if cmd[1] is not None else max(t_last, t_start)

            n = evs.size
            rate_count += n
            if n:
                t_last = int(evs["t"][-1])  # Before the status: the UI derives the synchronized stop time from it
            now = time.time()
            if now - last_status >= 0.25:
                dt = now - rate_t0
                status = {"rate": rate_count / dt if dt > 0 else 0.0, "lag": guard.lag, "t_last": t_last,
                          "recording": writer is not None, "t_start": t_start, "t_stop": t_stop}
                if writer is not None:
                    status.update(written=writer.n_written, backlog=writer.backlog_events,
                                  backlog_mb=writer.backlog_bytes / 2 ** 20, file_mb=writer.file_size() / 2 ** 20)
                send(("status", role, status))
                if dt >= 1.0:
                    rate_count, rate_t0 = 0, now
                last_status = now
            if n == 0:
                continue
            t = evs["t"]
            t_last = int(t[-1])

            # --- Producer: queue the events of the recording window (never skipped)
            if writer is not None:
                lo = int(np.searchsorted(t, t_start, "left")) if t[0] < t_start else 0
                hi = int(np.searchsorted(t, t_stop, "left")) if t_stop is not None else n
                if hi > lo:
                    writer.put(evs[lo:hi])
                if t_stop is not None and t_last >= t_stop:
                    writer.finish(t_stop)  # The writer thread flushes the queue and closes the file
                    log("Recording stopped, finishing the file...")
                    finishing.append(writer)  # Joined before the process exits (never cut a file short)
                    writer = None

            # --- Preview (optional work, skipped when behind)
            if guard.update(t_last):
                continue
            ev_state[evs["y"].astype(np.int64) * width + evs["x"].astype(np.int64)] = (evs["p"] > 0).astype(np.uint8) + 1
            if next_preview is None or t_last >= next_preview:
                img = cv2.resize(EVENT_PALETTE[ev_state].reshape(height, width, 3), (SMALL_W, SMALL_H),
                                 interpolation=cv2.INTER_AREA)
                send(("preview", role, t_last, img))
                ev_state[:] = 0
                next_preview = t_last + PREVIEW_PERIOD_US
        finished = not stop_evt.is_set()
    except Exception as e:
        log(f"ERROR: {e}")
        send(("log", role, traceback.format_exc(limit=4)), important=True)
    finally:
        state["loop_done"] = True
        if writer is not None:  # Camera closed while recording: keep what was recorded
            writer.finish(None)
            finishing.append(writer)
        for w in finishing:
            w.join()
        mv_it = events_stream = device = None
    send(("stopped", role, finished), important=True)
