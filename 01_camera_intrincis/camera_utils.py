"""
Live event-camera helpers shared by the calibration tools (no UI, no Metavision import at module level).

* apply_bias_file / configure_hw_filters: camera configuration. Filtering on the sensor (trail filter + event rate
  controller) instead of in Python is what keeps the processing real-time.
* RealTimeGuard: detects when the processing falls behind the camera and tells the caller to skip buffers until it has
  caught up, so that a backlog (an ever-growing display delay) can never build up.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

HW_TRAIL_FILTER_US = 100_000  # Sensor trail filter threshold (clamped to the supported range)
MAX_LAG_S = 0.25  # If the processing falls this much behind the camera, buffers are skipped...
RESUME_LAG_S = 0.05  # ...until the lag is back below this value
EVENT_PALETTE = np.array([[30, 30, 30], [200, 126, 64], [255, 255, 255]], np.uint8)  # BGR: none / OFF / ON


def apply_bias_file(device, path: str, log) -> None:
    """Loads a Metavision .bias file ('value % bias_name' per line)."""
    biases = device.get_i_ll_biases()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "%" not in line:
            continue
        value, _, name = line.partition("%")
        biases.set(name.strip(), int(value.strip()))
        log(f"  bias {name.strip()} = {int(value.strip())}")


def configure_hw_filters(device, erc_rate_mev: float, log, trail: bool = True) -> bool:
    """Enables the sensor trail filter (if ``trail``) and the Event Rate Controller (if erc_rate_mev > 0).

    A blinking chessboard filling the field of view easily produces > 50 Mev/s, much more than the Python processing can
    handle. Returns True if the sensor trail filter is active (the software trail filter is then not needed).
    """
    trail_on = False
    try:
        tf = device.get_i_event_trail_filter_module() if trail else None
        if tf is not None:
            thr = int(min(max(HW_TRAIL_FILTER_US, tf.get_min_supported_threshold()), tf.get_max_supported_threshold()))
            tf.set_type(tf.Type.TRAIL)
            tf.set_threshold(thr)
            tf.enable(True)
            trail_on = bool(tf.is_enabled())
            if trail_on:
                log(f"Sensor trail filter enabled ({thr / 1000:.0f} ms).")
    except Exception as e:
        log(f"Sensor trail filter not available ({e}); using the software trail filter.")
    if erc_rate_mev and erc_rate_mev > 0:
        try:
            erc = device.get_i_erc_module()
            if erc is not None:
                erc.set_cd_event_rate(int(erc_rate_mev * 1e6))
                erc.enable(True)
                log(f"Sensor event rate controller enabled ({erc.get_cd_event_rate() / 1e6:.0f} Mev/s max).")
        except Exception as e:
            log(f"Sensor event rate controller not available ({e}).")
    return trail_on


def set_sync_mode(device, mode: str, log) -> bool:
    """Timestamp synchronization through the SYNC IN/OUT cable (same as the SDK metavision_sync sample).

    mode: "master" (generates the clock on SYNC OUT), "slave" (follows the clock on SYNC IN: it produces NO event until
    the master streams, so start the slave first) or "standalone". The mode does not persist when the camera is closed.
    """
    sync = device.get_i_camera_synchronization()
    if sync is None:
        log("This camera has no synchronization facility.")
        return False
    {"master": sync.set_mode_master, "slave": sync.set_mode_slave,
     "standalone": sync.set_mode_standalone}[mode]()
    if mode != "standalone":
        log(f"Synchronization mode: {mode.upper()}" + (" (waits for the master clock)" if mode == "slave" else ""))
    return True


def open_event_source(serial: str, input_path: str, opts: dict, log, stop_evt=None):
    """Opens a live camera (by serial) or a RAW file (replayed in real time) with the real-time settings applied.

    Returns a dict: device, iterator, events_stream (None for files), serial, label, sw_trail (True if the software
    trail filter is still needed). opts: bias_file, hw_filters (default True), trail_filter (default True, only used
    when hw_filters), erc_mev (default 20, 0 = off), buffer_us, sync_mode (None: untouched, "standalone", "master" or "slave"; see set_sync_mode).
    Opening a camera is retried: it can fail transiently while another process probes the USB bus.
    """
    from metavision_core.event_io import EventsIterator
    from metavision_core.event_io.live_replay import LiveReplayEventsIterator
    from metavision_core.event_io.raw_reader import initiate_device

    buffer_us = int(opts.get("buffer_us", 10_000))
    if input_path:
        device = initiate_device(input_path)
        hw_id = device.get_i_hw_identification()
        return {"device": device, "events_stream": None, "sw_trail": True,
                "serial": hw_id.get_serial() if hw_id is not None else "",
                "iterator": LiveReplayEventsIterator(EventsIterator.from_device(device, delta_t=buffer_us)),
                "label": f"File {Path(input_path).name}"}
    log(f"Opening camera {serial}...")
    device = None
    for attempt in range(3):
        try:
            device = initiate_device(serial)
            break
        except Exception as e:
            if attempt == 2 or (stop_evt is not None and stop_evt.is_set()):
                raise
            log(f"Open failed ({e}), retrying...")
            time.sleep(1.5)
    serial = device.get_i_hw_identification().get_serial()
    if opts.get("bias_file"):
        apply_bias_file(device, opts["bias_file"], log)
    sw_trail = True
    if opts.get("hw_filters", True):
        sw_trail = not configure_hw_filters(device, opts.get("erc_mev", 20.0), log, trail=opts.get("trail_filter", True))
    if opts.get("sync_mode"):
        set_sync_mode(device, opts["sync_mode"], log)  # Must be done before the stream starts (iterator creation)
    return {"device": device, "events_stream": device.get_i_events_stream(), "sw_trail": sw_trail, "serial": serial,
            "iterator": EventsIterator.from_device(device, delta_t=buffer_us), "label": f"Camera {serial}"}


class RealTimeGuard:
    """Measures the delay between the camera and the processing, and decides when buffers must be skipped.

    The lag is (wall clock - event time) relative to its minimum over the stream, so it does not depend on the
    camera/PC clock offset. Call ``update(t_last_us)`` for every non-empty buffer; it returns True if the buffer must
    be skipped. ``resumed`` is True on the first processed buffer after a skipping episode (accumulators that missed
    events should be reset).
    """

    def __init__(self, max_lag_s: float = MAX_LAG_S, resume_lag_s: float = RESUME_LAG_S):
        self.max_lag_s = max_lag_s
        self.resume_lag_s = resume_lag_s
        self.lag = 0.0
        self.skipping = False
        self.resumed = False
        self.n_buffers = 0
        self.n_skipped = 0
        self.n_episodes = 0
        self._offset_min = None

    def update(self, t_last_us: int) -> bool:
        offset = time.perf_counter() - t_last_us * 1e-6
        self._offset_min = offset if self._offset_min is None else min(self._offset_min, offset)
        self.lag = offset - self._offset_min
        self.n_buffers += 1
        self.resumed = False
        if self.skipping and self.lag < self.resume_lag_s:
            self.skipping = False
            self.resumed = True
        elif not self.skipping and self.lag > self.max_lag_s:
            self.skipping = True
            self.n_episodes += 1
        if self.skipping:
            self.n_skipped += 1
        return self.skipping

    def pop_skipped_ratio(self) -> float:
        """Fraction of skipped buffers since the last call."""
        ratio = self.n_skipped / max(1, self.n_buffers)
        self.n_buffers = self.n_skipped = 0
        return ratio
