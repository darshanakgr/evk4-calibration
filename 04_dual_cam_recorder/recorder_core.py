"""
UI-independent core of the dual-camera recorder (step 4).

Recording (producer-consumer)
    The camera loop (producer) puts event batches in the queue of an EventFileWriter; the writer thread (consumer)
    appends them to an HDF5 file, so slow disk/compression never blocks the acquisition (no event is dropped, the queue
    absorbs the delay). One file per camera:

        /events/x  uint16     /events/y  uint16     /events/p  uint8 (0 = OFF, 1 = ON)     /events/t  int64 (us)
        /index/ms_to_idx  int64: ms_to_idx[k] = number of events with t < t_start_us + k * 1000  (seek + rate timeline)
        attributes: serial, role, width, height, t_start_us, t_stop_us, n_events, sync_mode, compression, ...

    With hardware sync the two cameras share the same time base, and both files cover exactly [t_start_us, t_stop_us).

Replay
    EventFileReader reads events in a time range (through the ms index) and the event-rate timeline.
    preprocess() renders event frames into a temporary HDF5 file read by FrameFile for smooth playback. Frames are
    consecutive windows of the accumulation time (N = duration / accumulation), with the selected polarity and optional
    rectification (step 2 calibration). The frame rate only sets the playback speed: N frames take N / fps seconds.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _d in (ROOT / "01_camera_intrincis", ROOT / "02_stereo_rectification", ROOT / "03_depth_estimation"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
from camera_utils import EVENT_PALETTE  # noqa: E402  (step 1)

FORMAT = "evk4-dual-recorder/1"
FIELDS = (("x", np.uint16), ("y", np.uint16), ("p", np.uint8), ("t", np.int64))
BYTES_PER_EVENT = 13  # In memory, before compression
COMPRESSIONS = {"lzf (fast)": ("lzf", None), "gzip 1": ("gzip", 1), "gzip 4 (smaller, slower)": ("gzip", 4),
                "none": (None, None)}
POLARITIES = ("BOTH", "ON", "OFF")
_FLUSH = object()  # Writer-internal marker


# ----------------------------------------------------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------------------------------------------------
class EventFileWriter:
    """Consumer of the recorder: a thread appends the event batches put in its queue to an HDF5 file."""

    def __init__(self, path, width: int, height: int, t_start_us: int, attrs: Optional[dict] = None,
                 compression: Optional[str] = "lzf", compression_opts=None, chunk: int = 262_144,
                 block: int = 500_000, on_done: Optional[Callable[["EventFileWriter"], None]] = None):
        self.path = Path(path)
        self.width, self.height = width, height
        self.t_start = int(t_start_us)
        self.t_stop: Optional[int] = None
        self.attrs = dict(attrs or {})
        self.compression, self.compression_opts = compression, compression_opts
        self.chunk, self.block = chunk, block
        self.on_done = on_done
        self.queue: "queue.Queue" = queue.Queue()  # Unbounded: absorbs any temporary slowness of the disk
        self.n_written = 0
        self.t_last: Optional[int] = None
        self.error: Optional[BaseException] = None
        self.done = False
        self._backlog = 0
        self._lock = threading.Lock()
        self._ms_idx = [0]  # ms_to_idx[k] = number of events with t < t_start + k * 1000
        self._thread = threading.Thread(target=self._run, name=f"writer_{self.path.stem}", daemon=True)

    # Producer side -----------------------------------------------------------------------------------------------
    def start(self) -> "EventFileWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def put(self, evs: np.ndarray) -> None:
        """Queues a structured array of events (fields x, y, p, t), sorted by t and >= t_start."""
        if evs.size:
            with self._lock:
                self._backlog += evs.size
            self.queue.put(evs.copy())  # The acquisition buffer may be reused: keep our own copy

    def finish(self, t_stop_us: Optional[int] = None) -> None:
        """No more events: the writer flushes the queue, writes the index and closes the file."""
        self.t_stop = t_stop_us
        self.queue.put(None)

    def join(self, timeout: Optional[float] = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def backlog_events(self) -> int:
        return self._backlog

    @property
    def backlog_bytes(self) -> int:
        return self._backlog * BYTES_PER_EVENT

    def file_size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    # Consumer side -----------------------------------------------------------------------------------------------
    def _run(self) -> None:
        f = None
        try:
            f = h5py.File(self.path, "w")
            g = f.create_group("events")
            self._ds = {name: g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(self.chunk,),
                                               compression=self.compression, compression_opts=self.compression_opts,
                                               shuffle=self.compression is not None)
                        for name, dt in FIELDS}
            self._write_attrs(f, final=False)  # A partial file (crash) is still self-describing
            pending, n_pending = [], 0
            while True:
                try:
                    item = self.queue.get(timeout=0.5)
                except queue.Empty:
                    item = _FLUSH  # Write what we have when the stream is slow
                if item is None or item is _FLUSH:  # Identity tests: items are numpy arrays
                    if pending:
                        self._flush(pending)
                        pending, n_pending = [], 0
                    if item is None:
                        break
                    continue
                pending.append(item)
                n_pending += item.size
                if n_pending >= self.block:
                    self._flush(pending)
                    pending, n_pending = [], 0
            # Finalize: exact end of the recording, index, attributes
            end = self.t_stop if self.t_stop is not None else (self.t_last + 1 if self.t_last is not None else
                                                               self.t_start)
            if self.t_stop is not None and self.t_last is not None and self.t_last >= self.t_stop:
                self._truncate(self.t_stop)  # Events already queued past the stop time (stop sent late)
            self._ms_idx = self._ms_idx[:int(-(-(end - self.t_start) // 1000)) + 1]
            self._extend_index(None, upto_us=end)
            f.create_dataset("index/ms_to_idx", data=np.asarray(self._ms_idx, np.int64), compression="gzip")
            self._write_attrs(f, final=True, t_stop=end)
        except BaseException as e:  # Reported to the UI by the camera process
            self.error = e
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            self.done = True
            if self.on_done is not None:
                self.on_done(self)

    def _flush(self, pending) -> None:
        evs = np.concatenate(pending) if len(pending) > 1 else pending[0]
        n0, n = self.n_written, evs.size
        for name, dt in FIELDS:
            ds = self._ds[name]
            ds.resize((n0 + n,))
            ds[n0:] = evs[name].astype(dt, copy=False) if name != "p" else (evs["p"] > 0).astype(np.uint8)
        self._extend_index(evs["t"], base=n0)
        self.n_written = n0 + n
        self.t_last = int(evs["t"][-1])
        with self._lock:
            self._backlog -= n

    def _truncate(self, t_stop: int) -> None:
        """Removes the events with t >= t_stop (the file then covers exactly [t_start, t_stop))."""
        k = int((t_stop - self.t_start) // 1000)
        i_k = self._ms_idx[min(k, len(self._ms_idx) - 1)]  # No event before this index is >= t_stop
        tail = self._ds["t"][i_k:self.n_written]
        cut = i_k + int(np.searchsorted(tail, t_stop, "left"))
        for ds in self._ds.values():
            ds.resize((cut,))
        self.n_written = cut
        self._ms_idx = [min(v, cut) for v in self._ms_idx]
        self.t_last = int(self._ds["t"][cut - 1]) if cut else None

    def _extend_index(self, t: Optional[np.ndarray], base: int = 0, upto_us: Optional[int] = None) -> None:
        """Adds the ms boundaries covered by the batch t (or up to upto_us at the end of the recording)."""
        if t is not None:
            k_last = int((t[-1] - self.t_start) // 1000)  # Boundaries up to this ms are complete only before t[-1]
            ks = np.arange(len(self._ms_idx), k_last + 1)
            if ks.size:
                self._ms_idx.extend((base + np.searchsorted(t, self.t_start + ks * 1000, side="left")).tolist())
        else:
            k_end = int(-(-(upto_us - self.t_start) // 1000))  # ceil
            while len(self._ms_idx) <= k_end:
                self._ms_idx.append(self.n_written)

    def _write_attrs(self, f, final: bool, t_stop: Optional[int] = None) -> None:
        f.attrs["format"] = FORMAT
        for k, v in self.attrs.items():
            f.attrs[k] = v if isinstance(v, (int, float, str, bool, np.integer, np.floating)) else json.dumps(v)
        f.attrs["width"], f.attrs["height"] = self.width, self.height
        f.attrs["t_start_us"] = self.t_start
        f.attrs["compression"] = str(self.compression)
        f.attrs["complete"] = bool(final)
        if final:
            f.attrs["t_stop_us"] = int(t_stop)
            f.attrs["n_events"] = int(self.n_written)


# ----------------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------------
class EventFileReader:
    def __init__(self, path, cache_mb: int = 256):
        self.path = Path(path)
        # Large chunk cache: consecutive frames read overlapping ranges of the same compressed chunks
        self.f = h5py.File(self.path, "r", rdcc_nbytes=cache_mb * 2 ** 20, rdcc_nslots=100_003)
        self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in self.f.attrs.items()}
        g = self.f["events"]
        self.ds = {name: g[name] for name, _ in FIELDS}
        self.n = int(self.ds["t"].shape[0])
        self.width, self.height = int(self.attrs["width"]), int(self.attrs["height"])
        self.t_start = int(self.attrs["t_start_us"])
        if "index/ms_to_idx" in self.f:
            self.ms_idx = self.f["index/ms_to_idx"][:]
        else:  # Interrupted recording: rebuild the index from the timestamps
            self.ms_idx = self._rebuild_index()
        self.t_stop = int(self.attrs.get("t_stop_us", self.t_start + (len(self.ms_idx) - 1) * 1000))

    @property
    def serial(self) -> str:
        return str(self.attrs.get("serial", ""))

    @property
    def role(self) -> str:
        return str(self.attrs.get("role", ""))

    @property
    def duration_s(self) -> float:
        return (self.t_stop - self.t_start) * 1e-6

    def _rebuild_index(self, block: int = 4_000_000) -> np.ndarray:
        idx = [0]
        for i0 in range(0, self.n, block):
            t = self.ds["t"][i0:i0 + block]
            k_last = int((t[-1] - self.t_start) // 1000)
            ks = np.arange(len(idx), k_last + 1)
            idx.extend((i0 + np.searchsorted(t, self.t_start + ks * 1000)).tolist())
        idx.append(self.n)
        return np.asarray(idx, np.int64)

    def index_range(self, t0: int, t1: int):
        """Event indices [i0, i1) with t0 <= t < t1."""
        m = len(self.ms_idx) - 1
        k0 = int(np.clip((t0 - self.t_start) // 1000, 0, m))
        k1 = int(np.clip(-(-(t1 - self.t_start) // 1000), 0, m))
        i0, i1 = int(self.ms_idx[k0]), int(self.ms_idx[k1]) if k1 < m else self.n
        if i1 <= i0:
            return i0, i0
        t = self.ds["t"][i0:i1]  # Refine inside the coarse (1 ms) range
        return i0 + int(np.searchsorted(t, t0, "left")), i0 + int(np.searchsorted(t, t1, "left"))

    def events_between(self, t0: int, t1: int) -> Dict[str, np.ndarray]:
        i0, i1 = self.index_range(t0, t1)
        return {name: self.ds[name][i0:i1] for name, _ in FIELDS}

    def rates(self, bin_us: int):
        """(bin start times in s relative to t_start, event rate in ev/s)."""
        bin_ms = max(1, int(round(bin_us / 1000)))
        counts = np.diff(self.ms_idx)
        n_bins = max(1, -(-len(counts) // bin_ms))
        padded = np.zeros(n_bins * bin_ms, np.int64)
        padded[:len(counts)] = counts
        per_bin = padded.reshape(n_bins, bin_ms).sum(axis=1)
        return np.arange(n_bins) * bin_ms * 1e-3, per_bin / (bin_ms * 1e-3)

    def close(self) -> None:
        self.f.close()


def find_recording_files(directory) -> Dict[str, Path]:
    """{"left": path, "right": path} of a recording folder (roles from session.json or from the file attributes)."""
    directory = Path(directory)
    files = sorted(directory.glob("*.h5"))
    roles = {}
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                role = f.attrs.get("role", "")
                role = role.decode() if isinstance(role, bytes) else str(role)
        except OSError:
            continue
        if role in ("left", "right") and role not in roles:
            roles[role] = p
    if len(roles) < 2:  # No role information: alphabetical order
        rest = [p for p in files if p not in roles.values()]
        for role in ("left", "right"):
            if role not in roles and rest:
                roles[role] = rest.pop(0)
    if len(roles) < 2:
        raise FileNotFoundError(f"Expected two camera files (*.h5) in {directory}")
    return roles


# ----------------------------------------------------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------------------------------------------------
# Temporal bins: Matplotlib's "tab10" qualitative palette (RGB), bin k always has the same color in both cameras
BIN_PALETTE_RGB = np.array([(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                            (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)], np.uint8)
BIN_LUT = np.vstack([EVENT_PALETTE[:1], BIN_PALETTE_RGB[:, ::-1]])  # BGR, index 0 = no event (background)
MIN_BINS, MAX_BINS = 2, 10


def render_state(ev: Dict[str, np.ndarray], width: int, height: int, polarity: str = "BOTH",
                 lut: Optional[np.ndarray] = None, time_bins: Optional[tuple] = None) -> np.ndarray:
    """uint8 (height, width) frame, 0 = no event (the latest event of each pixel wins).

    * Default: 1 = OFF, 2 = ON.
    * time_bins = (t_window_start, accumulation_us, n_bins): value = 1 + index of the temporal bin of the event
      (bins of accumulation / n_bins us). Polarity is then not encoded ("ON"/"OFF" still select the events shown).
    lut: optional rectification index (raw flat pixel -> rectified flat pixel, -1 = outside).
    """
    x, y, p = ev["x"], ev["y"], ev["p"]
    if polarity == "ON":
        keep = p > 0
    elif polarity == "OFF":
        keep = p == 0
    else:
        keep = None
    if time_bins is not None:
        t0, acc, n_bins = time_bins
        t = ev["t"] if keep is None else ev["t"][keep]
        value = (np.clip((t - t0) * n_bins // max(1, acc), 0, n_bins - 1) + 1).astype(np.uint8)
    else:
        value = None
    if keep is not None:
        x, y, p = x[keep], y[keep], p[keep]
    if value is None:
        value = p.astype(np.uint8) + 1
    idx = y.astype(np.int64) * width + x.astype(np.int64)
    if lut is not None:
        idx = lut[idx]
        ok = idx >= 0
        idx, value = idx[ok], value[ok]
    state = np.zeros(width * height, np.uint8)
    state[idx] = value
    return state.reshape(height, width)


def colorize(state: np.ndarray, time_bins: bool = False) -> np.ndarray:
    """BGR image of a frame: polarity colors, or one color per temporal bin."""
    return BIN_LUT[state] if time_bins else EVENT_PALETTE[state]


def bin_legend(accumulation_us: int, n_bins: int, width: int = 960, height: int = 26) -> np.ndarray:
    """BGR strip: the color of each temporal bin and its time range inside the frame window."""
    import cv2
    img = np.full((height, width, 3), 35, np.uint8)
    w = width // n_bins
    unit, scale = ("ms", 1e-3) if accumulation_us >= 10_000 else ("us", 1.0)
    for b in range(n_bins):
        x0 = b * w
        cv2.rectangle(img, (x0 + 2, 3), (x0 + 18, height - 4), tuple(int(c) for c in BIN_LUT[b + 1]), -1)
        a, z = b * accumulation_us / n_bins * scale, (b + 1) * accumulation_us / n_bins * scale
        cv2.putText(img, f"{a:g}-{z:g} {unit}", (x0 + 22, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return img


@dataclass
class PreprocessParams:
    """Frame k aggregates the events of [t_begin + k * acc, t_begin + (k + 1) * acc): consecutive, non-overlapping
    windows, so every event appears in exactly one frame. The playback frame rate is NOT a pre-processing parameter:
    it only sets how many of these frames are shown per second (replay speed = accumulation x fps)."""
    accumulation_us: int = 20_000
    polarity: str = "BOTH"
    rectify: bool = False
    stereo_json: str = ""
    time_bins: bool = False  # Color each event by its temporal bin inside the frame (polarity then ignored)
    n_bins: int = 4  # 2..10 bins per frame
    start_us: int = 0  # Time range to process, relative to the start of the recording (us)
    end_us: int = -1  # <= 0: until the end of the recording


def trim_range(rec_begin: int, rec_end: int, start_us: int, end_us: int):
    """Absolute [t_begin, t_end) to process from a time range relative to the recording start (clipped)."""
    t_begin = rec_begin + int(np.clip(start_us, 0, rec_end - rec_begin))
    t_end = rec_end if end_us <= 0 else rec_begin + int(np.clip(end_us, 0, rec_end - rec_begin))
    return t_begin, t_end


MAX_FRAMES = 200_000  # Per camera: beyond that the temporary file and the processing time become unreasonable


def frame_plan(duration_us: float, accumulation_us: float, fps: float):
    """(number of frames, replay duration in s, replay speed vs real time).

    Example: 5 s recorded, 1000 us accumulation, 10 fps -> 5000 frames, 500 s of replay, speed 0.01x.
    """
    acc = max(1, int(accumulation_us))
    n = max(1, int(-(-int(duration_us) // acc)))  # ceil: the last frame may be partial
    return n, n / max(1e-9, fps), acc * 1e-6 * fps


def iter_windows(rd: "EventFileReader", t_begin: int, acc: int, n: int, block: int = 2_000_000):
    """Events of the consecutive windows [t_begin + k acc, t_begin + (k + 1) acc), k = 0..n-1, read sequentially in
    large blocks (much faster than one small read per frame when frames are short)."""
    i_next = rd.index_range(t_begin, t_begin + 1)[0]
    buf = {name: np.zeros(0, dt) for name, dt in FIELDS}
    for k in range(n):
        t1 = t_begin + (k + 1) * acc
        while (buf["t"].size == 0 or buf["t"][-1] < t1) and i_next < rd.n:
            j = min(rd.n, i_next + block)
            for name, _ in FIELDS:
                buf[name] = np.concatenate([buf[name], rd.ds[name][i_next:j]])
            i_next = j
        cut = int(np.searchsorted(buf["t"], t1, "left"))
        yield {name: a[:cut] for name, a in buf.items()}
        buf = {name: a[cut:] for name, a in buf.items()}


def rectification_luts(readers: Dict[str, EventFileReader], stereo_json: str, log=print) -> Dict[str, np.ndarray]:
    """Per-camera rectification index LUTs, assigned by serial number (works even if the roles were swapped)."""
    import depth_core as dc  # Step 3
    import stereo_core as sc  # Step 2
    res = sc.load_stereo(stereo_json)
    luts = {}
    for role, rd in readers.items():
        if rd.serial and rd.serial == res.serial_left:
            side = "left"
        elif rd.serial and rd.serial == res.serial_right:
            side = "right"
        else:
            side = role
            log(f"Camera {rd.serial or role} is not in the calibration ({res.serial_left}/{res.serial_right}): "
                f"using the '{side}' parameters.")
        if tuple(res.image_size) != (rd.width, rd.height):
            raise ValueError(f"Calibration is for {res.image_size}, recording is {rd.width}x{rd.height}.")
        luts[role] = dc.index_lut_from_coords(sc.compute_event_lut(res, side), (rd.width, rd.height), 1.0)
    return luts


def preprocess(rec_files: Dict[str, str], out_path, params: PreprocessParams,
               progress: Optional[Callable[[int, int], None]] = None, cancel: Optional[threading.Event] = None,
               log=print) -> dict:
    """Renders the event frames of both cameras into an HDF5 file:

        /<role>   uint8 (N, H, W): 0 = none, 1 = OFF, 2 = ON  (time bins: 1 + bin index, see render_state)
        /times    int64 (N,): event time (us) of the end of each frame; frame k shows [times[k] - acc, times[k]),
                  with times[k] = t_begin + (k + 1) * acc (consecutive windows, N = ceil(duration / acc))
        /counts_<role> int64 (N,): number of events shown in each frame
    """
    t0_wall = time.time()
    readers = {role: EventFileReader(p) for role, p in rec_files.items()}
    try:
        sizes = {(rd.width, rd.height) for rd in readers.values()}
        if len(sizes) != 1:
            raise ValueError(f"Different camera resolutions: {sizes}")
        width, height = sizes.pop()
        luts = rectification_luts(readers, params.stereo_json, log) if params.rectify else {r: None for r in readers}
        rec_begin = min(rd.t_start for rd in readers.values())
        rec_end = max(rd.t_stop for rd in readers.values())
        t_begin, t_end = trim_range(rec_begin, rec_end, params.start_us, params.end_us)
        if t_end <= t_begin:
            raise ValueError("Empty time range: the end must be after the start.")
        acc = int(max(1, params.accumulation_us))
        n, _, _ = frame_plan(t_end - t_begin, acc, 1.0)
        if n > MAX_FRAMES:
            raise ValueError(f"{n} frames per camera (accumulation {acc} us over {(t_end - t_begin) * 1e-6:.3f} s): "
                             f"more than {MAX_FRAMES}. Shorten the time range or increase the accumulation time.")
        times = t_begin + acc * np.arange(1, n + 1, dtype=np.int64)
        n_bins = int(np.clip(params.n_bins, MIN_BINS, MAX_BINS))
        with h5py.File(out_path, "w") as out:
            ds = {role: out.create_dataset(role, shape=(n, height, width), dtype=np.uint8, chunks=(1, height, width),
                                           compression="lzf") for role in readers}
            counts = {role: np.zeros(n, np.int64) for role in readers}
            windows = {role: iter_windows(rd, t_begin, acc, n) for role, rd in readers.items()}
            for k in range(n):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("cancelled")
                for role, rd in readers.items():
                    ev = next(windows[role])
                    if params.polarity == "BOTH":  # Number of events shown in the frame
                        counts[role][k] = ev["t"].size
                    else:
                        counts[role][k] = int(np.count_nonzero((ev["p"] > 0) == (params.polarity == "ON")))
                    bins = (int(times[k]) - acc, acc, n_bins) if params.time_bins else None
                    ds[role][k] = render_state(ev, width, height, params.polarity, luts[role], bins)
                if progress is not None and (k % 10 == 0 or k == n - 1):
                    progress(k + 1, n)
            out.create_dataset("times", data=times)
            for role in readers:
                out.create_dataset(f"counts_{role}", data=counts[role])
            out.attrs["params"] = json.dumps(asdict(params))
            out.attrs["t_begin_us"] = t_begin
            out.attrs["width"], out.attrs["height"] = width, height
            for role, rd in readers.items():
                out.attrs[f"serial_{role}"] = rd.serial
                out.attrs[f"source_{role}"] = str(rd.path)
        return {"frames": n, "width": width, "height": height, "seconds": time.time() - t0_wall,
                "t_begin_us": t_begin}
    finally:
        for rd in readers.values():
            rd.close()


def run_preprocess(rec_files: dict, out_path: str, params: dict, msg_q, cancel_evt) -> None:
    """Process entry point (keeps the heavy reading/rendering out of the UI process)."""
    try:
        summary = preprocess(rec_files, out_path, PreprocessParams(**params),
                             progress=lambda k, n: msg_q.put(("progress", k, n)), cancel=cancel_evt,
                             log=lambda s: msg_q.put(("log", s)))
        msg_q.put(("done", summary))
    except InterruptedError:
        msg_q.put(("cancelled",))
    except Exception as e:
        import traceback
        msg_q.put(("error", f"{e}\n{traceback.format_exc(limit=3)}"))


class FrameFile:
    """Read access to the preprocessed frames."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = h5py.File(self.path, "r")
        self.times = self.f["times"][:]
        self.n = len(self.times)
        self.params = PreprocessParams(**json.loads(self.f.attrs["params"]))
        self.roles = [r for r in ("left", "right") if r in self.f]
        self.counts = {r: self.f[f"counts_{r}"][:] for r in self.roles}

    def frame(self, role: str, k: int) -> np.ndarray:
        return self.f[role][int(np.clip(k, 0, self.n - 1))]

    def index_for_time(self, t_us: float) -> int:
        """Frame whose window [times[k] - acc, times[k]) contains t_us."""
        return int(np.clip(np.searchsorted(self.times, t_us, "right"), 0, self.n - 1))

    def start_time(self, k: int) -> int:
        return int(self.times[k]) - int(self.params.accumulation_us)

    def close(self) -> None:
        self.f.close()
