"""Analysis helpers for the line-laser modulation experiments (see modulation_detectability.ipynb).

Recordings come from 04_dual_cam_recorder (one HDF5 file per camera, timestamps in us on the shared hardware-synced
clock). Folder names follow <frequency>_<dwell time>, e.g. 200kHz_50us.

Main ideas
----------
* Hot pixels: a few pixels firing at MHz rates are removed first (they dominate the right camera's event rate).
* Sweeps: the laser scan produces bursts of events ("sweeps"); only those periods are analysed.
* Per-pixel timing: intervals between consecutive events of the same pixel (the refractory floor bounds the rate at
  which a pixel can follow a modulation).
* Population spectra: per sweep, ON/OFF event counts in 1 us bins -> FFT. |F(f)|^2 normalised by its local mean is
  ~Exp(1) for unlocked events; summing S sweeps gives Gamma(S, 1). A carrier that phase-locks events shows up as an
  excess at f_mod (equivalent to a Rayleigh test of the event phases).
* Positive control: the motor steps (period = dwell + overhead) lock the events; the harmonics k*f_step measure up to
  which frequency the event timing still carries phase information ("temporal transfer").
"""
from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04_dual_cam_recorder"))
import recorder_core as rc  # noqa: E402

REC_ROOT = ROOT / "04_dual_cam_recorder" / "recordings"
FOLDERS = {"trail filter ON": "step_time_10us_with_trail_filter",
           "filters OFF": "step_time_10us"}
CACHE_DIR = Path(__file__).resolve().parent / "cache"
ANALYSIS_VERSION = 7

HOT_RATE = 1000.0          # ev/s: a pixel above this mean rate is "hot" (normal pixels: < 10 ev/s)
FFT_LEN = 2 ** 20          # 1 us bins -> 1.05 s per transform, 0.95 Hz resolution
BLOCK = 1024               # bins per normalisation block (~977 Hz)
SEARCH_HALF_WIDTH = 2000.0  # Hz searched around the nominal modulation frequency (generator/camera clock mismatch)
HARMONIC_FMAX = 150e3      # Hz: highest step harmonic evaluated
P_DETECT = 1e-3            # global false-alarm probability used as detection threshold
SEARCH_FREQS = (100e3, 200e3, 300e3)
ISI_EDGES = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6,
                      1e7, 1e9])


# --------------------------------------------------------------------------------------------------------- recordings
@dataclass(frozen=True)
class Recording:
    folder: str      # label, e.g. "filters OFF"
    name: str        # e.g. "200kHz_50us"
    fmod_hz: float
    dwell_us: int
    role: str        # "left" / "right"
    path: Path

    @property
    def key(self) -> str:
        return f"{FOLDERS[self.folder]}__{self.name}__{self.role}"

    @property
    def label(self) -> str:
        return f"{self.folder} | {self.name} | {self.role}"


def parse_combo(name: str) -> Tuple[float, int]:
    """'200kHz_50us' -> (200000.0, 50)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)kHz_(\d+)us", name)
    if not m:
        raise ValueError(f"Unexpected folder name {name!r} (expected <freq>kHz_<dwell>us)")
    return float(m.group(1)) * 1e3, int(m.group(2))


def list_recordings(folders: Dict[str, str] = FOLDERS) -> List[Recording]:
    recs = []
    for label, sub in folders.items():
        base = REC_ROOT / sub
        if not base.is_dir():
            continue
        combos = []
        for d in base.iterdir():
            try:
                combos.append((parse_combo(d.name), d))
            except ValueError:
                pass
        for (fmod, dwell), d in sorted(combos, key=lambda c: (c[0][0], c[0][1])):
            for role, p in rc.find_recording_files(d).items():
                recs.append(Recording(label, d.name, fmod, dwell, role, Path(p)))
    return recs


# ------------------------------------------------------------------------------------------------------------- events
@dataclass
class Events:
    t: np.ndarray        # int64, us from the recording start
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray        # uint8, 1 = ON
    width: int
    height: int
    duration_us: int
    attrs: dict = field(default_factory=dict)

    @property
    def pix(self) -> np.ndarray:
        return self.y.astype(np.int64) * self.width + self.x

    def subset(self, keep: np.ndarray) -> "Events":
        return Events(self.t[keep], self.x[keep], self.y[keep], self.p[keep], self.width, self.height,
                      self.duration_us, self.attrs)

    def __len__(self) -> int:
        return len(self.t)


def load_events(path) -> Events:
    r = rc.EventFileReader(path)
    try:
        t = r.ds["t"][:] - r.t_start
        ev = Events(t, r.ds["x"][:], r.ds["y"][:], r.ds["p"][:], r.width, r.height, r.t_stop - r.t_start,
                    dict(r.attrs))
    finally:
        r.close()
    return ev


def hot_pixels(ev: Events, rate_thr: float = HOT_RATE):
    """(mask over W*H pixels, list of dicts x, y, rate_evs, on_fraction) for pixels above ``rate_thr`` ev/s."""
    pix = ev.pix
    cnt = np.bincount(pix, minlength=ev.width * ev.height)
    rate = cnt / (ev.duration_us * 1e-6)
    mask = rate > rate_thr
    on = np.bincount(pix, weights=ev.p, minlength=ev.width * ev.height)
    rows = [dict(x=int(i % ev.width), y=int(i // ev.width), rate_evs=float(rate[i]), on_fraction=float(on[i] / cnt[i]))
            for i in np.argsort(rate)[::-1][:int(mask.sum())]]
    return mask, rows


def remove_pixels(ev: Events, mask: np.ndarray) -> Events:
    return ev.subset(~mask[ev.pix]) if mask.any() else ev


def rate_series(t: np.ndarray, duration_us: int, bin_us: int = 1000):
    """(bin start in s, rate in ev/s)."""
    n = int(duration_us // bin_us) + 1
    c = np.bincount(t // bin_us, minlength=n)[:n]
    return np.arange(n) * bin_us * 1e-6, c / (bin_us * 1e-6)


def find_sweeps(t: np.ndarray, duration_us: int, n_sigma: float = 6.0, merge_gap_ms: int = 20,
                min_len_ms: int = 10) -> List[Tuple[int, int]]:
    """Laser sweeps = 1 ms bins whose event count exceeds median + n_sigma robust sigma (MAD) of the background
    (gaps < merge_gap_ms merged, shorter than min_len_ms dropped). [(t0_us, t1_us)]."""
    _, rate = rate_series(t, duration_us, 1000)
    c = rate * 1e-3  # events per ms
    med = np.median(c)
    sigma = max(1.4826 * np.median(np.abs(c - med)), np.sqrt(max(med, 1.0)))
    act = np.where(c > med + n_sigma * sigma + 10)[0]
    if act.size == 0:
        return []
    brk = np.where(np.diff(act) > merge_gap_ms)[0]
    starts, ends = np.r_[act[0], act[brk + 1]], np.r_[act[brk], act[-1]] + 1
    return [(int(s) * 1000, int(e) * 1000) for s, e in zip(starts, ends) if e - s >= min_len_ms]


def stream_gaps(t: np.ndarray, min_gap_us: int = 2000, context: int = 2000, busy_us: int = 1000):
    """Gaps > min_gap_us in the event stream that occur while the stream is busy (the ``context`` events before and
    after span < busy_us each): data loss rather than a quiet scene. -> list of (t_us, gap_us)."""
    d = np.diff(t)
    out = []
    for i in np.where(d > min_gap_us)[0]:
        if context <= i < len(t) - context - 1 and t[i] - t[i - context] < busy_us and                 t[i + 1 + context] - t[i + 1] < busy_us:
            out.append((int(t[i]), int(d[i])))
    return out


# --------------------------------------------------------------------------------------------------- per-pixel timing
def isi_histograms(ev: Events, edges: np.ndarray = ISI_EDGES) -> dict:
    """Histograms of the interval between consecutive events of the same pixel, same / opposite polarity."""
    pix = ev.pix
    o = np.lexsort((ev.t, pix))
    ps, ts, pp = pix[o], ev.t[o], ev.p[o]
    same_pix = ps[1:] == ps[:-1]
    isi = (ts[1:] - ts[:-1])[same_pix]
    same_pol = (pp[1:] == pp[:-1])[same_pix]
    s = isi[same_pol]
    return dict(edges=edges, same=np.histogram(s, edges)[0], opposite=np.histogram(isi[~same_pol], edges)[0],
                same_q001=float(np.quantile(s, 0.001)) if s.size else np.nan,
                same_q01=float(np.quantile(s, 0.01)) if s.size else np.nan,
                n_pixels=int(np.unique(ps).size))


# ---------------------------------------------------------------------------------------------------------- spectra
def _counts(t_rel: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(t_rel, minlength=n)[:n].astype(np.float64)


def _normalise(power: np.ndarray) -> np.ndarray:
    """Divides a power spectrum by its mean over blocks of BLOCK bins (local white-noise level -> Exp(1))."""
    nb = (len(power) - 1) // BLOCK
    core = power[1:1 + nb * BLOCK].reshape(nb, BLOCK)
    level = np.repeat(core.mean(axis=1), BLOCK)
    out = np.ones_like(power)
    out[1:1 + nb * BLOCK] = core.ravel() / np.maximum(level, 1e-30)
    return out


def _peak_interp(power: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    m = np.where((f >= lo) & (f <= hi))[0]
    i = m[int(np.argmax(power[m]))]
    if 0 < i < len(power) - 1:  # Parabolic interpolation on log power
        a, b, c = np.log(power[i - 1:i + 2] + 1e-30)
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(f[i] + np.clip(d, -0.5, 0.5) * (f[1] - f[0]))
    return float(f[i])


def step_search_range(dwell_us: int) -> Tuple[float, float]:
    """Step frequency search range: period = dwell + 15..80 us of motor overhead."""
    return 1e6 / (dwell_us + 80.0), 1e6 / (dwell_us + 15.0)


def sweep_segments(sweeps: Sequence[Tuple[int, int]], max_len: int = FFT_LEN) -> List[Tuple[int, int]]:
    segs = []
    for s, e in sweeps:
        for a in range(s, e, max_len):
            segs.append((a, min(e, a + max_len)))
    return segs


@dataclass
class SpectralResult:
    f: np.ndarray                  # frequency axis (Hz) of the summed spectra
    pn: Dict[str, np.ndarray]      # "all"/"on"/"off": sum over segments of locally normalised power (null: Gamma(S,1))
    n_segments: int
    n_events: Dict[str, int]
    f_step: np.ndarray             # step frequency per segment
    harm_k: np.ndarray             # harmonic numbers
    harm: Dict[str, np.ndarray]    # "all"/"on"/"off": normalised power at k*f_step(segment), summed over segments
    fold_on: np.ndarray            # step-locked phase histograms (aligned on the ON fundamental)
    fold_off: np.ndarray
    fold_period_us: float
    step_log10_p: float            # global significance of the step peak in the summed spectrum


def _segment_spectra(t, p, s, e):
    L = FFT_LEN
    i0, i1 = np.searchsorted(t, [s, e])
    ts, ps = t[i0:i1] - s, p[i0:i1]
    on = ps == 1
    F_on, F_off = np.fft.rfft(_counts(ts[on], L)), np.fft.rfft(_counts(ts[~on], L))
    pn = {"all": _normalise(np.abs(F_on + F_off) ** 2), "on": _normalise(np.abs(F_on) ** 2),
          "off": _normalise(np.abs(F_off) ** 2)}
    return ts, on, pn


def sweep_spectra(t: np.ndarray, p: np.ndarray, sweeps, dwell_us: int, fold_bins: Optional[int] = None,
                  min_events: int = 1000) -> SpectralResult:
    """Per-sweep FFT of ON/OFF event counts (1 us bins), locally normalised and summed over sweeps.

    The step frequency is located on the sweep-summed spectrum, then refined per sweep (the motor speed differs
    slightly between sweeps); step harmonics and step-locked profiles use each sweep's own frequency.
    """
    f = np.fft.rfftfreq(FFT_LEN, 1e-6)
    df = f[1] - f[0]
    lo, hi = step_search_range(dwell_us)
    k_max = int(490e3 // hi)
    harm_k = np.arange(1, k_max + 1)
    segs = [(s, e) for s, e in sweep_segments(sweeps)
            if np.diff(np.searchsorted(t, [s, e]))[0] >= min_events]
    if not segs:
        raise ValueError("No usable sweep")
    pn = {k: np.zeros(len(f)) for k in ("all", "on", "off")}
    per_seg = []
    for s, e in segs:  # Pass 1: summed spectra
        ts, on, sp = _segment_spectra(t, p, s, e)
        for k in pn:
            pn[k] += sp[k]
        band = (f >= lo) & (f <= hi)  # Keep only what pass 2 needs (memory)
        per_seg.append((s, e, sp["all"][band].copy()))
    band_all = (f >= lo) & (f <= hi)
    f_band = f[band_all]
    f_glob = _peak_interp(pn["all"], f, lo, hi)
    step_log10_p = min(0.0, float(gamma_log10_sf(pn["all"][band_all].max(), len(segs))) + np.log10(band_all.sum()))
    harm = {k: np.zeros(len(harm_k)) for k in ("all", "on", "off")}
    n_ev = {"all": 0, "on": 0, "off": 0}
    f_steps, folds = [], []
    for s, e, band_p in per_seg:  # Pass 2: per-sweep step frequency, harmonics, folding
        near = np.abs(f_band - f_glob) <= 0.03 * f_glob
        i = np.where(near)[0][int(np.argmax(band_p[near]))]
        fs = f_band[i] if band_p[i] > 25 else f_glob  # Weak sweep: fall back to the global estimate
        ts, on, sp = _segment_spectra(t, p, s, e)
        # Refine fs to sub-bin precision with a Rayleigh scan on the events
        sub = ts[:: max(1, len(ts) // 300_000)].astype(np.float64)
        grid = fs + np.linspace(-1.5 * df, 1.5 * df, 31)
        z = [np.abs(np.exp(-2j * np.pi * g * 1e-6 * sub).sum()) for g in grid]
        fs = float(grid[int(np.argmax(z))])
        f_steps.append(fs)
        idx = np.clip(np.round(harm_k * fs / df).astype(int), 0, len(f) - 1)
        for k in harm:
            harm[k] += sp[k][idx]
        n_ev["all"] += len(ts); n_ev["on"] += int(on.sum()); n_ev["off"] += int((~on).sum())
        # Step-locked profile: phase of each event within the step period, aligned on the ON fundamental
        ton = ts[on][:: max(1, on.sum() // 400_000)].astype(np.float64)
        zon = np.exp(-2j * np.pi * fs * 1e-6 * ton).sum()
        ph = np.mod(ts.astype(np.float64) * fs * 1e-6 + np.angle(zon) / (2 * np.pi) + 0.5, 1.0)
        folds.append((ph[on], ph[~on]))
    period = float(1e6 / np.mean(f_steps))
    nb = fold_bins or max(8, int(round(period)))
    fold_on = sum(np.histogram(a, nb, (0, 1))[0] for a, _ in folds)
    fold_off = sum(np.histogram(b, nb, (0, 1))[0] for _, b in folds)
    return SpectralResult(f, pn, len(segs), n_ev, np.asarray(f_steps), harm_k, harm, fold_on, fold_off, period,
                          step_log10_p)


def gamma_log10_sf(x: np.ndarray, S: int) -> np.ndarray:
    """log10 P(Gamma(S,1) > x)."""
    return stats.gamma.logsf(x, S) / np.log(10)


def window_search(sr: SpectralResult, f_center: float, half_width: float = SEARCH_HALF_WIDTH, kind: str = "all"):
    """Largest normalised power within f_center +- half_width and its global false-alarm probability."""
    m = (sr.f >= f_center - half_width) & (sr.f <= f_center + half_width)
    P = sr.pn[kind][m]
    i = int(np.argmax(P))
    S, M = sr.n_segments, int(m.sum())
    log10_p = min(0.0, float(gamma_log10_sf(P[i], S)) + np.log10(M))
    thr = float(stats.gamma.isf(P_DETECT / M, S))  # Level that a peak must exceed for p_global < P_DETECT
    return dict(f_peak=float(sr.f[m][i]), peak=float(P[i]), excess=float(P[i] / S), log10_p=log10_p,
                detected=log10_p < np.log10(P_DETECT), threshold=thr, S=S, M=M,
                window_f=sr.f[m], window_p=P)


def harmonics(sr: SpectralResult, fmax: float = HARMONIC_FMAX, kind: str = "on"):
    """Normalised power at k*f_step of each sweep, summed -> (mean freqs, power, log10 p per harmonic)."""
    fk = sr.harm_k * float(np.mean(sr.f_step))
    keep = fk <= fmax
    P = sr.harm[kind][keep]
    return fk[keep], P, gamma_log10_sf(P, sr.n_segments)


def timing_spread_us(fk: np.ndarray, P: np.ndarray, S: int) -> float:
    """Gaussian-equivalent timing spread from the decay between the 1st and 2nd step harmonic.

    For events locked with a Gaussian jitter sigma, the phase-locking amplitude at f is ~exp(-2 pi^2 sigma^2 f^2);
    the normalised excess power (P - S) is proportional to its square.
    """
    if len(P) < 2 or P[0] <= S or P[1] <= S:
        return float("nan")
    r = np.sqrt((P[1] - S) / (P[0] - S))
    if r >= 1:
        return float("nan")
    return float(np.sqrt(-np.log(r) / (2 * np.pi ** 2 * (fk[1] ** 2 - fk[0] ** 2))) * 1e6)


def wideband_significance(sr: SpectralResult, kind: str = "all", fmin: float = 1000.0):
    """Per normalisation block: -log10 global p-value of its largest bin -> (block centre Hz, -log10 p)."""
    P = sr.pn[kind]
    nb = (len(P) - 1) // BLOCK
    core = P[1:1 + nb * BLOCK].reshape(nb, BLOCK)
    fc = sr.f[1:1 + nb * BLOCK].reshape(nb, BLOCK).mean(axis=1)
    mx = core.max(axis=1)
    lp = np.minimum(0.0, gamma_log10_sf(mx, sr.n_segments) + np.log10(BLOCK))
    keep = fc >= fmin
    return fc[keep], -lp[keep]


# ------------------------------------------------------------------------------------------------- injection test
def inject_locked(t: np.ndarray, fraction: float, f_hz: float, rng: np.random.Generator,
                  duty: float = 0.5) -> np.ndarray:
    """Moves a random ``fraction`` of the events into the ON half of a carrier at ``f_hz`` (same carrier cycle),
    keeping the 1 us timestamp quantisation. Simulates events that are perfectly phase-locked to the modulation."""
    t2 = t.copy()
    n = int(round(fraction * len(t)))
    if n == 0:
        return t2
    idx = rng.choice(len(t), n, replace=False)
    period = 1e6 / f_hz
    cyc = np.floor(t[idx] / period)
    t2[idx] = np.floor((cyc + rng.uniform(0, duty, n)) * period).astype(np.int64)
    return np.sort(t2)


# ------------------------------------------------------------------------------------------------ full per-file run
def analyze_recording(rec: Recording, use_cache: bool = True, isi: bool = True) -> dict:
    """Everything the notebook needs for one camera file (cached in ./cache)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cpath = CACHE_DIR / f"{rec.key}.pkl"
    mtime = rec.path.stat().st_mtime
    if use_cache and cpath.exists():
        with open(cpath, "rb") as fh:
            res = pickle.load(fh)
        if res.get("version") == ANALYSIS_VERSION and res.get("mtime") == mtime and (res.get("isi") or not isi):
            return res
    ev = load_events(rec.path)
    n_raw = len(ev)
    mask, hot = hot_pixels(ev)
    _, rate_raw = rate_series(ev.t, ev.duration_us, 1000)
    gaps = stream_gaps(ev.t)
    ev = remove_pixels(ev, mask)
    tr, rate_clean = rate_series(ev.t, ev.duration_us, 1000)
    sweeps = find_sweeps(ev.t, ev.duration_us)
    res = dict(version=ANALYSIS_VERSION, mtime=mtime, rec=rec, attrs=ev.attrs, duration_s=ev.duration_us * 1e-6,
               n_raw=n_raw, n_clean=len(ev), hot=hot, on_fraction=float(ev.p.mean()) if len(ev) else np.nan,
               rate_t=tr, rate_raw=rate_raw, rate_clean=rate_clean, sweeps=sweeps, gaps=gaps,
               peak_rate_mevs=float(rate_raw.max() / 1e6))
    if isi:
        res["isi"] = isi_histograms(ev)
    if sweeps:
        sr = sweep_spectra(ev.t, ev.p, sweeps, rec.dwell_us)
        in_sweep = sum(e - s for s, e in sweeps)
        fk, Pk, lpk = harmonics(sr, kind="on")
        _, Pk_off, lpk_off = harmonics(sr, kind="off")
        res.update(
            n_sweeps=len(sweeps), sweep_time_s=in_sweep * 1e-6, n_sweep_events=sr.n_events,
            step_log10_p=sr.step_log10_p, step_detected=sr.step_log10_p < np.log10(P_DETECT),
            sweep_len_ms=float(np.median([(e - s) * 1e-3 for s, e in sweeps])),
            f_step=sr.f_step, step_period_us=sr.fold_period_us, fold_on=sr.fold_on, fold_off=sr.fold_off,
            events_per_step=sr.n_events["all"] / (in_sweep * 1e-6 * float(np.mean(sr.f_step))),
            harm_f=fk, harm_on=Pk, harm_on_lp=lpk, harm_off=Pk_off, harm_off_lp=lpk_off, S=sr.n_segments,
            timing_spread_us=timing_spread_us(fk, Pk, sr.n_segments),
            # 100/200 kHz fundamentals + 300 kHz (3rd harmonic of a 100 kHz square wave)
            search={(fc, kind): window_search(sr, fc, kind=kind)
                    for fc in SEARCH_FREQS for kind in ("all", "on", "off")},
            wideband=wideband_significance(sr, "all"),
        )
    with open(cpath, "wb") as fh:
        pickle.dump(res, fh)
    return res


def bandwidth_khz(res: dict, log10_p: float = -4.0) -> float:
    """Highest step harmonic (kHz) still phase-locked with p < 10**log10_p (ON events)."""
    if "harm_f" not in res or not res.get("step_detected", False):
        return float("nan")
    ok = np.where(res["harm_on_lp"] < log10_p)[0]
    return float(res["harm_f"][ok.max()] / 1e3) if ok.size else 0.0
