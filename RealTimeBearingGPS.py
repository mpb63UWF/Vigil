r"""
RealTimeBearingGPS.py

Enhanced bearing estimation for the GUARD drone detection system.
Combines the ReSpeaker XVF3800 4-mic array with an NMEA GPS module.

Three DOA algorithms available:
  - GCC-PHAT  : fast, uses 3 opposing mic pairs
  - SRP-PHAT  : accurate, uses all 15 mic pairs
  - AMI-MUSIC : wideband UCA method (sensors-23-05061); 1.37 deg error on drone

GPS integration:
  Reads $GPRMC / $GNRMC NMEA sentences from any USB/UART GPS module.
  Geo-tags every detection with lat/lon and converts relative bearing to
  an absolute geographic bearing once you supply a compass offset.

─────────────────────────────────────────────────────────────────────────────
ReSpeaker XVF3800 — Mic Layout and Channel Mapping
─────────────────────────────────────────────────────────────────────────────

Physical mount — USB connector at TOP, board face up:

                        [USB Type-C]  ← geographic NORTH in your mount
                    Seeed DOA: 0° / geographic 0°N

              MIC4 (ch3) ○────────────○ MIC1 (ch0)
                          \          /
          geographic        \      /      geographic
             West 270°       \  /          East 90°
                              \/  ← board centre
                              /\
                             /  \
              MIC3 (ch2) ○────────────○ MIC2 (ch1)

                    Seeed DOA: 180° / geographic South

Mic positions in geographic bearings (USB = North):
  MIC1 (ch0): NE  ~45°   (lower-right on board = upper-right when USB is up)
  MIC2 (ch1): SE  ~135°  (upper-right on board = lower-right when USB is up)
  MIC3 (ch2): SW  ~225°  (upper-left  on board = lower-left  when USB is up)
  MIC4 (ch3): NW  ~315°  (lower-left  on board = upper-left  when USB is up)

COMPASS_OFFSET_DEG = 180 because the board's internal 0° (away from USB)
now points geographic South. Formula: geographic = (relative + 180) % 360

Channel mapping in 4-channel WDM-KS mode (device 25 / 26):
  ch0 = MIC1  (lower-right, ~SE)
  ch1 = MIC2  (upper-right, ~NE)
  ch2 = MIC3  (upper-left,  ~NW)
  ch3 = MIC4  (lower-left,  ~SW)

Opposing pairs used for GCC-PHAT:
  SE–NW diagonal :  ch0 (MIC1) ↔ ch2 (MIC3)
  NE–SW diagonal :  ch1 (MIC2) ↔ ch3 (MIC4)

All 6 pairs (used for SRP-PHAT):
  (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)

Measurement needed:
  MIC_ARRAY_DIAMETER = centre-to-centre distance between MIC1 and MIC3
                       (lower-right to upper-left, the full diagonal).
  Current value: 63.5 mm (2.5 in) — measure and update if different.
  Mic radius from centre: r = MIC_ARRAY_DIAMETER / 2

IMPORTANT — verify channel order on your board:
  Run:  python RealTimeBearingGPS.py --calibrate
  Follow the on-screen prompts. Once confirmed, set MIC_LAYOUT_VERIFIED = True.

─────────────────────────────────────────────────────────────────────────────
Absolute Bearing
─────────────────────────────────────────────────────────────────────────────

GCC/SRP-PHAT gives a bearing RELATIVE to mic 0 (0° = direction mic 0 faces).
To convert to a geographic (compass) bearing:

    geographic_bearing = (relative_bearing + COMPASS_OFFSET_DEG) % 360

COMPASS_OFFSET_DEG = the geographic bearing that mic 0 physically points toward.
  Example: if you mount the array with mic 0 facing North, offset = 0.
           if mic 0 faces East, offset = 90.

Quickest way to find your offset:
  1. Place a speaker due North of the array at a known distance.
  2. Read the relative bearing from the system.
  3. COMPASS_OFFSET_DEG = (360 - relative_bearing) % 360

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────

  # Real-time detection with GPS (SRP-PHAT)
  python RealTimeBearingGPS.py --gps-port COM4

  # Without GPS (bearing only)
  python RealTimeBearingGPS.py

  # Verify mic channel mapping before your experiment
  python RealTimeBearingGPS.py --calibrate

  # List audio devices to find RESPEAKER_DEVICE index
  python RealTimeBearingGPS.py --list-devices

  # Force GCC-PHAT instead of SRP-PHAT
  python RealTimeBearingGPS.py --method gcc

─────────────────────────────────────────────────────────────────────────────
Dependencies (add to requirements.txt if missing)
─────────────────────────────────────────────────────────────────────────────
  pip install sounddevice numpy scipy torch librosa pyserial
  pip install pynmea2          # optional — improves GPS parsing robustness
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from math import gcd
from scipy.signal import resample_poly

from AudioPP import AudioPreprocessor
from NeuralNetwork import DroneDetectorCNN

try:
    import serial as pyserial
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

try:
    import pynmea2
    _PYNMEA2_OK = True
except ImportError:
    _PYNMEA2_OK = False


# ═══════════════════════════════════════════════════════════════════════════
# HARDWARE CONFIGURATION — edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════

# Audio device index for the ReSpeaker XVF3800.
# Run --list-devices to find the correct index on your machine.
#   25 — Microphone Array 3 (), WDM-KS, 4 in   ← typical
#   26 — Microphone Array 4 (), WDM-KS, 4 in   ← try if 25 fails
RESPEAKER_DEVICE = 15   # legacy fallback index — auto-detect by name runs first
# Keywords to locate the mic array by name when the hardcoded index is stale.
# Covers both the old ReSpeaker XVF3800 and the new MiniDSP UMA-8 V2.
_RESPEAKER_KEYWORDS = ("respeaker", "xvf3800", "echo cancelling speakerphone",
                       "microphone array", "seeed",
                       "uma-8", "uma8", "minidsp", "usb audio device")

# UMA-8 V2: 8 channels, all raw (no processed/AEC channels).
# ch0-ch5 = ring mics (60° intervals), ch6 = centre mic, ch7 = aux/ref.
MIC_CHANNELS  = 8
RAW_MIC_START = 0        # all UMA-8 channels are raw; ring mics start at ch0
CAPTURE_RATE  = 48_000   # UMA-8 native sample rate; resampled to SAMPLE_RATE for CNN
SAMPLE_RATE   = 16_000   # CNN inference rate

# UMA-8 V2 ring geometry: 6 mics evenly spaced on a circle of radius 43 mm.
MIC_RADIUS_M = 0.043     # metres (86 mm outer ring diameter / 2)

# Set to True once you have run --calibrate and confirmed channel order.
MIC_LAYOUT_VERIFIED = False   # run --calibrate after first mount

# Bearing from geographic North to the direction mic 0 faces.
# Run --calibrate to measure this after mounting the UMA-8.
COMPASS_OFFSET_DEG = 0.0   # uncalibrated — run --calibrate

# Air temperature for speed-of-sound correction (degrees Celsius)
AIR_TEMP_C = 25.0

# ─── GPS serial port ──────────────────────────────────────────────────────
# Set to the COM port of your GPS module, e.g. "COM4" or "/dev/ttyUSB0".
# Typical GPS baud rates: 9600 (NEO-6M default), 115200 (u-blox configured)
GPS_PORT = None   # override with --gps-port
GPS_BAUD = 115200  # ESP32 GPS bridge uses 115200; direct GPS connection use 9600

# ─── LoRa serial output ────────────────────────────────────────────────────
LORA_PORT = None  # e.g. "COM3" — ESP32-S3 transmitter
LORA_BAUD = 115_200

# ─── CNN detection threshold ──────────────────────────────────────────────
CNN_THRESHOLD   = 0.75
MODEL_PATH      = "best_drone_detector_v2.pth"
INFERENCE_EVERY = 0.25  # seconds between CNN + DOA evaluations
DOA_WINDOW_S    = 0.5   # seconds of audio used for bearing estimation (shorter = more responsive)

# ─── Logging ──────────────────────────────────────────────────────────────
LOG_CSV = "bearing_gps_log.csv"
LOG_HEADER = [
    "timestamp_utc", "lat", "lon", "gps_utc",
    "relative_bearing_deg", "geographic_bearing_deg",
    "cnn_confidence", "method", "wind_speed_mps",
    "drone_est_lat", "drone_est_lon", "drone_est_range_m",
]

DRONE_RANGE_M = 100  # default assumed range for drone position projection


def _find_respeaker(required_channels: int = MIC_CHANNELS) -> int:
    """Return device index for the ReSpeaker, searching by name if the hardcoded index fails."""
    # First try the hardcoded index
    try:
        sd.check_input_settings(device=RESPEAKER_DEVICE, channels=required_channels,
                                samplerate=SAMPLE_RATE)
        return RESPEAKER_DEVICE
    except Exception:
        pass

    # Scan all devices for a name match
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name_lower = dev["name"].lower()
        if dev["max_input_channels"] < required_channels:
            continue
        if any(kw in name_lower for kw in _RESPEAKER_KEYWORDS):
            try:
                sd.check_input_settings(device=idx, channels=required_channels,
                                        samplerate=SAMPLE_RATE)
                print(f"[auto] ReSpeaker found at device {idx}: {dev['name']}")
                return idx
            except Exception:
                continue

    raise RuntimeError(
        f"ReSpeaker not found with {required_channels} channels @ {SAMPLE_RATE} Hz.\n"
        "Run with --list-devices to see all audio devices, then update RESPEAKER_DEVICE."
    )


def project_position(lat: float, lon: float,
                     bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Project a point from (lat, lon) along bearing_deg for distance_m metres."""
    import math
    R     = 6_371_000.0
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    brng  = math.radians(bearing_deg)
    d_r   = distance_m / R
    dlat  = math.asin(math.sin(lat_r) * math.cos(d_r) +
                      math.cos(lat_r) * math.sin(d_r) * math.cos(brng))
    dlon  = lon_r + math.atan2(math.sin(brng) * math.sin(d_r) * math.cos(lat_r),
                               math.cos(d_r) - math.sin(lat_r) * math.sin(dlat))
    return math.degrees(dlat), math.degrees(dlon)


# ═══════════════════════════════════════════════════════════════════════════
# Mic geometry — Cartesian positions (x=East, y=North) relative to centre
# ═══════════════════════════════════════════════════════════════════════════

# Mic positions in Cartesian (x=East/right, y=North/top) relative to centre.
# Mics sit at the four 45° diagonals of the circular board (confirmed from PCB).
# Each mic is at radius r from centre, but offset 45° from the cardinal axes.
# ch0=MIC1=SE, ch1=MIC2=NE, ch2=MIC3=NW, ch3=MIC4=SW
# MIC_POSITIONS in Cartesian (x=East, y=North) relative to board centre.
# USB at TOP = geographic North. Confirmed by mic_id.py channel sweep:
#   ch2 (raw[:,0]) = MIC2 = SW
#   ch3 (raw[:,1]) = MIC4 = NE
#   ch4 (raw[:,2]) = MIC1 = NW
#   ch5 (raw[:,3]) = MIC3 = SE
# Opposing pairs: (MIC1,MIC3) = (raw[:,2],raw[:,3]) and (MIC2,MIC4) = (raw[:,0],raw[:,1])
# UMA-8 V2 ring mic positions in Cartesian (x=East, y=North) relative to centre.
# 6 mics at 60-degree intervals; ch0 at 0deg (North by default), going clockwise.
# ch6 (centre mic) and ch7 (aux/ref) are excluded from DOA.
_angles_deg = np.arange(0, 360, 60)   # [0, 60, 120, 180, 240, 300]
MIC_POSITIONS = np.column_stack([
    MIC_RADIUS_M * np.sin(np.radians(_angles_deg)),   # x = East
    MIC_RADIUS_M * np.cos(np.radians(_angles_deg)),   # y = North
]).astype(np.float64)   # shape (6, 2)

# All 15 unique pairs from the 6-mic ring -- used by SRP-PHAT
MIC_PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]


def speed_of_sound(temp_c: float) -> float:
    """Speed of sound in dry air (m/s) at temperature temp_c (°C)."""
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


C = speed_of_sound(AIR_TEMP_C)


# ═══════════════════════════════════════════════════════════════════════════
# DOA Algorithms
# ═══════════════════════════════════════════════════════════════════════════

def _gcc_phat_tdoa(sig_i: np.ndarray, sig_j: np.ndarray, sr: int,
                   max_tau: float) -> float:
    """
    GCC-PHAT between two signals.
    Returns TDOA in seconds (positive = sig_j leads sig_i).
    max_tau caps the search window to the physically plausible range.
    """
    n  = len(sig_i) + len(sig_j) - 1
    Si = np.fft.rfft(sig_i, n=n)
    Sj = np.fft.rfft(sig_j, n=n)
    R  = Si * np.conj(Sj)
    R /= (np.abs(R) + 1e-8)
    cc = np.fft.irfft(R, n=n)

    max_shift = int(np.ceil(sr * max_tau))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / sr


def _raw_mics(audio: np.ndarray) -> np.ndarray:
    """Extract the 6 ring mic channels (ch0-5) from the UMA-8 capture buffer."""
    return audio[:, RAW_MIC_START: RAW_MIC_START + 6]


def gcc_phat_doa(audio: np.ndarray) -> float | None:
    """
    Fast 3-pair GCC-PHAT bearing using opposing ring mics from the UMA-8.
    Opposing pairs (180 degrees apart): (0,3), (1,4), (2,5).
    Pair axis bearings from North: 0, 60, 120 degrees.
    Returns bearing 0-360 (geographic), or None on failure.
    """
    audio_6ch = _raw_mics(audio)
    if audio_6ch.ndim < 2 or audio_6ch.shape[1] < 6:
        return None

    pair_dist = MIC_RADIUS_M * 2   # opposing mics are 2r = 86 mm apart
    max_tau   = pair_dist / C

    tau_03 = _gcc_phat_tdoa(audio_6ch[:, 0], audio_6ch[:, 3], SAMPLE_RATE, max_tau)
    tau_14 = _gcc_phat_tdoa(audio_6ch[:, 1], audio_6ch[:, 4], SAMPLE_RATE, max_tau)
    tau_25 = _gcc_phat_tdoa(audio_6ch[:, 2], audio_6ch[:, 5], SAMPLE_RATE, max_tau)

    # Project each normalised TDOA onto x (East) and y (North) via pair axis angle
    x = y = 0.0
    for tau, axis_deg in zip([tau_03, tau_14, tau_25], [0, 60, 120]):
        w  = np.clip(tau / max_tau, -1.0, 1.0)
        x += w * np.sin(np.radians(axis_deg))
        y += w * np.cos(np.radians(axis_deg))

    angle = np.degrees(np.arctan2(x, y)) % 360
    return round(float(angle), 1)


def srp_phat_doa(audio: np.ndarray, resolution_deg: float = 1.0) -> float | None:
    """
    SRP-PHAT bearing using all 6 mic pairs.
    Accepts full MIC_CHANNELS buffer; extracts raw mic channels automatically.
    Returns bearing 0-360° (geographic, USB=North), or None on failure.
    """
    audio_4ch = _raw_mics(audio)
    if audio_4ch.ndim < 2 or audio_4ch.shape[1] < 4:
        return None

    n_samples = audio_4ch.shape[0]

    # Precompute GCC-PHAT cross-spectra for all 6 pairs
    n_fft = n_samples + n_samples - 1
    cross_spectra: dict[tuple, np.ndarray] = {}
    for (i, j) in MIC_PAIRS:
        Si = np.fft.rfft(audio_4ch[:, i], n=n_fft)
        Sj = np.fft.rfft(audio_4ch[:, j], n=n_fft)
        R  = Si * np.conj(Sj)
        R /= (np.abs(R) + 1e-8)
        cross_spectra[(i, j)] = np.fft.irfft(R, n=n_fft)

    # Candidate directions (in radians, measured from North, clockwise)
    angles_deg = np.arange(0, 360, resolution_deg)
    angles_rad = np.deg2rad(angles_deg)

    # Unit direction vectors for each candidate (x=East, y=North)
    unit_x = np.sin(angles_rad)   # East component
    unit_y = np.cos(angles_rad)   # North component

    srp = np.zeros(len(angles_deg))

    for (i, j) in MIC_PAIRS:
        # Expected TDOA for this pair at each candidate direction
        delta = MIC_POSITIONS[i] - MIC_POSITIONS[j]   # (2,) vector
        tau_candidates = (delta[0] * unit_x + delta[1] * unit_y) / C  # (N_angles,)

        # Physical max TDOA for this pair
        pair_dist = np.linalg.norm(delta)
        max_tau   = pair_dist / C
        tau_candidates = np.clip(tau_candidates, -max_tau, max_tau)

        # Convert TDOA to sample shifts
        shift_samples = (tau_candidates * SAMPLE_RATE).astype(int)

        # Look up GCC-PHAT correlation at each expected shift
        cc = cross_spectra[(i, j)]
        for k, shift in enumerate(shift_samples):
            # Wrap index into the irfft output (circulant indexing)
            idx = shift % n_fft
            srp[k] += cc[idx]

    peak_idx = int(np.argmax(srp))
    return round(float(angles_deg[peak_idx]), 1)


def ami_music_doa(audio: np.ndarray,
                  f_low: float = 200.0,
                  f_high: float = 1200.0,
                  f0_hz: float = 700.0,
                  n_order: int = 2,
                  resolution_deg: float = 1.0,
                  n_sources: int = 1) -> float | None:
    """
    Wideband DOA for UCA using the modified AMI-MUSIC algorithm from
    sensors-23-05061 (proposed method).  Validated on the MiniDSP UMA-8:
    1.37 deg average error on a hovering drone (Table 3, paper Section 5.3).

    Algorithm (Figure 4 in paper):
      1. STFT each mic channel
      2. Build focusing matrix T(k_j) per frequency bin (Eq 22, Bessel-free)
      3. Covariance matrix R_j per bin
      4. Steered covariance R_tilde = sum_j T_j R_j T_j^H
      5. MUSIC pseudospectrum on R_tilde -> DOA peak

    Condition kr < 1 must hold for the small-argument Bessel approximation.
    At r=43 mm: f < c/(2*pi*r) ~= 1270 Hz.  Keep f_high <= 1200 Hz.
    """
    mics = _raw_mics(audio)
    if mics.ndim < 2 or mics.shape[1] < 6:
        return None

    M         = 6
    n_samples = mics.shape[0]
    n_fft     = 512          # 32 ms at 16 kHz -> 31.25 Hz/bin
    hop       = 128
    win       = np.hanning(n_fft)
    freqs     = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)

    n_frames = max(1, (n_samples - n_fft) // hop + 1)
    n_bins   = n_fft // 2 + 1

    # STFT: shape (n_bins, n_frames, M)
    X = np.zeros((n_bins, n_frames, M), dtype=np.complex128)
    for m in range(M):
        for t in range(n_frames):
            frame = mics[t * hop: t * hop + n_fft, m]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            X[:, t, m] = np.fft.rfft(frame * win)

    bin_mask = (freqs >= f_low) & (freqs <= f_high)
    bin_idxs = np.where(bin_mask)[0]
    if len(bin_idxs) == 0:
        return None

    f0 = f0_hz
    W  = np.exp(1j * 2 * np.pi / M)  # twiddle factor

    # Pre-build index matrix for circulant T (Eq 22)
    mr_idx = np.arange(M)[:, None]
    mc_idx = np.arange(M)[None, :]
    diff   = mr_idx - mc_idx                     # (M, M) integer differences
    ns     = np.arange(-n_order, n_order + 1)    # exponent range [-N..N]

    # Steered covariance accumulator
    R_tilde = np.zeros((M, M), dtype=np.complex128)

    for jj in bin_idxs:
        fj    = freqs[jj]
        ratio = f0 / fj   # (f0/fj)

        # T[mr,mc] = sum_{n=-N..N} ratio^n * W^{n*diff}
        # Shape broadcast: (2N+1,) x (M,M) -> sum over axis 0
        ratio_n = ratio ** ns                      # (2N+1,)
        W_n     = W ** (ns[:, None, None] * diff)  # (2N+1, M, M)
        T = np.einsum('n,nij->ij', ratio_n, W_n)  # (M, M)

        snap = X[jj]                          # (n_frames, M)
        R_j  = (snap.T @ snap.conj()) / max(1, n_frames)
        R_tilde += T @ R_j @ T.conj().T

    R_tilde /= len(bin_idxs)

    # MUSIC: eigendecompose R_tilde; noise subspace = M - n_sources smallest
    _, eigvecs = np.linalg.eigh(R_tilde)   # ascending eigenvalue order
    U_N = eigvecs[:, :M - n_sources]       # (M, M-1) noise subspace

    # Steering vectors using actual mic positions (consistent with srp_phat_doa)
    k0         = 2 * np.pi * f0 / C
    angles_deg = np.arange(0, 360, resolution_deg)
    angles_rad = np.deg2rad(angles_deg)
    px = MIC_POSITIONS[:M, 0]  # East (x) component of each mic
    py = MIC_POSITIONS[:M, 1]  # North (y) component of each mic

    # a(phi) shape (M, N_angles): exp(i*k0*(px*sin(phi)+py*cos(phi)))
    proj_dist = px[:, None] * np.sin(angles_rad) + py[:, None] * np.cos(angles_rad)
    A = np.exp(1j * k0 * proj_dist)   # (M, N_angles)

    # MUSIC pseudospectrum: 1 / ||U_N^H a||^2
    UN_H_A   = U_N.conj().T @ A          # (M-1, N_angles)
    spectrum = 1.0 / (np.einsum('ij,ij->j', UN_H_A.conj(), UN_H_A).real + 1e-12)

    peak_idx = int(np.argmax(spectrum))
    return round(float(angles_deg[peak_idx]), 1)


def relative_to_geographic(relative_deg: float,
                            offset_deg: float = COMPASS_OFFSET_DEG) -> float:
    """Convert array-relative bearing to geographic (compass) bearing."""
    return (relative_deg + offset_deg) % 360


def bearing_str(rel: float | None, geo: float | None) -> str:
    if rel is None:
        return "bearing=N/A"
    geo_s = f" -> {geo:.1f} deg geographic" if geo is not None else ""
    return f"bearing={rel:.1f}° (relative){geo_s}"


# ═══════════════════════════════════════════════════════════════════════════
# GPS Reader (background thread)
# ═══════════════════════════════════════════════════════════════════════════

class GPSReader:
    """
    Reads NMEA sentences from a serial GPS module in a daemon thread.
    Call .start() to begin; access .lat, .lon, .utc for the latest fix.
    Uses pynmea2 if installed, otherwise falls back to manual $GNRMC/$GPRMC
    parsing.
    """

    def __init__(self, port: str, baud: int = 9600):
        self.port   = port
        self.baud   = baud
        self.lat    = None
        self.lon    = None
        self.utc    = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

    def start(self) -> bool:
        if not _SERIAL_OK:
            print("GPS disabled: pyserial not installed. Run: pip install pyserial")
            return False
        t = threading.Thread(target=self._loop, daemon=True, name="gps-reader")
        t.start()
        print(f"GPS reader started on {self.port} @ {self.baud} baud")
        return True

    def stop(self):
        self._stop.set()

    @property
    def fix(self) -> tuple[float | None, float | None, str | None]:
        with self._lock:
            return self.lat, self.lon, self.utc

    def _loop(self):
        import serial
        try:
            with serial.Serial(self.port, self.baud, timeout=1.0) as ser:
                while not self._stop.is_set():
                    try:
                        line = ser.readline().decode("ascii", errors="ignore").strip()
                    except Exception:
                        continue
                    self._parse(line)
        except Exception as e:
            print(f"GPS serial error: {e}")

    def _parse(self, sentence: str):
        if not sentence.startswith("$"):
            return

        if _PYNMEA2_OK:
            try:
                msg = pynmea2.parse(sentence)
                if hasattr(msg, "latitude") and msg.status == "A":
                    with self._lock:
                        self.lat = msg.latitude
                        self.lon = msg.longitude
                        self.utc = str(msg.timestamp)
            except Exception:
                pass
        else:
            # Manual $GPRMC / $GNRMC parsing
            if "RMC" not in sentence:
                return
            parts = sentence.split(",")
            if len(parts) < 7 or parts[2] != "A":
                return
            try:
                lat_raw, lat_dir = float(parts[3]), parts[4]
                lon_raw, lon_dir = float(parts[5]), parts[6]

                lat_d = int(lat_raw / 100)
                lat   = lat_d + (lat_raw - lat_d * 100) / 60
                if lat_dir == "S":
                    lat = -lat

                lon_d = int(lon_raw / 100)
                lon   = lon_d + (lon_raw - lon_d * 100) / 60
                if lon_dir == "W":
                    lon = -lon

                with self._lock:
                    self.lat = lat
                    self.lon = lon
                    self.utc = parts[1]
            except (ValueError, IndexError):
                pass


# ═══════════════════════════════════════════════════════════════════════════
# CSV Logger
# ═══════════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, path: str):
        self.path = path
        write_header = not Path(path).exists()
        self._f = open(path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=LOG_HEADER)
        if write_header:
            self._w.writeheader()

    def write(self, row: dict):
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main Detector
# ═══════════════════════════════════════════════════════════════════════════

class BearingDetector:
    """
    Real-time drone detection + bearing estimation with optional GPS tagging.

    Pipeline:
      Audio → CNN (drone probability) → on detection: DOA → optional GPS tag
      → console output → CSV log → optional LoRa serial
    """

    def __init__(self, method: str = "srp", gps: GPSReader | None = None,
                 lora_port: str | None = None, wind_mps: float = 0.0,
                 drone_range_m: float = DRONE_RANGE_M, min_hits: int = 3):
        self.method        = method   # "srp" or "gcc"
        self.gps           = gps
        self.wind_mps      = wind_mps
        self.drone_range_m = drone_range_m
        self.min_hits      = min_hits
        self.logger        = Logger(LOG_CSV)

        # CNN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = DroneDetectorCNN()
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE)

        # Audio buffer (2-second rolling window, 4 channels)
        self._buf  = np.zeros((SAMPLE_RATE * 2, MIC_CHANNELS))
        self._q: queue.Queue = queue.Queue()

        # LoRa serial
        self._lora = None
        if lora_port and _SERIAL_OK:
            import serial
            self._lora = serial.Serial(lora_port, LORA_BAUD, timeout=1)
            print(f"LoRa serial open on {lora_port}")

        print(f"Detector ready | DOA={method.upper()} | device={self.device}")
        if not MIC_LAYOUT_VERIFIED:
            print("  WARNING: MIC_LAYOUT_VERIFIED = False. Run --calibrate before "
                  "collecting experiment data.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        self._q.put(indata.copy())

    def _infer(self, window: np.ndarray) -> float:
        ch0 = window[:, RAW_MIC_START]  # use raw mic (ch2) — Conference/ASR ch0 has noise suppression that kills drone signal
        if self.preprocessor.is_silent(ch0):
            return 0.0
        mel    = self.preprocessor.process(ch0)
        tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.predict_proba(tensor).item()

    def _doa(self, window: np.ndarray) -> float | None:
        if self.method == "srp":
            return srp_phat_doa(window)
        if self.method == "ami":
            return ami_music_doa(window)
        return gcc_phat_doa(window)

    def _send_lora(self, confidence: float, rel_bearing: float | None,
                   lat: float | None, lon: float | None):
        if self._lora is None:
            return
        b_val   = int(rel_bearing * 10) if rel_bearing is not None else -1
        lat_val = int(lat * 1e6) if lat is not None else 0
        lon_val = int(lon * 1e6) if lon is not None else 0
        payload = (f'{{"c":{int(confidence * 100)},"b":{b_val},'
                   f'"lat":{lat_val},"lon":{lon_val}}}\n')
        self._lora.write(payload.encode())

    def start(self):
        print("Listening... (Ctrl+C to stop)")
        # UMA-8 captures at 48 kHz; resample 3:1 to 16 kHz for CNN and DOA.
        capture_chunk = int(CAPTURE_RATE * INFERENCE_EVERY)
        _g  = gcd(SAMPLE_RATE, CAPTURE_RATE)
        _up = SAMPLE_RATE  // _g   # 1
        _dn = CAPTURE_RATE // _g   # 3
        samples_seen     = 0
        consecutive_hits = 0
        device_idx = _find_respeaker(MIC_CHANNELS)

        with sd.InputStream(
            device=device_idx,
            samplerate=CAPTURE_RATE,
            channels=MIC_CHANNELS,
            blocksize=capture_chunk,
            callback=self._audio_callback,
        ):
            while True:
                blk = self._q.get()   # (capture_chunk, MIC_CHANNELS) @ 48 kHz

                # Resample all channels to 16 kHz before buffering
                blk = resample_poly(blk, _up, _dn, axis=0).astype(np.float32)

                self._buf = np.roll(self._buf, -len(blk), axis=0)
                self._buf[-len(blk):] = blk
                samples_seen += len(blk)

                if samples_seen < SAMPLE_RATE // 2:
                    continue

                doa_samples = int(SAMPLE_RATE * DOA_WINDOW_S)
                window      = self._buf[-SAMPLE_RATE:]
                doa_window  = self._buf[-doa_samples:]
                confidence  = self._infer(window)

                if confidence >= CNN_THRESHOLD:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                if consecutive_hits >= self.min_hits:
                    consecutive_hits = 0
                    rel_bearing = self._doa(doa_window)
                    geo_bearing = (
                        relative_to_geographic(rel_bearing)
                        if rel_bearing is not None else None
                    )
                    lat, lon, gps_utc = self.gps.fix if self.gps else (None, None, None)

                    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

                    drone_lat = drone_lon = None
                    if lat is not None and geo_bearing is not None:
                        drone_lat, drone_lon = project_position(
                            lat, lon, geo_bearing, self.drone_range_m)

                    if lat is not None:
                        gps_str = f"observer=({lat:.6f},{lon:.6f})"
                    else:
                        gps_str = "GPS=no-fix"

                    if drone_lat is not None:
                        est_str = f"  drone_est=({drone_lat:.6f},{drone_lon:.6f}) ~{self.drone_range_m:.0f}m"
                    else:
                        est_str = ""

                    print(
                        f"\nDRONE DETECTED  conf={confidence:.1%}  "
                        f"{bearing_str(rel_bearing, geo_bearing)}  "
                        f"{gps_str}{est_str}"
                    )

                    self.logger.write({
                        "timestamp_utc":         ts,
                        "lat":                   lat,
                        "lon":                   lon,
                        "gps_utc":               gps_utc,
                        "relative_bearing_deg":  rel_bearing,
                        "geographic_bearing_deg": geo_bearing,
                        "cnn_confidence":        round(confidence, 4),
                        "method":                self.method.upper(),
                        "wind_speed_mps":        self.wind_mps,
                        "drone_est_lat":         round(drone_lat, 6) if drone_lat is not None else None,
                        "drone_est_lon":         round(drone_lon, 6) if drone_lon is not None else None,
                        "drone_est_range_m":     self.drone_range_m,
                    })

                    self._send_lora(confidence, rel_bearing, lat, lon)

                print(f"drone prob: {confidence:.1%}  hits:{consecutive_hits}/{self.min_hits}", end="\r")


    def stop(self):
        self.logger.close()
        if self._lora:
            self._lora.close()


# ═══════════════════════════════════════════════════════════════════════════
# Calibration routine
# ═══════════════════════════════════════════════════════════════════════════

def run_calibration():
    """
    Step-by-step routine to verify mic channel order and measure COMPASS_OFFSET_DEG.
    Places a speaker at 0°, 90°, 180°, 270° and checks the system output.
    """
    print("\n" + "═" * 62)
    print("ReSpeaker XVF3800 — Channel Calibration")
    print("═" * 62)
    print("""
Confirmed mic layout (USB at TOP, board face up):

        MIC1(ch4/NW) ●────────────● MIC4(ch3/NE)
                      \\          /
                       \\        /
                        \\      /
                         \\  /
                          \\/  centre
                          /\\
                         /  \\
                        /    \\
        MIC2(ch2/SW) ●────────────● MIC3(ch5/SE)

                         [USB ↑]
                    (faces geographic North)

Opposing pairs used for GCC-PHAT:
  MIC1(NW,ch4) <--> MIC3(SE,ch5)
  MIC2(SW,ch2) <--> MIC4(NE,ch3)

Step 1: Hold the board with USB connector pointing UP (toward you).
Step 2: Clap loudly from each direction when prompted.
""")
    input("Press Enter when the array is set up...")

    audio_q: queue.Queue = queue.Queue()

    def cb(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    chunk = int(SAMPLE_RATE * 0.5)
    buf   = np.zeros((SAMPLE_RATE, MIC_CHANNELS))
    results = {}

    for true_deg, label in [(0,   "toward the USB connector (in front of you, NORTH)"),
                             (90,  "to your RIGHT (EAST)"),
                             (180, "away from the USB connector (behind the board, SOUTH)"),
                             (270, "to your LEFT (WEST)")]:
        print(f"\n>>> Stand at {true_deg}° — {label}")
        print("    Make a sustained sharp noise (clap repeatedly or snap).")
        input("    Press Enter to start 3-second capture...")

        estimates = []
        t_end = time.time() + 3.0
        with sd.InputStream(device=_find_respeaker(MIC_CHANNELS), samplerate=SAMPLE_RATE,
                            channels=MIC_CHANNELS, blocksize=chunk,
                            callback=cb):
            samples_seen = 0
            while time.time() < t_end:
                try:
                    blk = audio_q.get(timeout=0.6)
                except queue.Empty:
                    continue
                buf = np.roll(buf, -len(blk), axis=0)
                buf[-len(blk):] = blk
                samples_seen += len(blk)
                if samples_seen >= SAMPLE_RATE:
                    est = srp_phat_doa(buf[-SAMPLE_RATE:])
                    if est is not None:
                        estimates.append(est)
                        print(f"    reading: {est:.1f}°", end="\r")

        if estimates:
            med = float(np.median(estimates))
            results[true_deg] = med
            print(f"\n    Median estimate for {true_deg}°: {med:.1f}°  "
                  f"(error: {(med - true_deg + 180) % 360 - 180:+.1f}°)")
        else:
            print("    No estimates captured — check mic device index.")

    print("\n" + "─" * 62)
    print("Calibration summary:")
    errors = []
    for true_deg, est in sorted(results.items()):
        err = (est - true_deg + 180) % 360 - 180
        errors.append(err)
        status = "OK" if abs(err) < 20 else "CHECK MIC ORDER"
        print(f"  True={true_deg:3d}°  Estimated={est:6.1f}°  Error={err:+6.1f}°  {status}")

    if errors:
        mean_err = float(np.mean(errors))
        print(f"\nMean signed error: {mean_err:+.1f}°")
        print(f"Suggested COMPASS_OFFSET_DEG = {(-mean_err) % 360:.1f}")
        print("Set this value at the top of this file and set MIC_LAYOUT_VERIFIED = True.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global GPS_PORT, COMPASS_OFFSET_DEG, C, CNN_THRESHOLD

    parser = argparse.ArgumentParser(
        description="GUARD — real-time bearing estimation with GPS"
    )
    parser.add_argument("--method",      choices=["srp", "gcc", "ami"], default="srp",
                        help="DOA algorithm: srp=SRP-PHAT, gcc=GCC-PHAT, ami=AMI-MUSIC UCA (default: srp)")
    parser.add_argument("--gps-port",    metavar="PORT",
                        help="Serial port of GPS module, e.g. COM4")
    parser.add_argument("--gps-baud",    type=int, default=GPS_BAUD)
    parser.add_argument("--lora-port",   metavar="PORT",
                        help="Serial port of ESP32-S3 LoRa transmitter")
    parser.add_argument("--compass-offset", type=float, default=COMPASS_OFFSET_DEG,
                        help="Geographic bearing that mic 0 faces (default 0 = North)")
    parser.add_argument("--drone-range",  type=float, default=DRONE_RANGE_M,
                        help="Assumed drone range in metres for position estimate (default 100)")
    parser.add_argument("--threshold",    type=float, default=CNN_THRESHOLD,
                        help="Detection confidence threshold 0-1 (default 0.75)")
    parser.add_argument("--min-hits",     type=int,   default=3,
                        help="Consecutive frames above threshold before alerting (default 3 = 0.75s)")
    parser.add_argument("--wind",        type=float, default=0.0,
                        help="Wind speed in m/s to log with detections")
    parser.add_argument("--temp-c",      type=float, default=AIR_TEMP_C,
                        help="Air temperature in °C for speed-of-sound correction")
    parser.add_argument("--calibrate",   action="store_true",
                        help="Run mic channel calibration routine")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        print(f"\nCurrent RESPEAKER_DEVICE = {RESPEAKER_DEVICE}")
        return

    if args.calibrate:
        run_calibration()
        return

    # Apply CLI overrides
    C = speed_of_sound(args.temp_c)
    COMPASS_OFFSET_DEG = args.compass_offset
    CNN_THRESHOLD = args.threshold

    gps = None
    if args.gps_port:
        gps = GPSReader(args.gps_port, args.gps_baud)
        gps.start()
        time.sleep(1.5)  # allow first NMEA fix

    detector = BearingDetector(
        method        = args.method,
        gps           = gps,
        lora_port     = args.lora_port,
        wind_mps      = args.wind,
        drone_range_m = args.drone_range,
        min_hits      = args.min_hits,
    )

    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()
        if gps:
            gps.stop()
        print("\nStopped. Log written to:", LOG_CSV)


if __name__ == "__main__":
    main()
