"""
DroneDetectorGUI.py
-------------------
Real-time drone detection GUI with live-mic and file-simulation modes.
Run with:

    python DroneDetectorGUI.py
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import os
import time
import numpy as np
import torch
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

from NeuralNetwork import DroneDetectorCNN
from AudioPP import AudioPreprocessor

# Bearing + GPS machinery is reused from RealTimeBearingGPS.py.
# Guarded so the GUI still launches if that module or its deps are missing.
try:
    from RealTimeBearingGPS import (
        srp_phat_doa,
        gcc_phat_doa,
        relative_to_geographic,
        project_position,
        GPSReader,
        RAW_MIC_START,
        MIC_CHANNELS,
        COMPASS_OFFSET_DEG,
        DRONE_RANGE_M,
    )
    _BEARING_OK = True
except Exception as _e:          # pragma: no cover - optional feature
    _BEARING_OK = False
    _BEARING_IMPORT_ERR = str(_e)
    RAW_MIC_START      = 2
    MIC_CHANNELS       = 6
    COMPASS_OFFSET_DEG = 0.0
    DRONE_RANGE_M      = 100

# Onboard hardware DOA (the raw mic channels are dead on this firmware, so the
# device's own direction-of-arrival engine is the bearing source). Pick whichever
# array is actually connected: the ReSpeaker XVF3800 (polled via xvf_host.exe) or
# the miniDSP UMA-8-SP (HID interrupt stream). Both expose the same DOAReader API.
DOAReader = None
_DOA_OK = False
try:
    from uma8_doa_reader import UMA8DOAReader
    if UMA8DOAReader.available():
        DOAReader = UMA8DOAReader
        _DOA_OK = True
except Exception:
    pass
if not _DOA_OK:
    try:
        from doa_reader import DOAReader as _RespeakerDOAReader
        if _RespeakerDOAReader.available():
            DOAReader = _RespeakerDOAReader
            _DOA_OK = True
    except Exception:
        pass

# ──────────────────────────────────────────────
_here       = os.path.dirname(os.path.abspath(__file__))
_local      = os.path.join(_here, "best_drone_detector_v2.pth")
_documents  = r"c:\Users\mbares\Documents\Drone_Audio_Project\best_drone_detector_v2.pth"
MODEL_PATH  = _local if os.path.exists(_local) else _documents

SAMPLE_RATE = 16000
CHUNK_SECS  = 0.5
HISTORY_LEN = 60   # ~30 seconds of history

# ──────────────────────────────────────────────
#  COLOURS
# ──────────────────────────────────────────────
BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#0f3460"
GREEN  = "#00d4aa"
YELLOW = "#f5a623"
RED    = "#e94560"
WHITE  = "#e0e0e0"
GREY   = "#555577"

_DIRECTIONAL_KEYWORDS = ("senal", "xu-1648", "xu1648", "xlr", "usb audio codec", "usb audio device")


def confidence_color(p):
    if p >= 0.75:
        return RED
    if p >= 0.40:
        return YELLOW
    return GREEN


_COMPASS_POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _compass_point(deg):
    """Nearest 8-point compass label for a bearing in degrees."""
    return _COMPASS_POINTS[int((deg % 360) / 45 + 0.5) % 8]


# ──────────────────────────────────────────────
#  LIVE DETECTOR  (microphone)
# ──────────────────────────────────────────────
class Detector:
    def __init__(self, result_q, device_index, threshold_var, channel=0,
                 bearing=False, gps=None, method="srp", drone_range_m=DRONE_RANGE_M,
                 doa=None, gain_var=None):
        self.result_q      = result_q
        self.device_index  = device_index
        self.threshold_var = threshold_var
        self.gain_var      = gain_var
        self.channel       = channel
        self.bearing       = bearing and _BEARING_OK
        self.gps           = gps
        self.method        = method
        self.drone_range_m = drone_range_m
        self.doa           = doa   # DOAReader (hardware) or None
        self._stop         = threading.Event()
        self.audio_q       = queue.Queue()
        # Mono buffer for plain detection; multichannel buffer for bearing mode.
        self.buffer        = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        self.mc_buffer     = None   # (samples, channels) — allocated in run()
        self.samples_seen  = 0
        self.capture_ch    = 1

    def _audio_callback(self, indata, frames, time_info, status):
        if self.bearing:
            self.audio_q.put(indata.copy())           # keep all channels
        else:
            ch = min(self.channel, indata.shape[1] - 1)
            self.audio_q.put(indata[:, ch].astype(np.float32))

    def _apply_gain(self, sig):
        g = self.gain_var.get() if self.gain_var is not None else 1.0
        if g != 1.0:
            sig = np.clip(sig * g, -1.0, 1.0).astype(np.float32)
        return sig

    def _infer(self, window):
        if self.preprocessor.is_silent(window):
            return 0.0
        mel    = self.preprocessor.process(window)
        tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            prob = self.model.predict_proba(tensor).item()
        return prob

    def _device_azimuth(self, mc_window):
        """Bearing source: onboard hardware DOA if present, else raw-mic SRP-PHAT.

        Returns the device/array azimuth in degrees (geographic offset is applied
        later in the UI thread). The raw mic channels are dead on this firmware,
        so the hardware DOA is the real source."""
        if self.doa is not None:
            az, _ = self.doa.fix
            return az
        rel = (srp_phat_doa(mc_window) if self.method == "srp"
               else gcc_phat_doa(mc_window))
        return rel

    def run(self):
        try:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_device = dev
            self.model = DroneDetectorCNN()
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=dev, weights_only=True)
            )
            self.model.eval()
            self.model.to(dev)
            self.preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE)
        except Exception as e:
            self.result_q.put({"error": f"Model load failed: {e}"})
            return

        try:
            info     = sd.query_devices(self.device_index, "input")
            dev_sr   = int(info["default_samplerate"])
            max_ch   = int(info["max_input_channels"])
        except Exception:
            dev_sr   = SAMPLE_RATE
            max_ch   = 1

        if self.bearing:
            # Need the raw mic channels (ch RAW_MIC_START .. +4)
            need = RAW_MIC_START + 4
            if max_ch < need:
                self.result_q.put({"error":
                    f"Bearing needs {need} channels, device has {max_ch}. "
                    "Select the 6-channel reSpeaker device."})
                return
            channels = min(max_ch, MIC_CHANNELS)
        else:
            channels = min(2, max_ch) or 1
        self.capture_ch = channels

        g              = gcd(SAMPLE_RATE, dev_sr)
        up             = SAMPLE_RATE // g
        down           = dev_sr      // g
        needs_resample = dev_sr != SAMPLE_RATE

        if self.bearing:
            self.mc_buffer = np.zeros((SAMPLE_RATE * 2, channels), dtype=np.float32)

        try:
            stream = sd.InputStream(
                device=self.device_index,
                samplerate=dev_sr,
                channels=channels,
                blocksize=int(dev_sr * CHUNK_SECS),
                callback=self._audio_callback,
            )
        except Exception as e:
            self.result_q.put({"error": str(e)})
            return

        with stream:
            while not self._stop.is_set():
                try:
                    chunk = self.audio_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if self.bearing:
                    if needs_resample:
                        chunk = resample_poly(chunk, up, down, axis=0).astype(np.float32)
                    n = len(chunk)
                    self.mc_buffer = np.roll(self.mc_buffer, -n, axis=0)
                    self.mc_buffer[-n:] = chunk
                    self.samples_seen += n
                    if self.samples_seen < SAMPLE_RATE:
                        continue

                    mc_window = self.mc_buffer[-SAMPLE_RATE:].copy()
                    # Detect on the working channel (raw mics are dead on this
                    # firmware; ch0/ch1 carry the audio). Bearing comes from the
                    # device's hardware DOA, not these samples.
                    det_idx   = min(self.channel, mc_window.shape[1] - 1)
                    det_ch    = self._apply_gain(mc_window[:, det_idx])
                    level     = float(np.max(np.abs(det_ch)))
                    prob      = self._infer(det_ch)
                    thresh    = self.threshold_var.get()
                    alert     = prob >= thresh

                    msg = {"prob": prob, "alert": alert, "level": level}
                    if alert:
                        # raw device azimuth; geographic offset applied in UI thread
                        msg["device_azimuth"] = self._device_azimuth(mc_window)
                    self.result_q.put(msg)
                else:
                    if needs_resample:
                        chunk = resample_poly(chunk, up, down).astype(np.float32)
                    self.buffer = np.roll(self.buffer, -len(chunk))
                    self.buffer[-len(chunk):] = chunk
                    self.samples_seen += len(chunk)
                    if self.samples_seen < SAMPLE_RATE:
                        continue
                    window = self._apply_gain(self.buffer[-SAMPLE_RATE:].copy())
                    level  = float(np.max(np.abs(window)))
                    prob   = self._infer(window)
                    thresh = self.threshold_var.get()
                    self.result_q.put({"prob": prob, "alert": prob >= thresh,
                                       "level": level})

    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────
#  FILE SIMULATOR  (offline audio file)
# ──────────────────────────────────────────────
class FileSimulator:
    def __init__(self, result_q, audio_paths, threshold_var, speed=1.0):
        self.result_q      = result_q
        self.audio_paths   = audio_paths   # list of file paths
        self.threshold_var = threshold_var
        self.speed         = speed
        self._stop         = threading.Event()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = dev
        self.model = DroneDetectorCNN()
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=dev, weights_only=True)
        )
        self.model.eval()
        self.model.to(dev)

        self.preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE)

    def _infer(self, window):
        if self.preprocessor.is_silent(window):
            return 0.0
        mel    = self.preprocessor.process(window)
        tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            prob = self.model.predict_proba(tensor).item()
        return prob

    def _run_file(self, path, file_idx, total_files):
        import librosa
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            self.result_q.put({"error": f"Cannot load {os.path.basename(path)}: {e}"})
            return

        chunk_size   = int(SAMPLE_RATE * CHUNK_SECS)
        buffer       = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        samples_seen = 0
        total_chunks = max(1, (len(audio) + chunk_size - 1) // chunk_size)

        for idx in range(total_chunks):
            if self._stop.is_set():
                return
            chunk = audio[idx * chunk_size : (idx + 1) * chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            buffer        = np.roll(buffer, -chunk_size)
            buffer[-chunk_size:] = chunk
            samples_seen += chunk_size

            if samples_seen >= SAMPLE_RATE:
                prob   = self._infer(buffer[-SAMPLE_RATE:].copy())
                thresh = self.threshold_var.get()
                overall = (file_idx + (idx + 1) / total_chunks) / total_files
                self.result_q.put({
                    "prob": prob, "alert": prob >= thresh, "progress": overall
                })

            time.sleep(CHUNK_SECS / self.speed)

    def run(self):
        total = len(self.audio_paths)
        for i, path in enumerate(self.audio_paths):
            if self._stop.is_set():
                break
            self.result_q.put({"now_playing": (os.path.basename(path), i + 1, total)})
            self._run_file(path, i, total)

        if not self._stop.is_set():
            self.result_q.put({"done": True})

    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drone Audio Detector")
        self.resizable(False, True)
        self.configure(bg=BG)

        self.result_q    = queue.Queue()
        self.detections  = []   # raw records for CSV export
        self.detector    = None
        self.det_thread  = None
        self.simulator   = None
        self.sim_thread  = None
        self.sim_paths   = []
        self.history     = [0.0] * HISTORY_LEN
        self.threshold_v = tk.DoubleVar(value=0.75)
        self.gain_v      = tk.DoubleVar(value=0.3)   # software input gain
        self.distance_v  = tk.StringVar(value="0")

        # manual-pointing bearing (single directional mic)
        self.manual_heading_v = tk.BooleanVar(value=False)
        self.heading_v        = tk.DoubleVar(value=0.0)

        # bearing / GPS state
        self.bearing_v   = tk.BooleanVar(value=False)
        self.gps_port_v  = tk.StringVar()
        self.gps         = None
        self.doa         = None   # hardware DOA reader
        self.offset_v    = tk.DoubleVar(value=float(COMPASS_OFFSET_DEG))
        self.flip_v      = tk.BooleanVar(value=True)  # device azimuth is CCW; mirror to compass CW
        self.last_geo    = None   # last geographic bearing for the compass
        self.last_azimuth = None  # last raw device azimuth

        # Scrollable container so the (tall) layout never runs off-screen.
        sh = self.winfo_screenheight()
        self.geometry(f"660x{min(980, sh - 80)}")
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)
        self._scroll = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=self._scroll.yview)
        self._scroll.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._scroll.pack(side="left", fill="both", expand=True)
        self.content = tk.Frame(self._scroll, bg=BG)
        self._scroll.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda e: self._scroll.configure(scrollregion=self._scroll.bbox("all")))
        self._scroll.bind_all(
            "<MouseWheel>",
            lambda e: self._scroll.yview_scroll(int(-e.delta / 120), "units"))

        self._build_ui()
        self._poll()

    # ── layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = {"padx": 12, "pady": 8}

        tk.Label(self.content, text="DRONE AUDIO DETECTOR",
                 font=("Helvetica", 16, "bold"),
                 bg=BG, fg=WHITE).pack(pady=(16, 4))

        # ── confidence + status ──────────────────────────────────────────
        top = tk.Frame(self.content, bg=BG)
        top.pack(fill="x", **pad)

        conf_panel = tk.Frame(top, bg=PANEL)
        conf_panel.pack(side="left", expand=True, fill="both", padx=(0, 6))
        tk.Label(conf_panel, text="CONFIDENCE",
                 font=("Helvetica", 9), bg=PANEL, fg=GREY).pack(pady=(10, 0))
        self.conf_label = tk.Label(conf_panel, text="0.0%",
                                   font=("Helvetica", 52, "bold"),
                                   bg=PANEL, fg=GREEN, width=6)
        self.conf_label.pack(pady=(0, 10))

        status_panel = tk.Frame(top, bg=PANEL)
        status_panel.pack(side="left", expand=True, fill="both", padx=(6, 0))
        tk.Label(status_panel, text="STATUS",
                 font=("Helvetica", 9), bg=PANEL, fg=GREY).pack(pady=(10, 0))
        self.status_label = tk.Label(status_panel, text="● CLEAR",
                                     font=("Helvetica", 22, "bold"),
                                     bg=PANEL, fg=GREEN)
        self.status_label.pack(pady=(0, 4))
        self.alert_count = 0
        self.alert_label = tk.Label(status_panel, text="Alerts: 0",
                                    font=("Helvetica", 10), bg=PANEL, fg=GREY)
        self.alert_label.pack(pady=(0, 10))

        # ── bearing compass + GPS readout ────────────────────────────────
        bear = tk.Frame(self.content, bg=BG)
        bear.pack(fill="x", **pad)

        compass_panel = tk.Frame(bear, bg=PANEL)
        compass_panel.pack(side="left", fill="both", padx=(0, 6))
        tk.Label(compass_panel, text="BEARING TO DRONE",
                 font=("Helvetica", 9), bg=PANEL, fg=GREY).pack(pady=(8, 0))
        self.compass_size = 150
        self.compass = tk.Canvas(compass_panel, width=self.compass_size,
                                 height=self.compass_size, bg=PANEL,
                                 highlightthickness=0)
        self.compass.pack(padx=12, pady=(2, 4))
        self.bearing_lbl = tk.Label(compass_panel, text="—°",
                                    font=("Helvetica", 14, "bold"),
                                    bg=PANEL, fg=GREY)
        self.bearing_lbl.pack(pady=(0, 8))

        gps_panel = tk.Frame(bear, bg=PANEL)
        gps_panel.pack(side="left", expand=True, fill="both", padx=(6, 0))
        tk.Label(gps_panel, text="GPS / POSITION",
                 font=("Helvetica", 9), bg=PANEL, fg=GREY).pack(pady=(8, 2), anchor="w", padx=10)
        self.gps_obs_lbl = tk.Label(gps_panel, text="Observer: no GPS",
                                    font=("Helvetica", 10), bg=PANEL, fg=WHITE,
                                    anchor="w", justify="left")
        self.gps_obs_lbl.pack(anchor="w", padx=10)
        self.gps_drone_lbl = tk.Label(gps_panel, text="Drone est: —",
                                      font=("Helvetica", 10), bg=PANEL, fg=WHITE,
                                      anchor="w", justify="left")
        self.gps_drone_lbl.pack(anchor="w", padx=10, pady=(2, 0))
        self.gps_range_lbl = tk.Label(gps_panel, text="",
                                      font=("Helvetica", 9), bg=PANEL, fg=GREY,
                                      anchor="w")
        self.gps_range_lbl.pack(anchor="w", padx=10, pady=(2, 8))

        self._draw_compass(None)

        # ── rolling graph ────────────────────────────────────────────────
        graph_frame = tk.Frame(self.content, bg=PANEL)
        graph_frame.pack(fill="x", **pad)
        tk.Label(graph_frame, text="CONFIDENCE HISTORY  (last 30 s)",
                 font=("Helvetica", 8), bg=PANEL, fg=GREY).pack(anchor="w", padx=8, pady=(6, 0))
        self.canvas_w = 500
        self.canvas_h = 90
        self.graph    = tk.Canvas(graph_frame, width=self.canvas_w,
                                  height=self.canvas_h, bg=ACCENT,
                                  highlightthickness=0)
        self.graph.pack(padx=8, pady=(2, 10))
        self._draw_graph()

        # ── detection log ────────────────────────────────────────────────
        log_frame = tk.Frame(self.content, bg=PANEL)
        log_frame.pack(fill="x", **pad)
        log_hdr = tk.Frame(log_frame, bg=PANEL)
        log_hdr.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(log_hdr, text="DETECTION LOG",
                 font=("Helvetica", 8), bg=PANEL, fg=GREY).pack(side="left")
        tk.Button(log_hdr, text="Clear", font=("Helvetica", 8),
                  bg=ACCENT, fg=WHITE, relief="flat",
                  command=self._clear_log).pack(side="right")
        tk.Button(log_hdr, text="Save CSV…", font=("Helvetica", 8),
                  bg=ACCENT, fg=WHITE, relief="flat",
                  command=self._save_csv).pack(side="right", padx=(0, 4))
        tk.Button(log_hdr, text="Save TXT…", font=("Helvetica", 8),
                  bg=ACCENT, fg=WHITE, relief="flat",
                  command=self._save_log).pack(side="right", padx=(0, 4))
        log_body = tk.Frame(log_frame, bg=PANEL)
        log_body.pack(fill="x", padx=8, pady=(2, 8))
        log_scroll = tk.Scrollbar(log_body)
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(log_body, height=4, bg=ACCENT, fg=WHITE,
                                font=("Consolas", 9), relief="flat",
                                yscrollcommand=log_scroll.set, state="disabled",
                                wrap="none")
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)
        self.log_text.tag_config("hdr", foreground=GREY)
        self._log_header_written = False

        # ── live controls ────────────────────────────────────────────────
        ctrl = tk.Frame(self.content, bg=BG)
        ctrl.pack(fill="x", **pad)

        thresh_row = tk.Frame(ctrl, bg=BG)
        thresh_row.pack(fill="x", pady=(0, 6))
        tk.Label(thresh_row, text="Threshold:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")
        self.thresh_val_lbl = tk.Label(thresh_row, text="0.75",
                                       font=("Helvetica", 10, "bold"),
                                       bg=BG, fg=YELLOW, width=4)
        self.thresh_val_lbl.pack(side="right")
        tk.Scale(thresh_row, variable=self.threshold_v,
                 from_=0.05, to=0.95, resolution=0.05,
                 orient="horizontal", bg=BG, fg=WHITE,
                 troughcolor=ACCENT, highlightthickness=0, length=300,
                 command=self._on_thresh_change).pack(side="left", expand=True, fill="x")

        gain_row = tk.Frame(ctrl, bg=BG)
        gain_row.pack(fill="x", pady=(0, 6))
        tk.Label(gain_row, text="Input gain:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")
        self.gain_val_lbl = tk.Label(gain_row, text="1.0×",
                                     font=("Helvetica", 10, "bold"),
                                     bg=BG, fg=YELLOW, width=5)
        self.gain_val_lbl.pack(side="right")
        tk.Scale(gain_row, variable=self.gain_v,
                 from_=0.1, to=20.0, resolution=0.1,
                 orient="horizontal", bg=BG, fg=WHITE,
                 troughcolor=ACCENT, highlightthickness=0, length=300,
                 command=self._on_gain_change).pack(side="left", expand=True, fill="x")

        dev_row = tk.Frame(ctrl, bg=BG)
        dev_row.pack(fill="x", pady=(0, 6))
        tk.Label(dev_row, text="Input device:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")

        devices     = sd.query_devices()
        input_devs  = [(i, d["name"]) for i, d in enumerate(devices)
                       if d["max_input_channels"] > 0]
        dev_names   = [f"{i}: {n[:40]}" for i, n in input_devs]
        self._dev_map = {name: idx for (idx, _), name in zip(input_devs, dev_names)}

        default_name = dev_names[0] if dev_names else ""
        self.dev_combo = ttk.Combobox(dev_row, values=dev_names, width=42,
                                      state="readonly")
        self.dev_combo.set(default_name)
        self.dev_combo.pack(side="left", padx=(4, 0))

        dir_match = next(
            (n for n in dev_names if any(k in n.lower() for k in _DIRECTIONAL_KEYWORDS)),
            None
        )
        # Prefer the reSpeaker entry whose native sample rate is already 16 kHz
        xvf_16k_match = next(
            (n for n in dev_names if any(k in n.lower() for k in ("xvf", "xiao", "respeaker", "uma", "minidsp"))
             and sd.query_devices(self._dev_map[n])["default_samplerate"] == 16000),
            None
        )
        xvf_match = next(
            (n for n in dev_names if any(k in n.lower() for k in ("xvf", "xiao", "respeaker", "uma", "minidsp"))),
            None
        )
        if dir_match:
            self.dev_combo.set(dir_match)
        elif xvf_16k_match:
            self.dev_combo.set(xvf_16k_match)
        elif xvf_match:
            self.dev_combo.set(xvf_match)
        else:
            try:
                default_idx = sd.default.device[0]
                for name in dev_names:
                    if name.startswith(f"{default_idx}:"):
                        self.dev_combo.set(name)
                        break
            except Exception:
                pass

        self.dev_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        ch_row = tk.Frame(ctrl, bg=BG)
        ch_row.pack(fill="x", pady=(0, 6))
        tk.Label(ch_row, text="Input channel:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")
        self.ch_v = tk.IntVar(value=0)
        self.ch_combo = ttk.Combobox(ch_row, textvariable=self.ch_v,
                                     values=["0"], width=30, state="readonly")
        self.ch_combo.set("0  (default)")
        self.ch_combo.pack(side="left", padx=(4, 0))

        # ── bearing + GPS controls ───────────────────────────────────────
        bear_row = tk.Frame(ctrl, bg=BG)
        bear_row.pack(fill="x", pady=(0, 6))
        self.bearing_chk = tk.Checkbutton(
            bear_row, text="Bearing (UMA-8 / reSpeaker mic array)",
            variable=self.bearing_v, command=self._on_bearing_toggle,
            font=("Helvetica", 10), bg=BG, fg=WHITE, selectcolor=ACCENT,
            activebackground=BG, activeforeground=WHITE, anchor="w")
        self.bearing_chk.pack(side="left")
        tk.Label(bear_row, text="N offset:", font=("Helvetica", 9),
                 bg=BG, fg=WHITE).pack(side="left", padx=(12, 2))
        tk.Spinbox(bear_row, from_=0, to=359, increment=1, width=5,
                   textvariable=self.offset_v, font=("Helvetica", 9),
                   command=self._on_offset_change).pack(side="left")
        tk.Label(bear_row, text="°",
                 font=("Helvetica", 9), bg=BG, fg=GREY).pack(side="left", padx=(2, 0))
        tk.Checkbutton(bear_row, text="mirror E/W", variable=self.flip_v,
                       command=self._on_offset_change,
                       font=("Helvetica", 9), bg=BG, fg=WHITE, selectcolor=ACCENT,
                       activebackground=BG, activeforeground=WHITE).pack(side="left", padx=(10, 0))
        if not _BEARING_OK:
            self.bearing_chk.config(state="disabled")
            tk.Label(bear_row, text="(import failed)",
                     font=("Helvetica", 8, "italic"), bg=BG, fg=RED).pack(side="left", padx=(6, 0))

        # Manual-pointing bearing for a single directional mic: you aim the mic
        # and enter the compass heading it points at. On a detection, that heading
        # is logged as the drone bearing and fed to the GPS projection.
        manual_row = tk.Frame(ctrl, bg=BG)
        manual_row.pack(fill="x", pady=(0, 6))
        tk.Checkbutton(manual_row, text="Manual heading (point mic)",
                       variable=self.manual_heading_v, command=self._on_manual_toggle,
                       font=("Helvetica", 10), bg=BG, fg=WHITE, selectcolor=ACCENT,
                       activebackground=BG, activeforeground=WHITE).pack(side="left")
        tk.Label(manual_row, text="heading:", font=("Helvetica", 9),
                 bg=BG, fg=WHITE).pack(side="left", padx=(12, 2))
        tk.Spinbox(manual_row, from_=0, to=359, increment=1, width=5,
                   textvariable=self.heading_v, font=("Helvetica", 9),
                   command=self._on_heading_change).pack(side="left")
        tk.Label(manual_row, text="°  (compass dir the mic points)",
                 font=("Helvetica", 8, "italic"), bg=BG, fg=GREY).pack(side="left", padx=(2, 0))

        gps_row = tk.Frame(ctrl, bg=BG)
        gps_row.pack(fill="x", pady=(0, 6))
        tk.Label(gps_row, text="GPS port:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")
        self.gps_combo = ttk.Combobox(gps_row, textvariable=self.gps_port_v,
                                      values=self._list_serial_ports(),
                                      width=14, state="normal")
        self.gps_combo.pack(side="left", padx=(4, 6))
        self.gps_btn = tk.Button(gps_row, text="Connect GPS",
                                 font=("Helvetica", 9), bg=PANEL, fg=WHITE,
                                 relief="flat", command=self._toggle_gps)
        self.gps_btn.pack(side="left")
        self.gps_status = tk.Label(gps_row, text="(optional)",
                                   font=("Helvetica", 9, "italic"),
                                   bg=BG, fg=GREY)
        self.gps_status.pack(side="left", padx=(8, 0))

        self.phantom_label = tk.Label(ctrl, text="",
                                      font=("Helvetica", 9, "italic"),
                                      bg=BG, fg=YELLOW)
        self.phantom_label.pack(fill="x", pady=(0, 4))
        self._on_device_change()

        dist_row = tk.Frame(ctrl, bg=BG)
        dist_row.pack(fill="x", pady=(0, 6))
        tk.Label(dist_row, text="Test distance:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")
        tk.Spinbox(dist_row, from_=0, to=9999, increment=1, width=7,
                   textvariable=self.distance_v, font=("Helvetica", 10),
                   bg=ACCENT, fg=WHITE, insertbackground=WHITE,
                   buttonbackground=ACCENT).pack(side="left", padx=(4, 4))
        tk.Label(dist_row, text="m  (logged with every detection)",
                 font=("Helvetica", 9, "italic"), bg=BG, fg=GREY).pack(side="left")

        btn_row = tk.Frame(ctrl, bg=BG)
        btn_row.pack(pady=(8, 4))
        self.start_btn = tk.Button(btn_row, text="▶  START",
                                   font=("Helvetica", 12, "bold"),
                                   bg=GREEN, fg=BG, relief="flat",
                                   padx=20, pady=6,
                                   command=self.start_detection)
        self.start_btn.pack(side="left", padx=8)
        self.stop_btn = tk.Button(btn_row, text="■  STOP",
                                  font=("Helvetica", 12, "bold"),
                                  bg=RED, fg=WHITE, relief="flat",
                                  padx=20, pady=6, state="disabled",
                                  command=self.stop_detection)
        self.stop_btn.pack(side="left", padx=8)

        self.info_label = tk.Label(self.content, text="Press START to begin",
                                   font=("Helvetica", 9), bg=BG, fg=GREY)
        self.info_label.pack(pady=(0, 4))

        # ── signal level meter ───────────────────────────────────────────
        level_row = tk.Frame(self.content, bg=BG)
        level_row.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(level_row, text="Input level:", font=("Helvetica", 9),
                 bg=BG, fg=GREY, width=10, anchor="w").pack(side="left")
        self.level_canvas = tk.Canvas(level_row, width=300, height=14,
                                      bg=ACCENT, highlightthickness=0)
        self.level_canvas.pack(side="left", padx=(4, 8))
        self.level_val_lbl = tk.Label(level_row, text="—",
                                      font=("Helvetica", 9), bg=BG, fg=GREY, width=8)
        self.level_val_lbl.pack(side="left")

        # ── simulation section ───────────────────────────────────────────
        tk.Frame(self.content, bg=GREY, height=1).pack(fill="x", padx=12, pady=(4, 0))

        sim_outer = tk.Frame(self.content, bg=BG)
        sim_outer.pack(fill="x", padx=12, pady=6)

        tk.Label(sim_outer, text="SIMULATION MODE",
                 font=("Helvetica", 9, "bold"), bg=BG, fg=GREY).pack(anchor="w")

        # File row
        file_row = tk.Frame(sim_outer, bg=BG)
        file_row.pack(fill="x", pady=(6, 2))
        tk.Label(file_row, text="Audio files:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=11, anchor="w").pack(side="left")
        self.sim_path_v = tk.StringVar(value="No files selected")
        tk.Entry(file_row, textvariable=self.sim_path_v,
                 font=("Helvetica", 9), bg=ACCENT, fg=WHITE,
                 insertbackground=WHITE, width=28,
                 relief="flat", state="readonly").pack(side="left", padx=(4, 4))
        tk.Button(file_row, text="Files…",
                  font=("Helvetica", 9), bg=PANEL, fg=WHITE, relief="flat",
                  command=self._browse_files).pack(side="left", padx=(0, 4))
        tk.Button(file_row, text="Folder…",
                  font=("Helvetica", 9), bg=PANEL, fg=WHITE, relief="flat",
                  command=self._browse_folder).pack(side="left")

        # YouTube row
        yt_row = tk.Frame(sim_outer, bg=BG)
        yt_row.pack(fill="x", pady=(2, 2))
        tk.Label(yt_row, text="YouTube URL:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=11, anchor="w").pack(side="left")
        self.yt_url_v = tk.StringVar()
        tk.Entry(yt_row, textvariable=self.yt_url_v,
                 font=("Helvetica", 9), bg=ACCENT, fg=WHITE,
                 insertbackground=WHITE, width=35,
                 relief="flat").pack(side="left", padx=(4, 4))
        self.yt_btn = tk.Button(yt_row, text="Download",
                                font=("Helvetica", 9), bg=PANEL, fg=WHITE,
                                relief="flat", command=self._download_yt)
        self.yt_btn.pack(side="left")

        # Speed + Simulate row
        spd_row = tk.Frame(sim_outer, bg=BG)
        spd_row.pack(fill="x", pady=(6, 4))
        tk.Label(spd_row, text="Speed:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=11, anchor="w").pack(side="left")
        self.speed_v = tk.StringVar(value="1.0")
        speed_cb = ttk.Combobox(spd_row, textvariable=self.speed_v,
                                values=["0.5", "1.0", "2.0", "4.0", "8.0"],
                                width=5, state="readonly")
        speed_cb.set("1.0")
        speed_cb.pack(side="left", padx=(4, 16))

        # Progress bar
        self.sim_progress = ttk.Progressbar(spd_row, length=200, mode="determinate")
        self.sim_progress.pack(side="left", padx=(0, 12))

        self.sim_btn = tk.Button(spd_row, text="▶  SIMULATE",
                                 font=("Helvetica", 12, "bold"),
                                 bg=YELLOW, fg=BG, relief="flat",
                                 padx=16, pady=6,
                                 command=self.start_simulation)
        self.sim_btn.pack(side="left")

        tk.Label(self.content, text="", bg=BG).pack(pady=(0, 8))

    # ── level meter ───────────────────────────────────────────────────────
    def _draw_level(self, level: float):
        c = self.level_canvas
        c.delete("all")
        w, h = 300, 14
        # level is peak absolute value in [0, 1]
        fill_w = int(min(level * w * 8, w))   # scale up so normal speech fills bar
        col = RED if level > 0.1 else (YELLOW if level > 0.01 else GREEN)
        if fill_w > 0:
            c.create_rectangle(0, 0, fill_w, h, fill=col, outline="")
        # silent gate marker at 1e-6 (basically leftmost pixel)
        self.level_val_lbl.config(
            text=f"{level:.4f}",
            fg=RED if level > 0.1 else (YELLOW if level > 0.001 else GREY)
        )

    # ── compass / bearing ─────────────────────────────────────────────────
    def _draw_compass(self, geo):
        """Draw a compass rose; geo = geographic bearing in degrees, or None."""
        import math
        c = self.compass
        c.delete("all")
        s   = self.compass_size
        cx  = cy = s / 2
        r   = s / 2 - 14

        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=GREY, width=2)
        for ang, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
            a  = math.radians(ang)
            lx = cx + (r + 8) * math.sin(a)
            ly = cy - (r + 8) * math.cos(a)
            c.create_text(lx, ly, text=lbl, fill=WHITE, font=("Helvetica", 8, "bold"))
        c.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=GREY, outline="")

        if geo is None:
            return
        a   = math.radians(geo)
        tx  = cx + r * math.sin(a)
        ty  = cy - r * math.cos(a)
        c.create_line(cx, cy, tx, ty, fill=RED, width=3, arrow="last")

    def _show_geo(self, geo, dev=None):
        """Draw the compass for a geographic bearing and update the label."""
        if geo is None:
            return None
        self.last_geo = geo
        self._draw_compass(geo)
        dev_s = f"   dev {dev:.0f}°" if dev is not None else ""
        self.bearing_lbl.config(text=f"{geo:.0f}°  ({_compass_point(geo)}){dev_s}", fg=RED)
        return geo

    def _apply_bearing(self, azimuth):
        """Apply mirror + N-offset to a device azimuth, update compass, return geographic deg."""
        if azimuth is None:
            return None
        self.last_azimuth = azimuth
        sign = -1.0 if self.flip_v.get() else 1.0
        geo = (sign * azimuth + self.offset_v.get()) % 360
        return self._show_geo(geo, dev=azimuth)

    def _on_offset_change(self):
        # Re-render the compass for the last reading using the new offset.
        az = getattr(self, "last_azimuth", None)
        if az is not None:
            self._apply_bearing(az)

    # ── detection log ──────────────────────────────────────────────────────
    def _log_detection(self, prob, dev_az, geo, lat, lon, dlat, dlon, dist_m=0):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        bear = f"{geo:6.1f}° {_compass_point(geo):<2}" if geo is not None else "   —    "
        devs = f"{dev_az:5.0f}°" if dev_az is not None else "  —  "
        gps  = f"{lat:.5f},{lon:.5f}" if lat is not None else "no-fix"
        drone = f"{dlat:.5f},{dlon:.5f}" if dlat is not None else "—"
        line = (f"{ts}  conf={prob*100:5.1f}%  bearing={bear}  dev={devs}  "
                f"obs={gps}  drone~={drone}\n")
        self.detections.append({
            "timestamp":        ts,
            "distance_m":       dist_m,
            "confidence":       round(prob, 4),
            "bearing_deg":      round(geo, 1) if geo is not None else "",
            "bearing_compass":  _compass_point(geo) if geo is not None else "",
            "device_azimuth":   round(dev_az, 1) if dev_az is not None else "",
            "observer_lat":     lat if lat is not None else "",
            "observer_lon":     lon if lon is not None else "",
            "drone_est_lat":    dlat if dlat is not None else "",
            "drone_est_lon":    dlon if dlon is not None else "",
        })
        self.log_text.config(state="normal")
        if not self._log_header_written:
            self.log_text.insert("end",
                "timestamp            confidence  bearing        dev    observer(lat,lon)     drone_est\n",
                "hdr")
            self._log_header_written = True
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self._log_header_written = False
        self.detections.clear()

    def _save_log(self):
        from tkinter import filedialog
        text = self.log_text.get("1.0", "end").strip()
        if not text:
            self.info_label.config(text="Detection log is empty.", fg=YELLOW)
            return
        path = filedialog.asksaveasfilename(
            title="Save detection log",
            defaultextension=".txt",
            initialfile=time.strftime("detection_log_%Y%m%d_%H%M%S.txt"),
            filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            self.info_label.config(text=f"Saved log: {os.path.basename(path)}", fg=GREEN)

    def _save_csv(self):
        import csv
        if not self.detections:
            self.info_label.config(text="No detections to export.", fg=YELLOW)
            return
        path = filedialog.asksaveasfilename(
            title="Save detections as CSV",
            defaultextension=".csv",
            initialfile=time.strftime("detections_%Y%m%d_%H%M%S.csv"),
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            fieldnames = [
                "timestamp", "distance_m", "confidence", "bearing_deg", "bearing_compass",
                "device_azimuth", "observer_lat", "observer_lon",
                "drone_est_lat", "drone_est_lon",
            ]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.detections)
            self.info_label.config(text=f"Saved CSV: {os.path.basename(path)}", fg=GREEN)

    def _list_serial_ports(self):
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()]
        except Exception:
            return []

    def _on_bearing_toggle(self):
        if self.bearing_v.get():
            self.info_label.config(
                text="Bearing mode: select the UMA-8 (8-ch) or reSpeaker device.", fg=YELLOW)
        else:
            self._draw_compass(None)
            self.bearing_lbl.config(text="—°", fg=GREY)

    def _toggle_gps(self):
        # Disconnect if already connected
        if self.gps is not None:
            try:
                self.gps.stop()
            except Exception:
                pass
            self.gps = None
            self.gps_btn.config(text="Connect GPS")
            self.gps_status.config(text="(disconnected)", fg=GREY)
            self.gps_obs_lbl.config(text="Observer: no GPS")
            return

        if not _BEARING_OK:
            self.gps_status.config(text="bearing module unavailable", fg=RED)
            return
        port = self.gps_port_v.get().strip()
        if not port:
            self.gps_status.config(text="enter a COM port", fg=RED)
            return
        try:
            self.gps = GPSReader(port, 115200)
            if not self.gps.start():
                self.gps = None
                self.gps_status.config(text="pyserial missing", fg=RED)
                return
            self.gps_btn.config(text="Disconnect")
            self.gps_status.config(text=f"listening on {port}", fg=GREEN)
        except Exception as e:
            self.gps = None
            self.gps_status.config(text=f"error: {str(e)[:24]}", fg=RED)

    # ── graph ─────────────────────────────────────────────────────────────
    def _draw_graph(self):
        c  = self.graph
        c.delete("all")
        w, h = self.canvas_w, self.canvas_h
        pad  = 4

        thresh = self.threshold_v.get()
        ty     = h - pad - (h - 2 * pad) * thresh
        c.create_line(0, ty, w, ty, fill=YELLOW, dash=(4, 4), width=1)
        c.create_text(w - 4, ty - 6, text=f"{thresh:.0%}",
                      fill=YELLOW, font=("Helvetica", 7), anchor="e")

        if not self.history:
            return
        bar_w = (w - 2 * pad) / HISTORY_LEN
        for i, val in enumerate(self.history):
            x0  = pad + i * bar_w
            y0  = h - pad - (h - 2 * pad) * val
            col = confidence_color(val) if val > 0.02 else GREY
            c.create_rectangle(x0, y0, x0 + bar_w - 1, h - pad, fill=col, outline="")

    # ── callbacks ──────────────────────────────────────────────────────────
    def _on_thresh_change(self, val):
        self.thresh_val_lbl.config(text=f"{float(val):.2f}")
        self._draw_graph()

    def _on_gain_change(self, val):
        self.gain_val_lbl.config(text=f"{float(val):.1f}×")

    def _on_manual_toggle(self):
        if self.manual_heading_v.get():
            self._show_geo(self.heading_v.get() % 360)
            self.info_label.config(
                text="Manual heading: aim the mic, set the heading, detections log that bearing.",
                fg=YELLOW)
        else:
            self._draw_compass(None)
            self.bearing_lbl.config(text="—°", fg=GREY)

    def _on_heading_change(self):
        if self.manual_heading_v.get():
            self._show_geo(self.heading_v.get() % 360)

    def _browse_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio files (hold Ctrl/Shift for multiple)",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.webm *.opus"),
                ("All files", "*.*"),
            ]
        )
        if paths:
            self.sim_paths = sorted(paths)
            self._update_file_display()

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder of audio files")
        if not folder:
            return
        exts  = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"}
        files = sorted(
            os.path.join(folder, n) for n in os.listdir(folder)
            if os.path.splitext(n)[1].lower() in exts
        )
        if files:
            self.sim_paths = files
            self._update_file_display()
        else:
            self.info_label.config(text="No audio files found in that folder.", fg=RED)

    def _update_file_display(self):
        n = len(self.sim_paths)
        if n == 0:
            self.sim_path_v.set("No files selected")
        elif n == 1:
            self.sim_path_v.set(os.path.basename(self.sim_paths[0]))
        else:
            self.sim_path_v.set(f"{n} files selected")

    def _download_yt(self):
        url = self.yt_url_v.get().strip()
        if not url:
            self.info_label.config(text="Paste a YouTube URL first.", fg=RED)
            return
        self.yt_btn.config(state="disabled")
        self.info_label.config(text="Downloading audio from YouTube…", fg=YELLOW)

        def _do_download():
            try:
                import yt_dlp
            except ImportError:
                self.result_q.put({"error": "yt-dlp not installed. Run: pip install yt-dlp"})
                return

            out_tmpl = os.path.join(_here, "%(title).60s.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": out_tmpl,
                "quiet": True,
                "no_warnings": True,
                # download without ffmpeg post-processing; librosa handles most formats
                "postprocessors": [],
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info  = ydl.extract_info(url, download=True)
                    ext   = info.get("ext", "webm")
                    title = info.get("title", "audio")[:60]
                    fname = os.path.join(_here, f"{title}.{ext}")
                self.result_q.put({"yt_done": fname})
            except Exception as e:
                self.result_q.put({"error": f"Download failed: {e}"})

        threading.Thread(target=_do_download, daemon=True).start()

    # ── device selection ───────────────────────────────────────────────────
    def _on_device_change(self, event=None):
        name = self.dev_combo.get()
        if any(k in name.lower() for k in _DIRECTIONAL_KEYWORDS):
            self.phantom_label.config(
                text="⚠  Directional mic detected — confirm 48V phantom power is ON before starting"
            )
        else:
            self.phantom_label.config(text="")

        # Update channel list for selected device
        dev_idx = self._dev_map.get(name, -1)
        try:
            info = sd.query_devices(dev_idx, "input")
            n_ch = int(info["max_input_channels"])
        except Exception:
            n_ch = 1
        is_respeaker = any(k in name.lower() for k in ("xvf", "respeaker", "uma", "minidsp"))
        is_uma8 = any(k in name.lower() for k in ("uma", "uma-8", "minidsp"))
        ch_labels = []
        for c in range(n_ch):
            if is_uma8:
                if c < 6:
                    ch_labels.append(f"{c}  (ring mic, {c*60} deg)")
                elif c == 6:
                    ch_labels.append("6  (centre mic)")
                else:
                    ch_labels.append(f"{c}  (aux/reference)")
            elif is_respeaker and c == 0:
                ch_labels.append("0  (processed / AEC out - carries audio)")
            elif is_respeaker and c == 1:
                ch_labels.append("1  (processed out - carries audio)")
            elif is_respeaker:
                ch_labels.append(f"{c}  (raw mic - silent on this firmware)")
            else:
                ch_labels.append(str(c))
        self.ch_combo["values"] = ch_labels
        # Default: ch0 for UMA-8 (all raw), ch1 for reSpeaker, ch0 otherwise
        if is_uma8:
            self.ch_combo.set(ch_labels[0])
        elif is_respeaker and n_ch > 1:
            self.ch_combo.set(ch_labels[1])
        else:
            self.ch_combo.set(ch_labels[0])

        # Auto-enable bearing when the array exposes enough raw mic channels.
        can_bear = _BEARING_OK and is_respeaker and n_ch >= (RAW_MIC_START + 6)
        if can_bear:
            self.bearing_chk.config(state="normal")
            self.bearing_v.set(True)
        else:
            self.bearing_v.set(False)
            if not is_respeaker:
                self._draw_compass(None)
                self.bearing_lbl.config(text="—°", fg=GREY)

    # ── live detection ─────────────────────────────────────────────────────
    def start_detection(self):
        sel_name       = self.dev_combo.get()
        dev_idx        = self._dev_map.get(sel_name, -1)
        is_directional = any(k in sel_name.lower() for k in _DIRECTIONAL_KEYWORDS)

        sel_ch  = int(self.ch_combo.get().split()[0])
        bearing = self.bearing_v.get()

        # Start the hardware DOA reader (bearing source) if available.
        if bearing and _DOA_OK and self.doa is None:
            self.doa = DOAReader()
            if not self.doa.start():
                self.doa = None
        elif not bearing and self.doa is not None:
            self.doa.stop(); self.doa = None

        self.detector   = Detector(self.result_q, dev_idx, self.threshold_v,
                                   channel=sel_ch, bearing=bearing, gps=self.gps,
                                   doa=self.doa, gain_var=self.gain_v)
        self.det_thread = threading.Thread(target=self.detector.run, daemon=True)
        self.det_thread.start()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal", command=self.stop_detection)
        self.sim_btn.config(state="disabled")
        self.dev_combo.config(state="disabled")
        if bearing:
            mode = "Bearing mode — listening for drone direction…"
        elif is_directional:
            mode = "Directional mic — listening…"
        else:
            mode = "Listening…"
        self.info_label.config(text=mode, fg=GREEN)
        self.alert_count = 0
        self.alert_label.config(text="Alerts: 0", fg=GREY)

    def stop_detection(self):
        if self.detector:
            self.detector.stop()
            self.detector = None
        if self.doa:
            self.doa.stop()
            self.doa = None
        self._reset_ui("Stopped.")

    # ── simulation ─────────────────────────────────────────────────────────
    def start_simulation(self):
        if not self.sim_paths:
            self.info_label.config(text="Select files or a folder first.", fg=RED)
            return
        missing = [p for p in self.sim_paths if not os.path.exists(p)]
        if missing:
            self.info_label.config(text=f"{len(missing)} file(s) no longer found.", fg=RED)
            return

        speed = float(self.speed_v.get())
        self.simulator  = FileSimulator(self.result_q, self.sim_paths, self.threshold_v, speed)
        self.sim_thread = threading.Thread(target=self.simulator.run, daemon=True)
        self.sim_thread.start()

        self.sim_btn.config(state="disabled")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal", command=self._stop_simulation)
        self.yt_btn.config(state="disabled")
        self.sim_progress["value"] = 0
        n = len(self.sim_paths)
        self.info_label.config(
            text=f"Simulating {n} file{'s' if n > 1 else ''}  ({speed}×)", fg=YELLOW
        )
        self.alert_count = 0

    def _stop_simulation(self):
        if self.simulator:
            self.simulator.stop()
            self.simulator = None
        self._reset_ui("Simulation stopped.")

    def _sim_finished(self):
        self.simulator = None
        self.sim_progress["value"] = 100
        self._reset_ui("Simulation complete.")

    def _reset_ui(self, msg=""):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled", command=self.stop_detection)
        self.sim_btn.config(state="normal")
        self.dev_combo.config(state="readonly")
        self.yt_btn.config(state="normal")
        self.info_label.config(text=msg, fg=GREY)
        self.conf_label.config(text="0.0%", fg=GREEN)
        self.status_label.config(text="● CLEAR", fg=GREEN)
        self.alert_label.config(text="Alerts: 0", fg=GREY)
        self._draw_compass(None)
        self.bearing_lbl.config(text="—°", fg=GREY)

    # ── poll queue ─────────────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                item = self.result_q.get_nowait()

                if "error" in item:
                    self._reset_ui()
                    self.info_label.config(text=f"Error: {item['error'][:60]}", fg=RED)
                    break

                if "yt_done" in item:
                    path = item["yt_done"]
                    if path not in self.sim_paths:
                        self.sim_paths.append(path)
                    self._update_file_display()
                    self.yt_btn.config(state="normal")
                    fname = os.path.basename(path)
                    self.info_label.config(
                        text=f"Downloaded: {fname[:45]}  — click SIMULATE", fg=GREEN
                    )
                    break

                if "now_playing" in item:
                    name, idx, total = item["now_playing"]
                    self.info_label.config(
                        text=f"[{idx}/{total}]  {name[:48]}", fg=YELLOW
                    )
                    continue

                if "done" in item:
                    self._sim_finished()
                    break

                prob  = item["prob"]
                alert = item["alert"]
                if "level" in item:
                    self._draw_level(item["level"])

                self.history.append(prob)
                if len(self.history) > HISTORY_LEN:
                    self.history.pop(0)

                col = confidence_color(prob)
                self.conf_label.config(text=f"{prob:.1%}", fg=col)

                if alert:
                    self.alert_count += 1
                    self.status_label.config(text="● DRONE DETECTED", fg=RED)
                    self.alert_label.config(text=f"Alerts: {self.alert_count}", fg=RED)
                else:
                    self.status_label.config(text="● CLEAR", fg=GREEN)
                    self.alert_label.config(fg=GREY)

                # ── bearing: array hardware DOA, else manual pointing ──────
                geo = dev_az = lat = lon = dlat = dlon = None
                if "device_azimuth" in item:
                    dev_az = item.get("device_azimuth")
                    geo    = self._apply_bearing(dev_az)
                elif alert and self.manual_heading_v.get():
                    geo = self.heading_v.get() % 360
                    self._show_geo(geo)

                # ── GPS tagging + projection (any mode, on a detection) ────
                if alert:
                    if geo is not None:
                        if self.gps is not None:
                            lat, lon, _utc = self.gps.fix
                        if lat is not None:
                            self.gps_obs_lbl.config(text=f"Observer: {lat:.6f}, {lon:.6f}")
                            if _BEARING_OK:
                                rng = self.detector.drone_range_m if self.detector else DRONE_RANGE_M
                                dlat, dlon = project_position(lat, lon, geo, rng)
                                self.gps_drone_lbl.config(text=f"Drone est: {dlat:.6f}, {dlon:.6f}")
                                self.gps_range_lbl.config(text=f"assumed range {rng:.0f} m")
                        else:
                            self.gps_obs_lbl.config(
                                text="Observer: GPS no fix" if self.gps else "Observer: no GPS")
                    self._log_detection(prob, dev_az, geo, lat, lon, dlat, dlon,
                                        self.distance_v.get())

                if "progress" in item:
                    self.sim_progress["value"] = item["progress"] * 100

                self._draw_graph()
        except queue.Empty:
            pass

        # Live hardware-DOA readout on the compass, independent of detection.
        # Lets you calibrate the offset with ANY loud sound (no drone needed).
        if self.doa is not None:
            az, _ = self.doa.fix
            if az is not None:
                self._apply_bearing(az)

        self.after(200, self._poll)


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
