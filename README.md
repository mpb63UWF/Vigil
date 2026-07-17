# GUARD — Drone Audio Detection System

Real-time drone detection using a CNN trained on mel spectrograms, with direction-of-arrival (DOA) bearing estimation, GPS geo-tagging, and optional LoRa forwarding to a remote ESP32-S3 transmitter.

---

## How It Works

```
ReSpeaker XVF3800 (6ch)
  ch0/ch1 — Conference/ASR (processed audio)
  ch2–ch5 — Raw mics (used for DOA)
       │
       ▼
  Channel 0 → Bandpass (50–7500 Hz) → Mel Spectrogram → CNN → Drone prob
                                                                    │
                                                          if prob ≥ threshold
                                                                    │
                                                          SRP-PHAT / GCC-PHAT
                                                          (all 4 raw mics)
                                                                    │
                                                       Relative bearing → Geographic bearing
                                                       (+ COMPASS_OFFSET_DEG)
                                                                    │
                                                     GPS lat/lon tag + CSV log + LoRa serial
```

1. **Audio capture** — 6-channel audio streamed from the ReSpeaker XVF3800 at 16 kHz in 0.5-second chunks.
2. **Preprocessing** — Channel 0 is bandpass-filtered (50–7500 Hz) and converted to a 128-band mel spectrogram.
3. **Inference** — CNN outputs a drone-presence probability every 0.5 seconds.
4. **DOA** — On detection, SRP-PHAT (all 6 mic pairs, recommended) or GCC-PHAT (2 opposing pairs, fast) estimates bearing.
5. **Geo-tagging** — If a GPS module is connected, every detection is tagged with lat/lon and logged to CSV.
6. **LoRa output** — Optional JSON payload forwarded to an ESP32-S3 LoRa transmitter.

---

## Project Structure

```
GUARD SOFTWARE/
├── NeuralNetwork.py              # DroneDetectorCNN model definition
├── AudioPP.py                    # Audio preprocessing pipeline (bandpass + mel)
├── DroneDetectorGUI.py           # Primary GUI — live mic, simulation, file mode
├── doa_reader.py                 # ReSpeaker XVF3800 onboard DOA (via xvf_host.exe)
├── uma8_doa_reader.py            # miniDSP UMA-8-SP onboard DOA (via USB HID)
├── RealTimeBearingGPS.py         # CLI — bearing + GPS + LoRa (field use)
├── RealTimeInferenceEdgeDevice.py# CLI — bearing + LoRa (no GPS, older)
├── RealTimeInferenceSerial.py    # CLI — ESP32-S3 serial input path
├── RealTimeDirectionalMic.py     # CLI — AT897 shotgun mic via XLR-USB
├── RealTimeMic.py                # CLI — basic single-mic inference
├── TrainingLoop_v2.py            # Model training (HuggingFace dataset)
├── test_model.py                 # Pipeline test suite
├── best_drone_detector_v2.pth    # Trained weights (not in repo)
├── bearing_gps_log.csv           # Auto-generated detection log
└── reSpeaker_XVF3800_USB_4MIC_ARRAY/  # Firmware + DFU flashing guide
```

---

## Hardware

| Component | Details |
|-----------|---------|
| **ReSpeaker XVF3800** | 4-mic USB array — **6-channel firmware v2.0.8 must be flashed**. Bearing source: onboard DOA via `xvf_host.exe` |
| **miniDSP UMA-8-SP** | XMOS VocalFusion USB array (VID `0x2752`/PID `0x001C`). Alternative bearing source: onboard DOA streamed over USB HID (no firmware flash, no WinUSB/Zadig needed) |
| **GPS module** | Any NMEA USB/UART module (e.g. u-blox NEO-6M). Optional but recommended for field use |
| **LoRa transmitter** | ESP32-S3 on a serial COM port. Optional |
| **AT897 shotgun mic** | Directional mic via XLR-USB interface (Senal XU-1648). Optional alternative input |
| **GPU** | Optional — CPU is sufficient for inference at 16 kHz / 0.5 s windows |

### Firmware

The ReSpeaker must be running the **6-channel USB firmware** for DOA to work. Channels 0–1 carry processed audio; channels 2–5 carry the 4 raw mic signals used by SRP-PHAT.

Flash with:
```bash
dfu-util -R -e -a 1 -D reSpeaker_XVF3800_USB_4MIC_ARRAY/xmos_firmwares/usb/respeaker_xvf3800_usb_dfu_firmware_6chl_v2.0.8.bin
```
See `reSpeaker_XVF3800_USB_4MIC_ARRAY/xmos_firmwares/dfu_guide.md` for the full Zadig + dfu-util setup.

---

## Python Dependencies

```bash
pip install torch librosa numpy scipy sounddevice scikit-learn pyserial pynmea2 hidapi
```

| Package | Required for |
|---------|-------------|
| `torch` | CNN inference and training |
| `librosa` | Mel spectrogram, audio file loading |
| `numpy` / `scipy` | DSP, resampling, GCC-PHAT |
| `sounddevice` | Real-time audio capture |
| `scikit-learn` | Training metrics (AUC) |
| `pyserial` | GPS and LoRa serial I/O |
| `pynmea2` | Robust NMEA GPS parsing (optional but recommended) |
| `hidapi` | UMA-8-SP onboard DOA over USB HID (only if using the miniDSP array) |
| `datasets` | HuggingFace dataset — training only |
| `yt-dlp` | YouTube audio download in GUI — optional |

---

## Running the System

### GUI (recommended for most use)

```bash
python DroneDetectorGUI.py
```

- Select your audio input device from the dropdown (ReSpeaker auto-detected)
- Adjust threshold slider (default 0.75)
- Press **START** for live detection or **SIMULATE** to run against audio files
- Supports YouTube URL download for simulation
- Resamples automatically for non-16kHz devices (e.g. AT897 via XLR-USB)

### Field use — bearing + GPS

```bash
# SRP-PHAT bearing with GPS geo-tagging (recommended)
python RealTimeBearingGPS.py --gps-port COM4

# Without GPS (bearing only)
python RealTimeBearingGPS.py

# Force GCC-PHAT instead of SRP-PHAT
python RealTimeBearingGPS.py --method gcc

# List audio devices to confirm RESPEAKER_DEVICE index
python RealTimeBearingGPS.py --list-devices

# With LoRa forwarding to ESP32-S3
python RealTimeBearingGPS.py --gps-port COM4 --lora-port COM3
```

Detections are logged automatically to `bearing_gps_log.csv` with timestamp, lat/lon, relative bearing, geographic bearing, confidence, DOA method, and wind speed.

---

## Mic Layout and Bearing

Physical mount — USB connector pointing geographic **North**:

```
              [USB ↑ = North]

    MIC1(ch4/NW) ●────────────● MIC4(ch3/NE)
                  \          /
                   \        /
                    \      /
                     \  /
                      \/  centre
                      /\
                     /  \
                    /    \
    MIC2(ch2/SW) ●────────────● MIC3(ch5/SE)
```

- **GCC-PHAT** uses 2 opposing pairs: MIC1↔MIC3 and MIC2↔MIC4
- **SRP-PHAT** uses all 6 pairs — more accurate, ~5 ms slower

Geographic bearing = `(relative_bearing + COMPASS_OFFSET_DEG) % 360`

`COMPASS_OFFSET_DEG` is currently **123.0°** (measured via indoor calibration). Refine outdoors:

```bash
python RealTimeBearingGPS.py --calibrate
```

The calibration routine prompts you to clap from North/East/South/West and automatically suggests the corrected offset.

### Bearing source

The GUI auto-selects whichever array is connected as the DOA (bearing) source — the raw mic channels are processed/dead on both firmwares, so the device's own DOA engine is used:

| Array | Module | Transport | Notes |
|-------|--------|-----------|-------|
| ReSpeaker XVF3800 | `doa_reader.py` | polls `xvf_host.exe` (~200 ms) | needs the bundled host-control binary |
| miniDSP UMA-8-SP | `uma8_doa_reader.py` | USB HID interrupt stream | `pip install hidapi`; no driver swap |

Both expose the same `DOAReader` API (`available()`, `start()`, `stop()`, `.fix → (azimuth, sources)`). Verify the UMA-8 live with:

```bash
python uma8_doa_reader.py     # live azimuth readout — move a sound around the array
```

**UMA-8 caveats:**
- The reported azimuth is relative to the UMA-8's own mic-1 reference, so `COMPASS_OFFSET_DEG` must be **re-calibrated** for this mic (the 123.0° value is for the ReSpeaker).
- DOA is **VAD-gated**: the firmware emits a bearing only when its voice-activity detector fires. Confirm it tracks a drone's acoustic signature outdoors before relying on it — the CNN still does the actual detection; the HID stream only supplies the bearing.

---

## Configuration

Key constants at the top of `RealTimeBearingGPS.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `RESPEAKER_DEVICE` | `9` | sounddevice index — run `--list-devices` to verify after firmware change |
| `MIC_CHANNELS` | `6` | 6-channel firmware |
| `MIC_RADIUS_M` | `0.05385` | Centre-to-mic distance (53.85 mm) |
| `COMPASS_OFFSET_DEG` | `123.0` | Bearing mic 0 faces — calibrate outdoors |
| `CNN_THRESHOLD` | `0.75` | Minimum confidence to trigger alert |
| `AIR_TEMP_C` | `25.0` | For speed-of-sound correction |
| `GPS_BAUD` | `115200` | Match your GPS module baud rate |

---

## Model Architecture

`DroneDetectorCNN` treats the mel spectrogram as a grayscale image. A Squeeze-and-Excitation channel attention block after layer 3 focuses the network on drone-relevant frequency bands (motor harmonics, blade pass frequencies) vs. wind and background noise.

```
Input (1 × 128 × T)
  → Block 1: Conv2d(1→32)    + BN + ReLU + MaxPool2d + Dropout2d(0.1)
  → Block 2: Conv2d(32→64)   + BN + ReLU + MaxPool2d + Dropout2d(0.1)
  → Block 3: Conv2d(64→128)  + BN + ReLU + MaxPool2d + Dropout2d(0.2)
  → Squeeze-and-Excite channel attention
  → Block 4: Conv2d(128→256) + BN + ReLU + AdaptiveAvgPool(4×4)
  → FC(4096→256) + ReLU + Dropout(0.5) → FC(256→2)
Output: softmax → P(drone)
```

---

## Training

```bash
python TrainingLoop_v2.py
```

Dataset: [`geronimobasso/drone-audio-detection-samples`](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples) (~6.8 GB, auto-downloaded from HuggingFace on first run).

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Window size | 1.0 s |
| Batch size | 32 |
| Epochs | 50 |
| Validation split | 20% (file-level) |
| Test split | 10% (file-level) |

Augmentation (training set): Gaussian noise, time shift ±100 ms, pitch shift ±2 semitones, SpecAugment. Class weights upweight drone (1:2) to penalise missed detections more than false alarms.

Best model saved to `best_drone_detector_v2.pth`.

---

## Verify Pipeline

```bash
python test_model.py                          # architecture + preprocessing checks
python test_model.py --weights                # also loads best_drone_detector_v2.pth
python test_model.py --weights --audio file.wav  # single-file inference
```
#   V i g i l  
 