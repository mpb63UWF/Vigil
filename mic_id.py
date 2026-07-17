"""
mic_id.py — identify which raw channel corresponds to which physical mic.
Snap your fingers RIGHT ON TOP of each mic hole when prompted.
"""
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION    = 2      # seconds to record per snap
RAW_START   = 2      # 6-ch firmware: ch2-ch5 = raw mics

def record_and_report(device, label):
    print(f"\nSnap on {label} now — recording {DURATION}s...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=6, device=device, dtype='float32', blocking=True)
    raw = audio[:, RAW_START: RAW_START + 4]
    peaks = [float(np.max(np.abs(raw[:, c]))) for c in range(4)]
    strongest = peaks.index(max(peaks))
    print(f"  ch2={peaks[0]:.4f}  ch3={peaks[1]:.4f}  ch4={peaks[2]:.4f}  ch5={peaks[3]:.4f}")
    print(f"  --> Strongest: ch{RAW_START + strongest}")
    return strongest

# Try device 17 first, fall back to 9
for dev in [17, 9]:
    try:
        test = sd.rec(int(0.1 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                      channels=6, device=dev, dtype='float32', blocking=True)
        peaks = [float(np.max(np.abs(test[:, RAW_START + c]))) for c in range(4)]
        print(f"Using device {dev}  (raw ch peaks: {[f'{p:.4f}' for p in peaks]})")
        DEVICE = dev
        break
    except Exception as e:
        print(f"Device {dev} failed: {e}")

print("\nHold board with USB at TOP.")
print("Board layout (USB at top):\n")
print("    MIC3 ●────────────● MIC2")
print("          \\          /")
print("           \\        /")
print("    MIC4 ●────────────● MIC1")
print("              [USB ↑]\n")

input("Press Enter, then snap on MIC1 (lower-right)...")
r1 = record_and_report(DEVICE, "MIC1 lower-right")

input("Press Enter, then snap on MIC2 (upper-right)...")
r2 = record_and_report(DEVICE, "MIC2 upper-right")

input("Press Enter, then snap on MIC3 (upper-left)...")
r3 = record_and_report(DEVICE, "MIC3 upper-left")

input("Press Enter, then snap on MIC4 (lower-left)...")
r4 = record_and_report(DEVICE, "MIC4 lower-left")

print("\n=== CHANNEL MAP RESULT ===")
print(f"  MIC1 (lower-right) → ch{RAW_START + r1}")
print(f"  MIC2 (upper-right) → ch{RAW_START + r2}")
print(f"  MIC3 (upper-left)  → ch{RAW_START + r3}")
print(f"  MIC4 (lower-left)  → ch{RAW_START + r4}")
