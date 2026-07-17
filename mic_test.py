"""Quick mic test — records 3 seconds from device 28 and reports signal levels."""
import sounddevice as sd
import numpy as np

DEVICE = 28
RATE   = 48000
SECS   = 3

print(f"Recording {SECS}s from device {DEVICE} — make some noise...")
audio = sd.rec(int(SECS * RATE), samplerate=RATE, channels=2,
               device=DEVICE, dtype='float32')
sd.wait()

for ch in range(2):
    rms  = np.sqrt(np.mean(audio[:, ch] ** 2))
    peak = np.max(np.abs(audio[:, ch]))
    bar  = '#' * int(peak * 40)
    print(f"  ch{ch}: RMS={rms:.5f}  peak={peak:.4f}  {bar}")

if np.max(np.abs(audio)) < 1e-6:
    print("\nSILENT — mic not picking up anything. Check USB connection and Windows privacy settings.")
else:
    print("\nSignal detected — mic is working.")
