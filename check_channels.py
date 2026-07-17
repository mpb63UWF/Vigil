import sounddevice as sd
import numpy as np
import time

SR = 16000
DURATION = 2

print("=== All input devices ===")
for i, d in enumerate(sd.query_devices()):
    ch = d['max_input_channels']
    if ch > 0:
        print(f"  Dev {i:2d}  [{ch}ch]  {d['name']}")

print("\n=== Channel signal check (clap near board repeatedly) ===\n")
time.sleep(1)

for dev in range(35):
    try:
        info = sd.query_devices(dev)
        ch_in = info['max_input_channels']
        if ch_in < 1:
            continue
        name = info['name'].lower()
        if not any(k in name for k in ['respeaker', 'microphone array', 'array']):
            continue

        ch = min(ch_in, 6)
        audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=ch,
                       device=dev, dtype='float32', blocking=True)

        active = []
        for c in range(ch):
            peak = float(np.max(np.abs(audio[:, c])))
            if peak > 0.001 and peak < 100:
                active.append(f"ch{c}(peak={peak:.3f})")

        status = ", ".join(active) if active else "ALL SILENT"
        print(f"Dev {dev:2d} [{ch_in}ch max, tested {ch}ch]  {info['name']}")
        print(f"         Active: {status}")

    except Exception as e:
        pass

print("\nDone.")
