import sounddevice as sd
import numpy as np

print("=== Input Devices ===")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"{i:2d}: {d['name'][:50]} | ch={d['max_input_channels']} | sr={d['default_samplerate']}")

print("\n=== Quick level test on each reSpeaker device (2 seconds each) ===")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0 and "respeaker" in d["name"].lower():
        sr = int(d["default_samplerate"])
        ch = min(2, d["max_input_channels"])
        try:
            audio = sd.rec(int(sr * 2), samplerate=sr, channels=ch, dtype="float32", device=i)
            sd.wait()
            level = float(np.max(np.abs(audio)))
            print(f"  Device {i} ({d['name'][:40]}): peak level = {level:.6f}")
        except Exception as e:
            print(f"  Device {i} ({d['name'][:40]}): ERROR - {e}")
