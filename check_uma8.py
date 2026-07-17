import sounddevice as sd
devices = sd.query_devices()
print("All miniDSP / VocalFusion entries:")
for i, d in enumerate(devices):
    if "minidsp" in d["name"].lower() or "vocalfusion" in d["name"].lower():
        print(f"  [{i:>2}] {d['name']}")
        print(f"        in={d['max_input_channels']}  out={d['max_output_channels']}  sr={d['default_samplerate']}")
