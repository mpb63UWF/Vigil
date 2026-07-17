"""
bearing_monitor.py — live DOA + per-channel level monitor (field diagnostic)

Prints the SRP-PHAT bearing and the peak level of each raw mic every ~0.3 s.
Use it to confirm the array actually resolves direction before a flight:

  1. Run it.
  2. Make a LOUD sustained sound (clap repeatedly / play a tone) and walk it
     around the array.
  3. The 'bearing' number should track the source. The 4 raw-mic levels should
     differ as you move (the nearest mic reads highest).

If the bearing never moves off one value, a fixed source (usually WIND) is
dominating — shield the mics / add foam windscreens and try again.

Run:
    python bearing_monitor.py
"""
import queue
import numpy as np
import sounddevice as sd

from RealTimeBearingGPS import (
    srp_phat_doa, gcc_phat_doa, relative_to_geographic,
    _find_respeaker, RAW_MIC_START, MIC_CHANNELS, SAMPLE_RATE,
)

_POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
def _pt(d): return _POINTS[int((d % 360) / 45 + 0.5) % 8]


def main():
    dev = _find_respeaker(MIC_CHANNELS)
    print(f"Using device {dev}  ({MIC_CHANNELS} ch @ {SAMPLE_RATE} Hz)")
    print("Move a loud sound around the array — bearing should follow it.")
    print("Ctrl+C to stop.\n")

    q: queue.Queue = queue.Queue()
    buf = np.zeros((SAMPLE_RATE, MIC_CHANNELS), dtype=np.float32)

    def cb(indata, frames, t, status):
        q.put(indata.copy())

    chunk = int(SAMPLE_RATE * 0.3)
    seen = 0
    with sd.InputStream(device=dev, samplerate=SAMPLE_RATE,
                        channels=MIC_CHANNELS, blocksize=chunk, callback=cb):
        while True:
            blk = q.get()
            n = len(blk)
            buf = np.roll(buf, -n, axis=0)
            buf[-n:] = blk
            seen += n
            if seen < SAMPLE_RATE:
                continue

            raw = buf[:, RAW_MIC_START:RAW_MIC_START + 4]
            levels = np.max(np.abs(raw), axis=0)
            rel_srp = srp_phat_doa(buf)
            rel_gcc = gcc_phat_doa(buf)
            geo = relative_to_geographic(rel_srp) if rel_srp is not None else None

            lvl_str = " ".join(f"m{c}={levels[c]:.3f}" for c in range(4))
            geo_str = f"{geo:6.1f}° {_pt(geo):<2}" if geo is not None else "  --  "
            print(f"SRP rel={rel_srp!s:>6}  GCC rel={rel_gcc!s:>6}  "
                  f"geo={geo_str}   {lvl_str}", end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
