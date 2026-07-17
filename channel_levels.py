"""
channel_levels.py — live peak-level bar meter for ALL 6 device channels.

Tells us whether the reSpeaker USB channels are 4 spatially-distinct RAW mics
(needed for bearing) or processed/beamformed channels (bearing impossible).

Run it, then clap over each corner in turn and watch which channels move:
  - REAL raw mics  -> the channel nearest your clap jumps highest; the loudest
                      channel CHANGES as you move the clap around the board.
  - Processed out  -> the same channel(s) stay loudest no matter where you clap,
                      or ch3/4/5 barely move at all.

Run:
    python channel_levels.py
"""
import queue
import numpy as np
import sounddevice as sd

from RealTimeBearingGPS import _find_respeaker, MIC_CHANNELS, SAMPLE_RATE

BAR_W = 30


def bar(level, peak=0.3):
    n = int(min(level / peak, 1.0) * BAR_W)
    return "#" * n + "-" * (BAR_W - n)


def main():
    dev = _find_respeaker(MIC_CHANNELS)
    print(f"Using device {dev}  ({MIC_CHANNELS} ch @ {SAMPLE_RATE} Hz)")
    print("Clap over each corner. Watch which channel bars jump.\n"
          "Ctrl+C to stop.\n")

    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata.copy())

    chunk = int(SAMPLE_RATE * 0.15)
    with sd.InputStream(device=dev, samplerate=SAMPLE_RATE,
                        channels=MIC_CHANNELS, blocksize=chunk, callback=cb):
        while True:
            blk = q.get()
            peaks = [float(np.max(np.abs(blk[:, c]))) for c in range(MIC_CHANNELS)]
            loudest = int(np.argmax(peaks))
            lines = []
            for c in range(MIC_CHANNELS):
                mark = " <<" if c == loudest else ""
                lines.append(f"ch{c} {peaks[c]:.4f} [{bar(peaks[c])}]{mark}")
            # redraw block in place
            print("\033[H\033[J" + "\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
