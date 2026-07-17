"""
doa_reader.py — background reader for the XVF3800's onboard direction-of-arrival.

The reSpeaker computes DOA in hardware. We poll it over USB via the bundled
xvf_host.exe control binary and cache the latest azimuth so the GUI can read it
instantly on a detection (the USB call takes ~200 ms, too slow to do inline).

Raw mic channels are dead on this firmware, so this hardware DOA is the bearing
source for the system.
"""
import os
import re
import subprocess
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
XVF_HOST = os.path.join(_HERE, "reSpeaker_XVF3800_USB_4MIC_ARRAY",
                        "host_control", "win32", "xvf_host.exe")

_DEG_RE = re.compile(r"\(([-\d.]+)\s*deg\)")

# Flags to hide the console window when shelling out on Windows
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def read_azimuths(command="AEC_AZIMUTH_VALUES"):
    """Return list of azimuth angles in degrees from the device, or []."""
    if not os.path.exists(XVF_HOST):
        return []
    try:
        out = subprocess.run([XVF_HOST, command], capture_output=True,
                             text=True, timeout=3, cwd=os.path.dirname(XVF_HOST),
                             creationflags=_NO_WINDOW).stdout
    except Exception:
        return []
    return [float(m) for m in _DEG_RE.findall(out)]


class DOAReader:
    """
    Polls the device DOA in a daemon thread and caches the primary azimuth.

    .azimuth  -> latest device azimuth in degrees (0-360), or None
    .sources  -> full list of detected source azimuths (degrees)
    """

    def __init__(self, poll_interval=0.25, command="AEC_AZIMUTH_VALUES"):
        self.poll_interval = poll_interval
        self.command       = command
        self.azimuth       = None
        self.sources       = []
        self._lock         = threading.Lock()
        self._stop         = threading.Event()
        self._thread       = None

    @staticmethod
    def available():
        return os.path.exists(XVF_HOST)

    def start(self):
        if not self.available():
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="doa-reader")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()

    @property
    def fix(self):
        with self._lock:
            return self.azimuth, list(self.sources)

    def _loop(self):
        while not self._stop.is_set():
            az = read_azimuths(self.command)
            with self._lock:
                self.sources = az
                self.azimuth = az[0] if az else None
            time.sleep(self.poll_interval)
