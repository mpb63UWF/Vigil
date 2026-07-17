"""
uma8_doa_reader.py — background reader for the miniDSP UMA-8-SP's onboard
direction-of-arrival.

The UMA-8-SP (XMOS VocalFusion) computes DOA in hardware and pushes it to the
host as an unsolicited 6-byte HID interrupt report whenever its voice-activity
detector (VAD) fires or the direction changes. No control request is needed —
we just open the vendor HID interface (VID 0x2752 / PID 0x001C, MI_04) and read
the stream. This is the UMA-8 analogue of doa_reader.py, which polls the
ReSpeaker XVF3800 via xvf_host.exe. It exposes the SAME interface
(`available()`, `start()`, `stop()`, `.fix`, `.azimuth`, `.sources`) so it
drops straight into DroneDetectorGUI / RealTimeBearingGPS.

HID report layout (per the UMA-8-SP user manual):
    byte 0 : 0x06   report id
    byte 1 : 0x36   fixed marker
    byte 2 : VAD    1 = voice/activity detected, 0 = none
    byte 3 : angle high byte ┐ azimuth 0-360 deg, big-endian
    byte 4 : angle low byte  ┘
    byte 5 : Dir    nearest mic 1-6

The azimuth is RELATIVE to the array's own orientation; the pipeline converts it
to a geographic bearing via COMPASS_OFFSET_DEG, which must be re-calibrated for
this mic (its mic-1 reference differs from the ReSpeaker's).

Requires: pip install hidapi
"""
import threading
import time

try:
    import hid
    _HID_OK = True
except Exception:          # hidapi not installed
    _HID_OK = False

UMA8_VID = 0x2752
UMA8_PID = 0x001C

_REPORT_ID     = 0x06
_REPORT_MARKER = 0x36
_REPORT_LEN    = 6


def _parse_report(data):
    """Return (vad, azimuth_deg, mic_index) from a raw report, or None."""
    if len(data) < _REPORT_LEN:
        return None
    if data[0] != _REPORT_ID or data[1] != _REPORT_MARKER:
        return None
    vad     = data[2]
    azimuth = (data[3] << 8) | data[4]      # big-endian, degrees
    mic     = data[5]
    if not (0 <= azimuth <= 360):
        return None
    return vad, float(azimuth), mic


class UMA8DOAReader:
    """
    Reads the UMA-8-SP onboard DOA in a daemon thread and caches the latest
    azimuth so the GUI can read it instantly on a detection.

    .azimuth  -> latest device azimuth in degrees (0-360), or None
    .sources  -> latest azimuth as a single-element list (device reports one DOA)
    .vad      -> True if the device currently reports voice/activity
    .fix      -> (azimuth, sources) snapshot, matching doa_reader.DOAReader
    """

    def __init__(self, hold_s=2.0):
        # hold_s: how long a cached azimuth stays valid after the last report
        # before .azimuth reverts to None (the device only emits on activity).
        self.hold_s   = hold_s
        self.azimuth  = None
        self.sources  = []
        self.vad      = False
        self.mic      = None
        self._last_ts = 0.0
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._thread  = None
        self._dev     = None

    @staticmethod
    def available():
        """True if hidapi is present and a UMA-8 HID interface is connected."""
        if not _HID_OK:
            return False
        try:
            return len(hid.enumerate(UMA8_VID, UMA8_PID)) > 0
        except Exception:
            return False

    def start(self):
        if not self.available():
            return False
        try:
            self._dev = hid.device()
            self._dev.open(UMA8_VID, UMA8_PID)
            # Non-blocking reads are reliable on Windows hidapi; the blocking
            # read(timeout_ms=...) path can silently stall on this platform.
            self._dev.set_nonblocking(True)
        except Exception:
            self._dev = None
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="uma8-doa-reader")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None

    @property
    def fix(self):
        with self._lock:
            # Expire a stale cached azimuth: the device stops emitting when the
            # sound stops, and a several-seconds-old bearing is misleading.
            if self.azimuth is not None and \
                    (time.time() - self._last_ts) > self.hold_s:
                return None, []
            return self.azimuth, list(self.sources)

    def _loop(self):
        # Non-blocking poll with a short sleep: reliable on Windows hidapi and
        # still near-idle on CPU. The device pushes reports only while there is
        # acoustic activity, so most polls return nothing.
        while not self._stop.is_set():
            try:
                data = self._dev.read(64)
            except Exception:
                break
            if not data:
                time.sleep(0.005)
                continue
            parsed = _parse_report(data)
            if parsed is None:
                continue
            vad, azimuth, mic = parsed
            with self._lock:
                self.vad      = bool(vad)
                self.azimuth  = azimuth
                self.sources  = [azimuth]
                self.mic      = mic
                self._last_ts = time.time()


# Backwards/interchangeable name so callers can do `from uma8_doa_reader import
# DOAReader` exactly like the ReSpeaker module.
DOAReader = UMA8DOAReader


def _monitor():
    """Live readout — run `python uma8_doa_reader.py` and make noise near the mic."""
    if not UMA8DOAReader.available():
        if not _HID_OK:
            print("hidapi not installed. Run:  pip install hidapi")
        else:
            print("No UMA-8 HID interface found (VID 0x2752 / PID 0x001C).")
        return
    r = UMA8DOAReader()
    if not r.start():
        print("Could not open the UMA-8 HID interface.")
        return
    print("Reading UMA-8 onboard DOA. Move a sound around the array. Ctrl+C to stop.\n")
    try:
        while True:
            az, _ = r.fix
            if az is None:
                print("  (no recent activity)                         ", end="\r")
            else:
                vad = "VAD" if r.vad else "   "
                print(f"  azimuth={az:6.1f}°   mic={r.mic}   {vad}        ", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        r.stop()


if __name__ == "__main__":
    _monitor()
