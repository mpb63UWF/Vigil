"""
doa_monitor.py — live readout of the XVF3800's ONBOARD direction-of-arrival.

The reSpeaker computes DOA in hardware. We read it over USB via the bundled
xvf_host.exe control binary (no raw mic channels needed — those are dead on
this firmware). Move a sound around the array and watch the angle track it.

Run:
    python doa_monitor.py
"""
import os
import re
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
XVF_HOST = os.path.join(_HERE, "reSpeaker_XVF3800_USB_4MIC_ARRAY",
                        "host_control", "win32", "xvf_host.exe")

_DEG_RE = re.compile(r"\(([-\d.]+)\s*deg\)")


def read_azimuths(command="AEC_AZIMUTH_VALUES"):
    """Return list of azimuth angles in degrees from the device, or []."""
    try:
        out = subprocess.run([XVF_HOST, command], capture_output=True,
                             text=True, timeout=3,
                             cwd=os.path.dirname(XVF_HOST)).stdout
    except Exception as e:
        print(f"  xvf_host error: {e}")
        return []
    return [float(m) for m in _DEG_RE.findall(out)]


def main():
    if not os.path.exists(XVF_HOST):
        print(f"xvf_host.exe not found at:\n  {XVF_HOST}")
        return
    print("Reading onboard DOA. Move a sound around the array.")
    print("AEC_AZIMUTH_VALUES = up to 4 detected sources (degrees).")
    print("Ctrl+C to stop.\n")
    while True:
        az = read_azimuths()
        if az:
            primary = az[0]
            others = "  ".join(f"{a:6.1f}" for a in az)
            print(f"primary={primary:6.1f}°    all=[{others}]      ", end="\r")
        time.sleep(0.25)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
