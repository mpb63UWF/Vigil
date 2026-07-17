"""
BearingAccuracyExperiment.py

Structured bearing accuracy experiment for the GUARD SOFTWARE drone detection system.
Implements the minimum publishable experiment protocol:
  - Acoustic source at 8 known angles (every 45 degrees)
  - Minimum 3 trials per position
  - Two distance conditions (10 m and 30 m)
  - Wind speed logged at each distance condition
  - Outputs raw CSV, statistics table, and Figure 1 (polar + error bar plots)

Usage:
  python BearingAccuracyExperiment.py

To resume a partial session:
  python BearingAccuracyExperiment.py --resume

To analyse existing results without running new trials:
  python BearingAccuracyExperiment.py --analyse-only
"""

import argparse
import csv
import json
import os
import queue
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Hardware configuration — match RealTimeInferenceEdgeDevice.py
# ---------------------------------------------------------------------------
RESPEAKER_DEVICE   = 9    # reSpeaker XVF3800 6-ch firmware (fallback: 17)       # WDM-KS 4-channel; try 26 if 25 fails
MIC_CHANNELS       = 6    # 6-ch firmware
RAW_MIC_START      = 2   # ch0=Conference, ch1=ASR, ch2-5=raw mics
MIC_ARRAY_DIAMETER = 0.10772  # metres — adjacent mics are 3" apart (square); opposing = 3"×√2 = 107.7 mm
SAMPLE_RATE        = 16_000   # Hz
# Systematic offset measured by indoor calibration (0° and 90° both showed -123° error).
# Converts raw array-relative bearing to USB-referenced bearing.
# Validate/refine outdoors with a sustained tone source.
COMPASS_OFFSET_DEG = 123.0

# ---------------------------------------------------------------------------
# Experiment protocol
# ---------------------------------------------------------------------------
TARGET_ANGLES_DEG  = [0, 45, 90, 135, 180, 225, 270, 315]
TRIALS_PER_POS     = 3        # minimum for publication; increase if time allows
DISTANCES_M        = [10, 30]
CAPTURE_SECONDS    = 6        # audio captured per trial (yields ~12 GCC-PHAT readings)
GCC_INTERVAL_S     = 0.5      # one GCC-PHAT estimate every 0.5 s

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
RAW_CSV    = "bearing_results_raw.csv"
STATS_CSV  = "bearing_results_stats.csv"
FIGURE_PNG = "figure1_bearing_accuracy.png"
SESSION_LOG = "bearing_session_log.json"

RAW_HEADER = [
    "timestamp", "distance_m", "wind_speed_mps",
    "true_bearing_deg", "trial_num",
    "estimated_bearing_deg", "signed_error_deg", "abs_error_deg",
]


# ---------------------------------------------------------------------------
# GCC-PHAT bearing estimation (extracted from RealTimeInferenceEdgeDevice.py)
# ---------------------------------------------------------------------------

def _gcc_phat(sig1: np.ndarray, sig2: np.ndarray, sr: int) -> float:
    """Return TDOA in seconds between sig1 and sig2."""
    max_tau = MIC_ARRAY_DIAMETER / 343.0
    n = len(sig1) + len(sig2) - 1

    S1 = np.fft.rfft(sig1, n=n)
    S2 = np.fft.rfft(sig2, n=n)
    R  = S1 * np.conj(S2)
    R /= (np.abs(R) + 1e-8)
    cc = np.fft.irfft(R, n=n)

    max_shift = int(np.ceil(sr * max_tau))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / sr


def compute_doa(audio: np.ndarray) -> float | None:
    """
    GCC-PHAT bearing from full 6-channel capture buffer.
    Extracts raw mics ch2-ch5 (6-ch firmware: ch0=Conference, ch1=ASR).
    """
    if audio.ndim < 2 or audio.shape[1] < RAW_MIC_START + 4:
        return None

    raw = audio[:, RAW_MIC_START: RAW_MIC_START + 4]
    c = 343.0
    d = MIC_ARRAY_DIAMETER

    # Confirmed pairs (mic_id.py): MIC1(NW)=raw[:,2]↔MIC3(SE)=raw[:,3], MIC2(SW)=raw[:,0]↔MIC4(NE)=raw[:,1]
    tau_02 = _gcc_phat(raw[:, 2], raw[:, 3], SAMPLE_RATE)
    tau_13 = _gcc_phat(raw[:, 0], raw[:, 1], SAMPLE_RATE)

    tau_02 = np.clip(tau_02, -d / c, d / c)
    tau_13 = np.clip(tau_13, -d / c, d / c)

    # Mics are on 45° diagonals (MIC1=SE, MIC2=NE, MIC3=NW, MIC4=SW).
    # Project diagonal TDOAs onto East/North to recover compass bearing.
    raw_angle = np.degrees(np.arctan2(tau_02 + tau_13, tau_13 - tau_02)) % 360
    corrected = (raw_angle + COMPASS_OFFSET_DEG) % 360
    return round(corrected, 1)


def bearing_error(estimated: float, true: float) -> float:
    """
    Signed bearing error in degrees, wrapped to [-180, 180].
    Positive = clockwise overshoot.
    """
    err = (estimated - true + 180) % 360 - 180
    return round(err, 2)


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

def capture_bearings(duration_s: float, verbose: bool = True) -> list[float]:
    """
    Capture `duration_s` seconds from the ReSpeaker array and return a list
    of GCC-PHAT bearing estimates taken every GCC_INTERVAL_S seconds.
    Returns [] if hardware is not available.
    """
    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio status] {status}")
        audio_q.put(indata.copy())

    chunk  = int(SAMPLE_RATE * GCC_INTERVAL_S)
    window_len = SAMPLE_RATE  # 1-second analysis window
    buffer = np.zeros((window_len, MIC_CHANNELS))
    samples_seen = 0
    estimates: list[float] = []

    try:
        with sd.InputStream(
            device=RESPEAKER_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=MIC_CHANNELS,
            blocksize=chunk,
            callback=callback,
        ):
            deadline = time.time() + duration_s
            while time.time() < deadline:
                try:
                    blk = audio_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                buffer = np.roll(buffer, -len(blk), axis=0)
                buffer[-len(blk):] = blk
                samples_seen += len(blk)

                if samples_seen >= window_len:
                    est = compute_doa(buffer)
                    if est is not None:
                        estimates.append(est)
                        if verbose:
                            print(f"  GCC-PHAT reading: {est:.1f}°", end="\r")

    except Exception as exc:
        print(f"\n  ERROR opening mic: {exc}")
        print("  Check that RESPEAKER_DEVICE={RESPEAKER_DEVICE} is correct.")
        print("  Run:  python -c \"import sounddevice as sd; print(sd.query_devices())\"")
        return []

    if verbose:
        print()  # newline after \r updates
    return estimates


def median_bearing(estimates: list[float]) -> float | None:
    """Circular median of a list of bearings (handles 0/360 wraparound)."""
    if not estimates:
        return None
    rad = np.deg2rad(estimates)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    return round(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360, 1)


# ---------------------------------------------------------------------------
# Session state — save/load so experiments can be resumed
# ---------------------------------------------------------------------------

def load_session(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": []}  # list of (distance_m, angle_deg, trial_num) tuples


def save_session(session: dict, path: str):
    with open(path, "w") as f:
        json.dump(session, f, indent=2)


def already_done(session: dict, dist: float, angle: float, trial: int) -> bool:
    return [dist, angle, trial] in session["completed"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def append_raw(row: dict):
    write_header = not os.path.exists(RAW_CSV)
    with open(RAW_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_raw() -> list[dict]:
    if not os.path.exists(RAW_CSV):
        return []
    with open(RAW_CSV, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(rows: list[dict]) -> list[dict]:
    """
    Group raw results by (distance_m, true_bearing_deg) and compute:
      n, mean_signed_error, mean_abs_error, std_dev, RMSE
    """
    from collections import defaultdict
    groups: dict[tuple, list[float]] = defaultdict(list)

    for r in rows:
        key = (float(r["distance_m"]), float(r["true_bearing_deg"]))
        groups[key].append(float(r["abs_error_deg"]))

    stats = []
    for (dist, angle), abs_errors in sorted(groups.items()):
        # collect signed errors for the same group
        signed = [
            float(r["signed_error_deg"])
            for r in rows
            if float(r["distance_m"]) == dist and float(r["true_bearing_deg"]) == angle
        ]
        stats.append({
            "distance_m":       dist,
            "true_bearing_deg": angle,
            "n_trials":         len(abs_errors),
            "mean_signed_error_deg": round(float(np.mean(signed)), 2),
            "mean_abs_error_deg":    round(float(np.mean(abs_errors)), 2),
            "std_dev_deg":           round(float(np.std(abs_errors, ddof=1)) if len(abs_errors) > 1 else 0.0, 2),
            "rmse_deg":              round(float(np.sqrt(np.mean(np.array(abs_errors) ** 2))), 2),
        })
    return stats


def write_stats_csv(stats: list[dict]):
    with open(STATS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    print(f"\nStatistics written to: {STATS_CSV}")


def print_stats_table(stats: list[dict]):
    """Pretty-print the statistics as an ASCII table (mirrors Table 1)."""
    header = f"{'Dist(m)':>8} {'Angle°':>7} {'N':>3} {'MeanErr°':>9} {'MAE°':>6} {'SD°':>6} {'RMSE°':>7}"
    print("\n" + "=" * len(header))
    print("TABLE 1 — Bearing Accuracy Statistics")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['distance_m']:>8.0f} {s['true_bearing_deg']:>7.0f} "
            f"{s['n_trials']:>3} {s['mean_signed_error_deg']:>+9.2f} "
            f"{s['mean_abs_error_deg']:>6.2f} {s['std_dev_deg']:>6.2f} "
            f"{s['rmse_deg']:>7.2f}"
        )
    print("=" * len(header))

    # Overall per-distance summary
    for dist in sorted(set(s["distance_m"] for s in stats)):
        subset = [s for s in stats if s["distance_m"] == dist]
        mae_vals = [s["mean_abs_error_deg"] for s in subset]
        print(
            f"\n  {dist:.0f} m overall: "
            f"mean MAE = {np.mean(mae_vals):.2f}°, "
            f"max MAE = {np.max(mae_vals):.2f}°"
        )


# ---------------------------------------------------------------------------
# Figure 1 — two-panel publication figure
# ---------------------------------------------------------------------------

def generate_figure(rows: list[dict], stats: list[dict]):
    """
    Two-panel figure:
      Left:  Polar scatter — true bearing (radius=constant) vs estimated bearing
      Right: Bar chart — mean absolute error ± std dev per angle, grouped by distance
    """
    distances = sorted(set(float(r["distance_m"]) for r in rows))
    colors    = ["#2196F3", "#FF5722"]  # blue for near, orange for far
    dist_color = dict(zip(distances, colors))

    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ---- Left panel: polar scatter ----------------------------------------
    ax_polar = fig.add_subplot(gs[0], projection="polar")
    ax_polar.set_theta_zero_location("N")   # 0° at top (compass convention)
    ax_polar.set_theta_direction(-1)         # clockwise

    radii_true = {"10": 0.65, "30": 0.85}   # concentric rings per distance

    for dist in distances:
        dist_rows = [r for r in rows if float(r["distance_m"]) == dist]
        ring = radii_true.get(str(int(dist)), 0.75)

        # Estimated bearings
        est_angles = np.deg2rad([float(r["estimated_bearing_deg"]) for r in dist_rows])
        ax_polar.scatter(
            est_angles,
            [ring] * len(est_angles),
            color=dist_color[dist],
            alpha=0.55,
            s=35,
            label=f"{dist:.0f} m estimated",
        )

        # True bearings (small solid markers on same ring)
        true_angles = np.deg2rad(sorted(set(float(r["true_bearing_deg"]) for r in dist_rows)))
        ax_polar.scatter(
            true_angles,
            [ring] * len(true_angles),
            color=dist_color[dist],
            marker="|",
            s=120,
            linewidths=2,
            zorder=5,
            label=f"{dist:.0f} m true",
        )

    ax_polar.set_ylim(0, 1.0)
    ax_polar.set_yticks([0.65, 0.85])
    ax_polar.set_yticklabels(["10 m", "30 m"], fontsize=8)
    ax_polar.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_polar.set_xticklabels(["0°\n(N)", "45°", "90°\n(E)", "135°", "180°\n(S)", "225°", "270°\n(W)", "315°"])
    ax_polar.set_title("Estimated vs. True Bearing\n(tick = true, dots = estimates)", pad=18, fontsize=10)
    ax_polar.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=8)

    # ---- Right panel: grouped error bar chart ----------------------------
    ax_bar = fig.add_subplot(gs[1])

    angles     = sorted(set(float(r["true_bearing_deg"]) for r in rows))
    n_angles   = len(angles)
    n_dists    = len(distances)
    bar_width  = 0.35
    x          = np.arange(n_angles)

    for i, dist in enumerate(distances):
        mae_vals = []
        sd_vals  = []
        for angle in angles:
            match = [s for s in stats
                     if s["distance_m"] == dist and s["true_bearing_deg"] == angle]
            if match:
                mae_vals.append(match[0]["mean_abs_error_deg"])
                sd_vals.append(match[0]["std_dev_deg"])
            else:
                mae_vals.append(0.0)
                sd_vals.append(0.0)

        offset = (i - (n_dists - 1) / 2) * bar_width
        ax_bar.bar(
            x + offset,
            mae_vals,
            bar_width,
            yerr=sd_vals,
            capsize=4,
            color=dist_color[dist],
            alpha=0.82,
            label=f"{dist:.0f} m",
        )

    ax_bar.set_xlabel("True Bearing (degrees)", fontsize=10)
    ax_bar.set_ylabel("Mean Absolute Error (degrees)", fontsize=10)
    ax_bar.set_title("GCC-PHAT Bearing Error by Angle and Distance\n(error bars = ±1 SD)", fontsize=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{int(a)}°" for a in angles])
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
    ax_bar.set_ylim(bottom=0)

    fig.suptitle(
        "GUARD System — Bearing Accuracy Experiment\n"
        f"ReSpeaker XVF3800 · GCC-PHAT · {SAMPLE_RATE/1000:.0f} kHz",
        fontsize=12, y=1.01,
    )

    plt.savefig(FIGURE_PNG, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {FIGURE_PNG}")
    plt.show()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(resume: bool):
    session = load_session(SESSION_LOG) if resume else {"completed": []}

    print("\n" + "=" * 60)
    print("GUARD — Bearing Accuracy Experiment")
    print("=" * 60)
    print(f"Protocol: {len(TARGET_ANGLES_DEG)} angles × "
          f"{TRIALS_PER_POS} trials × "
          f"{len(DISTANCES_M)} distances = "
          f"{len(TARGET_ANGLES_DEG) * TRIALS_PER_POS * len(DISTANCES_M)} total trials\n")

    print("Before you start:")
    print("  1. Mount the ReSpeaker XVF3800 on a stable tripod.")
    print("  2. Mark the 8 target positions on the ground (8 × 45°).")
    print("  3. Confirm RESPEAKER_DEVICE matches your system (run --list-devices).")
    input("\nPress Enter when ready...\n")

    for dist_m in DISTANCES_M:
        print("\n" + "─" * 60)
        print(f"DISTANCE CONDITION: {dist_m} m")
        print("─" * 60)

        wind_speed = _prompt_float(
            f"  Enter wind speed for this condition (m/s, e.g. 1.5): "
        )

        for angle_deg in TARGET_ANGLES_DEG:
            print(f"\n  TRUE BEARING: {angle_deg}° — Distance: {dist_m} m")
            print(f"  Place the acoustic source (drone / speaker) at {angle_deg}° "
                  f"from the array centre, {dist_m} m away.")
            input("  Press Enter when source is in position...")

            trial_num = 1
            while trial_num <= TRIALS_PER_POS:
                if already_done(session, dist_m, angle_deg, trial_num):
                    print(f"    Trial {trial_num}/{TRIALS_PER_POS} — skipped (already recorded)")
                    trial_num += 1
                    continue

                print(f"\n    Trial {trial_num}/{TRIALS_PER_POS}")
                print(f"    Capturing {CAPTURE_SECONDS} s of audio... (make sure the source is active)")

                estimates = capture_bearings(CAPTURE_SECONDS, verbose=True)

                if not estimates:
                    print("    WARNING: No bearing estimates captured (hardware issue?).")
                    retry = input("    Retry this trial? [Y/n]: ").strip().lower()
                    if retry != "n":
                        continue  # retry same trial_num

                est_bearing = median_bearing(estimates)
                if est_bearing is None:
                    print("    Could not compute bearing — skipping trial.")
                    continue

                err_signed = bearing_error(est_bearing, angle_deg)
                err_abs    = abs(err_signed)

                row = {
                    "timestamp":            datetime.now().isoformat(timespec="seconds"),
                    "distance_m":           dist_m,
                    "wind_speed_mps":       wind_speed,
                    "true_bearing_deg":     angle_deg,
                    "trial_num":            trial_num,
                    "estimated_bearing_deg": est_bearing,
                    "signed_error_deg":     err_signed,
                    "abs_error_deg":        err_abs,
                }

                append_raw(row)
                session["completed"].append([dist_m, angle_deg, trial_num])
                save_session(session, SESSION_LOG)

                n_est = len(estimates)
                print(f"    Estimated: {est_bearing:.1f}°  |  Error: {err_signed:+.1f}°  "
                      f"({n_est} readings, median taken)")
                trial_num += 1

    print("\n" + "=" * 60)
    print("All trials complete.")
    analyse()


def analyse():
    """Load raw CSV, compute statistics, print table, generate figure."""
    rows = load_raw()
    if not rows:
        print("No data found. Run the experiment first.")
        return

    print(f"\nLoaded {len(rows)} trial rows from {RAW_CSV}")
    stats = compute_statistics(rows)
    write_stats_csv(stats)
    print_stats_table(stats)
    generate_figure(rows, stats)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _prompt_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a number (e.g. 1.5)")


def list_devices():
    print("\nAvailable audio input devices:")
    print(sd.query_devices())
    print(f"\nCurrent RESPEAKER_DEVICE = {RESPEAKER_DEVICE}")
    print("Edit the RESPEAKER_DEVICE constant at the top of this file if needed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUARD bearing accuracy experiment")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from a partial session (reads bearing_session_log.json)")
    parser.add_argument("--analyse-only", action="store_true",
                        help="Skip capture, compute stats and figure from existing CSV")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print sounddevice audio input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    if args.analyse_only:
        analyse()
        sys.exit(0)

    run_experiment(resume=args.resume)
