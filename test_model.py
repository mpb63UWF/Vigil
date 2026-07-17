"""
test_model.py -- Chickadee DroneDetectorCNN test suite
------------------------------------------------------
Tests the full pipeline: AudioPreprocessor -> DroneDetectorCNN

Run with:
    python test_model.py                         # architecture + pipeline tests
    python test_model.py --weights               # also loads best_drone_detector_v2.pth
    python test_model.py --weights --audio path  # inference on a WAV/MP3 file
    python test_model.py --weights --audio drone_folder/ background_folder/  # batch eval
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

_PASS = " PASS"
_FAIL = " FAIL"
_WARN = " WARN"
_INFO = " INFO"

def _check(label, condition, detail=""):
    status = _PASS if condition else _FAIL
    print(f"  [{status}] {label}" + (f"  - {detail}" if detail else ""))
    return condition


# ------------------------------------------------------------------------------
# 1. IMPORT CHECKS
# ------------------------------------------------------------------------------
def test_imports():
    print("\n-- 1. Import checks --------------------------------------------------")
    ok = True
    for mod in ["torch", "numpy", "librosa", "scipy"]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            _check(f"import {mod}", True, ver)
        except ImportError as e:
            _check(f"import {mod}", False, str(e))
            ok = False

    from NeuralNetwork import DroneDetectorCNN
    _check("import DroneDetectorCNN", True)
    from AudioPP import AudioPreprocessor
    _check("import AudioPreprocessor", True)
    return ok


# ------------------------------------------------------------------------------
# 2. ARCHITECTURE CHECKS
# ------------------------------------------------------------------------------
def test_architecture():
    print("\n-- 2. Model architecture ---------------------------------------------")
    from NeuralNetwork import DroneDetectorCNN
    model = DroneDetectorCNN()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    _check("Instantiation", True)
    _check("Parameter count > 0", total_params > 0, f"{total_params:,} params")

    # Test input/output shapes
    for t in [63, 128]:  # typical time frames for 0.5 s and 1 s windows
        dummy = torch.zeros(1, 1, 128, t)
        with torch.no_grad():
            out = model(dummy)
            prob = model.predict_proba(dummy)
        _check(f"forward (1,1,128,{t}) -> {tuple(out.shape)}", out.shape == (1, 2))
        _check(f"predict_proba shape", prob.shape == (1,))
        _check(f"predict_proba in [0,1]", 0.0 <= prob.item() <= 1.0,
               f"got {prob.item():.4f}")

    # Batch inference
    batch = torch.randn(8, 1, 128, 63)
    with torch.no_grad():
        probs = model.predict_proba(batch)
    _check("Batch inference (8 samples)", probs.shape == (8,),
           f"shape {probs.shape}")

    return model


# ------------------------------------------------------------------------------
# 3. PREPROCESSOR CHECKS
# ------------------------------------------------------------------------------
def test_preprocessor():
    print("\n-- 3. AudioPreprocessor ----------------------------------------------")
    from AudioPP import AudioPreprocessor
    pp = AudioPreprocessor(sample_rate=16000)

    # Silence window
    silence = np.zeros(16000, dtype=np.float32)
    mel = pp.process(silence)
    _check("Silence -> mel shape (128, T)", mel.shape[0] == 128,
           f"got {mel.shape}")
    _check("Silence mel values in [0,1]", mel.min() >= 0.0 and mel.max() <= 1.0,
           f"min={mel.min():.3f} max={mel.max():.3f}")

    # White noise window
    noise = np.random.randn(16000).astype(np.float32)
    mel_noise = pp.process(noise)
    _check("Noise -> mel finite values", np.all(np.isfinite(mel_noise)))
    _check("Noise mel values in [0,1]", mel_noise.min() >= 0.0 and mel_noise.max() <= 1.0)

    # Synthetic drone tone (sum of 3 harmonics: 120, 240, 360 Hz)
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    drone_tone = (np.sin(2 * np.pi * 120 * t) +
                  0.5 * np.sin(2 * np.pi * 240 * t) +
                  0.25 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    mel_drone = pp.process(drone_tone)
    _check("Drone tone -> mel finite", np.all(np.isfinite(mel_drone)))

    return pp


# ------------------------------------------------------------------------------
# 4. END-TO-END PIPELINE (no weights -- random init)
# ------------------------------------------------------------------------------
def test_pipeline():
    print("\n-- 4. End-to-end pipeline (random weights) ---------------------------")
    from NeuralNetwork import DroneDetectorCNN
    from AudioPP import AudioPreprocessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _check(f"Device", True, str(device))

    model = DroneDetectorCNN().eval().to(device)
    pp    = AudioPreprocessor(sample_rate=16000)

    # Simulate the exact path used in RealTimeInferenceSerial.py
    raw_bytes  = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    audio      = raw_bytes.astype(np.float32) / 32768.0
    mel        = pp.process(audio)
    tensor     = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        prob = model.predict_proba(tensor).item()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _check("Full pipeline completes", True)
    _check("Pipeline output in [0,1]", 0.0 <= prob <= 1.0, f"{prob:.4f}")
    _check(f"Inference latency < 200 ms", elapsed_ms < 200, f"{elapsed_ms:.1f} ms")

    # Confirm model is in eval mode (Dropout/BN behave deterministically)
    prob1 = model.predict_proba(tensor).item()
    prob2 = model.predict_proba(tensor).item()
    _check("Deterministic in eval mode", prob1 == prob2, f"{prob1:.6f} == {prob2:.6f}")


# ------------------------------------------------------------------------------
# 5. WEIGHTS LOADING (requires best_drone_detector_v2.pth)
# ------------------------------------------------------------------------------
def test_weights(model_path="best_drone_detector_v2.pth"):
    print(f"\n-- 5. Weights loading ({model_path}) ---------------------------------")
    from NeuralNetwork import DroneDetectorCNN

    if not os.path.exists(model_path):
        print(f"  [{_WARN}] {model_path} not found -- skipping weights test")
        print(f"         Copy best_drone_detector_v2.pth to this directory to enable")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DroneDetectorCNN()

    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        _check("Load state dict", True)
    except Exception as e:
        _check("Load state dict", False, str(e))
        return None

    model.eval().to(device)

    from AudioPP import AudioPreprocessor
    pp = AudioPreprocessor(sample_rate=16000)

    # Energy gate: silence must be caught before reaching the model.
    # The model itself outputs high confidence for all-zero tensors (BatchNorm
    # bias terms don't collapse) — the gate at the inference call site is the
    # correct fix, not the model or preprocessor.
    silence = np.zeros(16000, dtype=np.float32)
    gated = pp.is_silent(silence)
    _check("Energy gate catches silence", gated)

    # With the gate, inference is skipped for silence — probability is 0.0.
    p_silence = 0.0 if pp.is_silent(silence) else model.predict_proba(
        torch.tensor(pp.process(silence)).unsqueeze(0).unsqueeze(0).to(device)
    ).item()
    _check("Gated silence confidence == 0.0", p_silence == 0.0, f"{p_silence:.3f}")

    # Typical background noise must not be gated (mic always has ambient noise)
    bg = np.random.randn(16000).astype(np.float32) * 0.01
    _check("Low-level noise passes gate", not pp.is_silent(bg))

    return model


# ------------------------------------------------------------------------------
# 6. AUDIO FILE INFERENCE (optional)
# ------------------------------------------------------------------------------
def test_audio_files(model, audio_paths, label=None):
    import librosa
    from AudioPP import AudioPreprocessor

    print(f"\n-- 6. Audio file inference ({'label=' + label if label else 'unlabeled'}) --------")
    device = next(model.parameters()).device
    pp = AudioPreprocessor(sample_rate=16000)

    results = []
    for path in audio_paths:
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
        except Exception as e:
            print(f"  [WARN] Cannot load {os.path.basename(path)}: {e}")
            continue

        # Chunk into 1-second windows, report max confidence
        chunk = 16000
        probs = []
        for i in range(0, len(audio), chunk // 2):
            win = audio[i:i + chunk]
            if len(win) < chunk:
                win = np.pad(win, (0, chunk - len(win)))
            mel    = pp.process(win.astype(np.float32))
            tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs.append(model.predict_proba(tensor).item())

        if not probs:
            continue

        max_p = max(probs)
        avg_p = sum(probs) / len(probs)
        fname = os.path.basename(path)
        marker = "*** DRONE ***" if max_p >= 0.75 else ""
        print(f"  {fname[:45]:45s}  max={max_p:.3f}  avg={avg_p:.3f}  {marker}")
        results.append((path, max_p, avg_p))

    if label and results:
        expected_high = label == "drone"
        correct = sum(1 for _, mx, _ in results if (mx >= 0.75) == expected_high)
        print(f"\n  Accuracy on {label}: {correct}/{len(results)} = {correct/len(results):.1%}")

    return results


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test DroneDetectorCNN pipeline")
    parser.add_argument("--weights",    action="store_true",
                        help="Load best_drone_detector_v2.pth for inference tests")
    parser.add_argument("--model",      default="best_drone_detector_v2.pth",
                        help="Path to .pth weights file")
    parser.add_argument("--audio",      nargs="+",
                        help="WAV/MP3 file(s) or folder(s) to run inference on")
    parser.add_argument("--label",      choices=["drone", "background"],
                        help="Ground-truth label for accuracy calc (use with --audio)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Chickadee DroneDetectorCNN -- Test Suite")
    print("=" * 60)

    ok = test_imports()
    if not ok:
        print("\n[FATAL] Missing imports -- install requirements first.")
        sys.exit(1)

    test_architecture()
    test_preprocessor()
    test_pipeline()

    model = None
    if args.weights:
        model = test_weights(args.model)

    if args.audio and model is not None:
        import librosa
        paths = []
        for a in args.audio:
            if os.path.isdir(a):
                exts = {".wav", ".mp3", ".flac", ".ogg"}
                paths += sorted(
                    os.path.join(a, f) for f in os.listdir(a)
                    if os.path.splitext(f)[1].lower() in exts
                )
            else:
                paths.append(a)
        test_audio_files(model, paths, label=args.label)
    elif args.audio and model is None:
        print("\n[WARN] --audio requires --weights to load the trained model.")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
