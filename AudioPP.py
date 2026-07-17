import librosa
import numpy as np
from scipy.signal import butter, lfilter


class AudioPreprocessor:
    """
    Two-stage audio preprocessing pipeline for drone detection.

    Stage 1 — Pre-filter (noise cleaning, drop-in safe, no retraining needed):
      - Rolling noise profile built from silent windows via spectral subtraction
      - HPSS harmonic separation: keeps steady motor tones, discards wind/transients
      - Tightened bandpass 80-4000 Hz (drone fundamentals live here; cuts wind rumble
        below 80 Hz and high-freq hiss above 4 kHz that drowns weak distant signals)

    Stage 2 — Detection filter (existing chain, unchanged):
      - Bandpass -> mel spectrogram (128 bands, normalized to [0,1])

    The mel spectrogram fed to the CNN stays in the same format the model was
    trained on — only the waveform going in is cleaner.
    """

    # Number of silent frames to average for the noise profile.
    # At 16 kHz / 1-second windows, 10 frames = ~10 seconds of ambient buildup.
    _NOISE_PROFILE_FRAMES = 10

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self._noise_profile: np.ndarray | None = None   # magnitude spectrum of noise floor
        self._noise_buf: list[np.ndarray]       = []    # accumulator for silent frames

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_noise_profile(self, audio: np.ndarray) -> None:
        """
        Accumulate FFT magnitude of a silent frame into the rolling noise profile.
        Keeps the most recent _NOISE_PROFILE_FRAMES frames and averages them.
        """
        mag = np.abs(np.fft.rfft(audio))
        self._noise_buf.append(mag)
        if len(self._noise_buf) > self._NOISE_PROFILE_FRAMES:
            self._noise_buf.pop(0)
        self._noise_profile = np.mean(self._noise_buf, axis=0)

    def spectral_subtraction(self, audio: np.ndarray,
                             over_subtraction: float = 1.5) -> np.ndarray:
        """
        Subtract the stored noise floor estimate from the signal in the frequency
        domain.  over_subtraction > 1 removes more noise at the cost of slight
        musical-noise artifacts — 1.5 is a good field compromise.

        If no noise profile exists yet (first few seconds of runtime), the audio
        is returned unchanged so inference can still begin immediately.
        """
        if self._noise_profile is None:
            return audio

        S     = np.fft.rfft(audio)
        mag   = np.abs(S)
        phase = np.angle(S)

        # Subtract scaled noise floor; floor at 5% of original magnitude to
        # avoid total cancellation (which creates zero-input artifacts in the CNN).
        mag_clean = mag - over_subtraction * self._noise_profile
        mag_clean = np.maximum(mag_clean, 0.05 * mag)

        audio_clean = np.fft.irfft(mag_clean * np.exp(1j * phase), n=len(audio))
        return audio_clean.astype(audio.dtype)

    def hpss_harmonic(self, audio: np.ndarray, margin: float = 3.0) -> np.ndarray:
        """
        Harmonic-Percussive Source Separation — keep only the harmonic component.
        Drone motor tones are harmonic and steady; wind gusts and handling noise
        are percussive/transient and get discarded here.
        margin >= 3 gives aggressive separation for outdoor environments.
        """
        harmonic, _ = librosa.effects.hpss(audio, margin=margin)
        return harmonic.astype(audio.dtype)

    def bandpass_filter(self, audio: np.ndarray,
                        low_hz: float = 80.0,
                        high_hz: float = 4000.0) -> np.ndarray:
        """
        Bandpass 80-4000 Hz (tightened from original 50-7500 Hz).
        - 80 Hz low cut  : removes wind rumble and body handling noise
        - 4000 Hz high cut: removes high-freq hiss that drowns weak distant signals
        Drone motor fundamentals and blade-pass frequencies live within this band.
        Uses causal lfilter (not filtfilt) so it works on real-time windows.
        """
        nyq  = self.sr / 2
        low  = low_hz / nyq
        high = min(high_hz / nyq, 0.99)
        b, a = butter(5, [low, high], btype='band')
        return lfilter(b, a, audio)

    def is_silent(self, audio: np.ndarray, threshold: float = 1e-6) -> bool:
        """
        Returns True if the audio window has no usable signal energy.
        Call this before process() to skip inference on silent windows --
        the model outputs high confidence on all-zero tensors (BatchNorm
        bias terms do not collapse), so the gate must live at the call site.

        Silent frames are also used to build the noise profile automatically.
        """
        silent = np.max(np.abs(audio)) < threshold
        if silent:
            self._update_noise_profile(audio)
        return silent

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — mel spectrogram (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────

    def extract_mel_spectrogram(self, audio: np.ndarray,
                                n_mels: int = 128,
                                n_fft: int  = 1024,
                                hop_length: int = 256) -> np.ndarray:
        """
        Waveform -> 128-band mel spectrogram, normalized to [0, 1].
        n_fft=1024 / hop_length=256 tuned for 16 kHz input.
        fmin/fmax updated to match the tightened bandpass.
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=80,
            fmax=4000,
        )
        mel_db   = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_norm.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Full pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Original inference pipeline matching the training distribution:
          bandpass (80-4000 Hz) -> mel spectrogram (128 bands, normalized)

        Stage 1 (spectral subtraction + HPSS) is intentionally skipped here
        because the model was trained without it. Applying HPSS at inference
        time produces out-of-distribution spectrograms and pushes confidence
        toward ~50%, preventing reliable detection. Use process_with_denoising()
        only if the model is retrained with the same Stage 1 pipeline.
        """
        audio = self.bandpass_filter(audio)
        return self.extract_mel_spectrogram(audio)

    def process_with_denoising(self, audio: np.ndarray) -> np.ndarray:
        """
        Full two-stage chain including Stage 1 denoising.
        Only use this if the model was trained with the same pipeline.
          Stage 1: spectral subtraction -> HPSS harmonic -> bandpass (80-4000 Hz)
          Stage 2: mel spectrogram (128 bands, normalized)
        """
        audio = self.spectral_subtraction(audio)
        audio = self.hpss_harmonic(audio)
        audio = self.bandpass_filter(audio)
        return self.extract_mel_spectrogram(audio)

    def reset_noise_profile(self) -> None:
        """Clear the accumulated noise profile (call if environment changes)."""
        self._noise_profile = None
        self._noise_buf.clear()

    def __repr__(self) -> str:
        has_profile = self._noise_profile is not None
        return (f"AudioPreprocessor(sr={self.sr}, "
                f"stage1=[spectral_sub, hpss, bandpass=80-4000Hz], "
                f"stage2=[mel128], "
                f"noise_profile={'ready' if has_profile else 'building'})")
