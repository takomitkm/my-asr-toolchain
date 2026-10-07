#!/usr/bin/env python3
"""
FireRedVAD ONNX inference  —  no PyTorch / kaldi_native_fbank required.

Dependencies: numpy, scipy, soundfile, onnxruntime

Usage:

    python infer_onnx.py assets/hello_zh.wav --model_dir /path/to/FireRedVAD_onnx
    python infer_onnx.py assets/hello_zh.wav
    python infer_onnx.py assets/hello_zh.wav --model_dir /path/to/FireRedVAD_onnx
    python infer_onnx.py assets/hello_en.wav --speech_threshold 0.4 --min_speech_frame 20
"""

import argparse
import json
import math
import os
from collections import deque

import numpy as np
import soundfile as sf
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Constants (matches fireredvad/core/constants.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
FRAME_LENGTH_S = 0.025
FRAME_SHIFT_S = 0.010


# ---------------------------------------------------------------------------
# Kaldi-compatible Fbank (replaces kaldi_native_fbank dependency)
# ---------------------------------------------------------------------------

def _mel_to_hz(mel):
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)

def _hz_to_mel(hz):
    return 1127.0 * np.log(1.0 + hz / 700.0)

def _build_mel_filterbank(n_fft: int, n_mels: int = 80,
                          f_min: float = 0.0, f_max: float = 8000.0,
                          sample_rate: int = 16000) -> np.ndarray:
    """
    Build Kaldi-style mel filterbank matrix.
    Returns shape (n_mels, n_fft // 2 + 1) — real spectrum bins.
    Kaldi uses triangular filters defined on the mel scale, NOT on the FFT bin scale.
    """
    n_freqs = n_fft // 2 + 1
    freq_bins = np.linspace(0, sample_rate / 2, n_freqs)  # Hz for each FFT bin

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    # n_mels + 2 points: left edge, n_mels centers, right edge
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # For each mel band, compute the triangular weight for each FFT bin
    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left   = hz_points[m - 1]
        center = hz_points[m]
        right  = hz_points[m + 1]
        for k, f in enumerate(freq_bins):
            if left <= f <= center:
                filters[m - 1, k] = (f - left) / (center - left)
            elif center < f <= right:
                filters[m - 1, k] = (right - f) / (right - center)
    return filters


def _extract_kaldi_fbank(wav_int16: np.ndarray,
                         sample_rate: int = 16000,
                         num_mel_bins: int = 80,
                         frame_length_ms: float = 25.0,
                         frame_shift_ms: float = 10.0) -> np.ndarray:
    """
    Compute log Mel filterbank features identical to kaldi_native_fbank with:
        samp_freq=16000, frame_length_ms=25, frame_shift_ms=10,
        dither=0, snip_edges=True, num_mel_bins=80

    Input : int16 PCM array (1-D)
    Output: (T, 80) float32 array  — log fbank energies
    """
    assert wav_int16.dtype == np.int16 and wav_int16.ndim == 1

    frame_len  = int(round(sample_rate * frame_length_ms / 1000))  # 400
    frame_shift = int(round(sample_rate * frame_shift_ms  / 1000)) # 160
    n_fft = 1 << (frame_len - 1).bit_length()  # next power of 2 >= 400  → 512

    # Hann window  (Kaldi uses the same)
    window = np.hanning(frame_len).astype(np.float32)

    # Convert to float, Kaldi: raw int16 → float32 (NOT divided by 32768 at fbank stage)
    # Actually Kaldi processes the waveform as-is (int16 values as float).
    wav_f = wav_int16.astype(np.float32)

    # Pre-emphasis  (Kaldi default: 0.97)
    wav_preemph = np.append(wav_f[0], wav_f[1:] - 0.97 * wav_f[:-1])

    # Framing (snip_edges=True: only frames that fit completely)
    n_frames = 1 + (len(wav_preemph) - frame_len) // frame_shift
    if n_frames <= 0:
        return np.zeros((0, num_mel_bins), dtype=np.float32)

    frames = np.stack([
        wav_preemph[i * frame_shift: i * frame_shift + frame_len] * window
        for i in range(n_frames)
    ])  # (T, frame_len)

    # FFT → power spectrum
    spec = np.fft.rfft(frames, n=n_fft)           # (T, n_fft//2+1) complex
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)  # (T, n_fft//2+1)

    # Mel filterbank
    mel_fb = _build_mel_filterbank(n_fft, num_mel_bins,
                                   f_min=0.0, f_max=sample_rate / 2,
                                   sample_rate=sample_rate)  # (80, n_fft//2+1)
    mel_energy = power @ mel_fb.T  # (T, 80)

    # Log compression — floor at 1.0 (Kaldi: log_energy_floor = FLT_MIN, effectively 0)
    mel_energy = np.maximum(mel_energy, 1.0)
    log_mel = np.log(mel_energy).astype(np.float32)

    return log_mel  # (T, 80)


# ---------------------------------------------------------------------------
# CMVN (matches fireredvad/core/audio_feat.py  CMVN.__call__)
# ---------------------------------------------------------------------------

class CMVN:
    def __init__(self, cmvn_json_path: str):
        with open(cmvn_json_path) as f:
            d = json.load(f)
        self.means = np.array(d["means"], dtype=np.float32)
        self.inv_std = np.array(d["inverse_std_variances"], dtype=np.float32)

    def __call__(self, fbank: np.ndarray) -> np.ndarray:
        return (fbank - self.means) * self.inv_std


# ---------------------------------------------------------------------------
# VAD postprocessor (pure Python, mirrors vad_postprocessor.py)
# ---------------------------------------------------------------------------

class VadPostprocessor:
    def __init__(self, smooth_window_size=5, prob_threshold=0.4,
                 min_speech_frame=20, max_speech_frame=2000,
                 min_silence_frame=20, merge_silence_frame=0,
                 extend_speech_frame=0):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = prob_threshold
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame

    def process(self, raw_probs):
        if not raw_probs:
            return []
        smoothed = self._smooth(raw_probs)
        binary = (np.asarray(smoothed) >= self.prob_threshold).astype(int).tolist()
        decisions = self._state_machine(binary)
        decisions = self._fix_start(decisions)
        decisions = self._merge_silence(decisions)
        decisions = self._extend_speech(decisions)
        decisions = self._split_long(decisions, raw_probs)
        return decisions

    def decisions_to_segments(self, decisions, wav_dur=None):
        segments = []
        speech_start = None
        for t, d in enumerate(decisions):
            if d == 1 and speech_start is None:
                speech_start = t
            elif d == 0 and speech_start is not None:
                segments.append((speech_start * FRAME_SHIFT_S, t * FRAME_SHIFT_S))
                speech_start = None
        if speech_start is not None:
            end = len(decisions) * FRAME_SHIFT_S + FRAME_LENGTH_S
            if wav_dur is not None:
                end = min(end, wav_dur)
            segments.append((speech_start * FRAME_SHIFT_S, end))
        return [(round(s, 3), round(e, 3)) for s, e in segments]

    def _smooth(self, probs):
        if self.smooth_window_size <= 1:
            return probs
        probs_np = np.array(probs)
        kernel = np.ones(self.smooth_window_size) / self.smooth_window_size
        smoothed = np.convolve(probs_np, kernel, mode='full')[:len(probs)]
        for i in range(min(self.smooth_window_size - 1, len(probs))):
            smoothed[i] = np.mean(probs_np[:i + 1])
        return smoothed

    def _state_machine(self, binary):
        SILENCE, POSSIBLE_SPEECH, SPEECH, POSSIBLE_SILENCE = 0, 1, 2, 3
        decisions = [0] * len(binary)
        state = SILENCE
        speech_start = silence_start = -1
        for t, is_speech in enumerate(binary):
            if state == SILENCE:
                if is_speech:
                    state = POSSIBLE_SPEECH; speech_start = t
            elif state == POSSIBLE_SPEECH:
                if is_speech:
                    if t - speech_start >= self.min_speech_frame:
                        state = SPEECH
                        decisions[speech_start:t] = [1] * (t - speech_start)
                else:
                    state = SILENCE; speech_start = -1
            elif state == SPEECH:
                if not is_speech:
                    state = POSSIBLE_SILENCE; silence_start = t
            elif state == POSSIBLE_SILENCE:
                if not is_speech:
                    if t - silence_start >= self.min_silence_frame:
                        state = SILENCE; speech_start = -1
                else:
                    state = SPEECH; silence_start = -1
            decisions[t] = 1 if state in (SPEECH, POSSIBLE_SILENCE) else 0
        return decisions

    def _fix_start(self, decisions):
        new = decisions.copy()
        for t, d in enumerate(decisions):
            if t > 0 and decisions[t - 1] == 0 and d == 1:
                start = max(0, t - self.smooth_window_size)
                new[start:t] = [1] * (t - start)
        return new

    def _merge_silence(self, decisions):
        if self.merge_silence_frame <= 0:
            return decisions
        new = decisions.copy()
        silence_start = None
        for t, d in enumerate(decisions):
            if t > 0 and decisions[t - 1] == 1 and d == 0 and silence_start is None:
                silence_start = t
            elif t > 0 and decisions[t - 1] == 0 and d == 1 and silence_start is not None:
                if t - silence_start < self.merge_silence_frame:
                    new[silence_start:t] = [1] * (t - silence_start)
                silence_start = None
        return new

    def _extend_speech(self, decisions):
        if self.extend_speech_frame <= 0:
            return decisions
        d = np.array(decisions)
        k = np.ones(2 * self.extend_speech_frame + 1)
        return (np.convolve(d, k, mode='same') > 0).astype(int).tolist()

    def _split_long(self, decisions, probs):
        new = decisions.copy()
        segments = self.decisions_to_segments(decisions)
        for s_s, e_s in segments:
            sf_ = int(s_s / FRAME_SHIFT_S)
            ef_ = int(e_s / FRAME_SHIFT_S)
            if ef_ - sf_ > self.max_speech_frame:
                seg_probs = probs[sf_:ef_]
                for split in self._find_splits(seg_probs):
                    new[sf_ + split] = 0
        return new

    def _find_splits(self, probs):
        splits, L, start = [], len(probs), 0
        while start < L:
            if (L - start) <= self.max_speech_frame:
                break
            ws = int(start + self.max_speech_frame / 2)
            we = int(start + self.max_speech_frame)
            splits.append(ws + int(np.argmin(probs[ws:we])))
            start = splits[-1] + 1
        return splits


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class FireRedVadOnnx:
    """
    End-to-end FireRedVAD inference using ONNX Runtime.
    No PyTorch or kaldi_native_fbank required.
    """

    def __init__(self, model_dir: str,
                 smooth_window_size: int = 5,
                 speech_threshold: float = 0.4,
                 min_speech_frame: int = 20,
                 max_speech_frame: int = 2000,
                 min_silence_frame: int = 20,
                 merge_silence_frame: int = 0,
                 extend_speech_frame: int = 0,
                 chunk_max_frame: int = 30000,
                 use_coreml: bool = False):

        # CMVN params
        # Support both naming conventions: cmvn.json and cmvn_params.json
        for name in ("cmvn.json", "cmvn_params.json"):
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                self.cmvn = CMVN(p)
                break
        else:
            raise FileNotFoundError(f"No cmvn JSON found in {model_dir}")

        # ONNX session
        model_path = os.path.join(model_dir, "model.onnx")
        providers = ["CPUExecutionProvider"]
        if use_coreml:
            providers = ["CoreMLExecutionProvider"] + providers
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=providers)

        # Fbank config (fixed to match training)
        self.num_mel_bins = 80
        self.chunk_max_frame = chunk_max_frame

        # Postprocessor
        self.postprocessor = VadPostprocessor(
            smooth_window_size=smooth_window_size,
            prob_threshold=speech_threshold,
            min_speech_frame=min_speech_frame,
            max_speech_frame=max_speech_frame,
            min_silence_frame=min_silence_frame,
            merge_silence_frame=merge_silence_frame,
            extend_speech_frame=extend_speech_frame)

    def _load_audio(self, audio_path: str):
        wav, sr = sf.read(audio_path, dtype="int16")
        assert sr == SAMPLE_RATE, f"Expected 16kHz, got {sr}Hz. Convert with ffmpeg first."
        assert wav.ndim == 1, "Expected mono audio."
        return wav

    def _extract_features(self, wav_int16: np.ndarray) -> np.ndarray:
        """Returns (T, 80) CMVN-normalized log-fbank."""
        fbank = _extract_kaldi_fbank(wav_int16, SAMPLE_RATE, self.num_mel_bins)
        return self.cmvn(fbank)

    def _run_model(self, feat: np.ndarray) -> np.ndarray:
        """
        feat: (T, 80)
        Returns probs: (T,) float32
        """
        T = feat.shape[0]
        all_probs = []

        for chunk_start in range(0, T, self.chunk_max_frame):
            chunk = feat[chunk_start: chunk_start + self.chunk_max_frame]
            inp = chunk[np.newaxis, :, :].astype(np.float32)  # (1, t, 80)

            outputs = self.session.run(["output"], {"input": inp})
            probs_chunk = outputs[0][0, :, 0]  # (t,)
            all_probs.append(probs_chunk)

        return np.concatenate(all_probs)  # (T,)

    def detect(self, audio_path: str):
        """
        Run VAD on a 16kHz mono WAV file.

        Returns:
            result : dict with keys 'dur', 'timestamps', 'wav_path'
            probs  : (T,) float32 array of raw per-frame speech probabilities
        """
        wav = self._load_audio(audio_path)
        dur = len(wav) / SAMPLE_RATE

        feat = self._extract_features(wav)          # (T, 80)
        probs = self._run_model(feat)               # (T,)

        decisions = self.postprocessor.process(probs.tolist())
        segments  = self.postprocessor.decisions_to_segments(decisions, dur)

        result = {
            "dur": round(dur, 3),
            "timestamps": segments,
            "wav_path": audio_path,
        }
        return result, probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FireRedVAD ONNX inference")
    parser.add_argument("wav_path", help="Path to 16kHz mono WAV file")
    parser.add_argument("--model_dir",
                        default="/path/to/FireRedVAD_onnx",
                        help="Directory containing model.onnx and cmvn.json")
    parser.add_argument("--smooth_window_size", type=int, default=5)
    parser.add_argument("--speech_threshold", type=float, default=0.4)
    parser.add_argument("--min_speech_frame", type=int, default=20)
    parser.add_argument("--max_speech_frame", type=int, default=2000)
    parser.add_argument("--min_silence_frame", type=int, default=20)
    parser.add_argument("--merge_silence_frame", type=int, default=0)
    parser.add_argument("--extend_speech_frame", type=int, default=0)
    parser.add_argument("--use_coreml", action="store_true",
                        help="Use CoreML execution provider (Apple Silicon)")
    args = parser.parse_args()

    vad = FireRedVadOnnx(
        model_dir=args.model_dir,
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        min_speech_frame=args.min_speech_frame,
        max_speech_frame=args.max_speech_frame,
        min_silence_frame=args.min_silence_frame,
        merge_silence_frame=args.merge_silence_frame,
        extend_speech_frame=args.extend_speech_frame,
        use_coreml=args.use_coreml,
    )

    import time
    start = time.time()
    result, probs = vad.detect(args.wav_path)

    du = time.time() - start
    print(f"vad : {du}s")
    print(f"Duration  : {result['dur']:.3f}s")
    print(f"Segments  : {len(result['timestamps'])}")
    for i, (s, e) in enumerate(result["timestamps"]):
        print(f"  [{i+1:3d}] {s:.3f}s -- {e:.3f}s  ({e-s:.3f}s)")


if __name__ == "__main__":
    main()

# python infer_onnx.py assets/hello_zh.wav --model_dir /path/to/FireRedVAD_onnx
