"""Signal processing for the reference clip: resampling, length and loudness.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The ONNX speech encoder takes the clip at 24 kHz and does the rest itself
(mel spectrograms, speech tokens, speaker vector). Before it, the add-on does
what upstream's prepare_conditionals does with librosa and pyloudnorm:

- clips over 10 seconds are cut, near a pause (upstream keeps the first 10 s
  for the decoder);
- clips under 6 seconds are looped with a short gap: Turbo babbles with less
  than 5 s of reference, which upstream refuses outright;
- for Turbo, the level goes to -27 LUFS (ITU-R BS.1770), as upstream's
  norm_loudness does.
"""

from __future__ import annotations

import math

import numpy as np

MAX_SECONDS = 10.0
MIN_SECONDS = 6.0
LOOP_GAP_SECONDS = 0.3
TARGET_LUFS = -27.0


def resample(wav: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Hann-windowed sinc with 16 taps a side, normalized by the weight sum."""
    wav = np.asarray(wav, dtype=np.float32)
    if source_rate == target_rate:
        return wav.copy()
    length = max(1, round(len(wav) * target_rate / source_rate))
    radius, ratio = 16, source_rate / target_rate
    out = np.empty(length, dtype=np.float32)
    taps = np.arange(-radius + 1, radius + 1)
    block = 8192
    for first in range(0, length, block):
        center = (np.arange(first, min(length, first + block)) + 0.5) * ratio - 0.5
        source = np.floor(center).astype(np.int64)[:, None] + taps[None, :]
        x = center[:, None] - source
        with np.errstate(invalid='ignore', divide='ignore'):
            sinc = np.where(np.abs(x) < 1e-8, 1.0, np.sin(np.pi * x) / (np.pi * x))
        weight = sinc * (0.5 + 0.5 * np.cos(np.pi * x / radius))
        valid = (source >= 0) & (source < len(wav))
        weight = np.where(valid, weight, 0.0)
        values = wav[np.clip(source, 0, len(wav) - 1)]
        total = weight.sum(axis=1)
        out[first:first + len(center)] = np.where(total != 0, (values * weight).sum(axis=1) / np.where(total != 0, total, 1), 0)
    return out


def _frame_rms(wav: np.ndarray, frame: int) -> np.ndarray:
    count = len(wav) // frame
    frames = np.asarray(wav[:count * frame], dtype=np.float64).reshape(count, frame)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


def cut(wav: np.ndarray, sample_rate: int, seconds: float = MAX_SECONDS) -> np.ndarray:
    """The first `seconds`, ended at the quietest 50 ms of the last 3 s."""
    limit = int(seconds * sample_rate)
    if len(wav) <= limit:
        return wav
    frame = int(0.05 * sample_rate)
    start = limit - int(3 * sample_rate)
    rms = _frame_rms(wav[start:limit], frame)
    end = start + (int(np.argmin(rms)) + 1) * frame if len(rms) else limit
    return wav[:end]


def loop(wav: np.ndarray, sample_rate: int, seconds: float = MIN_SECONDS) -> np.ndarray:
    """Repeat a short clip, with a short gap of silence, to at least `seconds`."""
    target = int(seconds * sample_rate)
    if len(wav) >= target or not len(wav):
        return wav
    gap = np.zeros(int(LOOP_GAP_SECONDS * sample_rate), dtype=np.float32)
    parts, total = [wav], len(wav)
    while total < target:
        parts += [gap, wav]
        total += len(gap) + len(wav)
    return np.concatenate(parts)


def _biquad_response(b, a, n: int) -> np.ndarray:
    """Frequency response of a biquad at rfft bins of an n-point FFT."""
    z = np.exp(-1j * np.pi * np.arange(n // 2 + 1) / (n // 2))
    return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)


def _k_weighting(sample_rate: int) -> tuple:
    """BS.1770's two K-weighting biquads at `sample_rate` (pyloudnorm's)."""
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    K = math.tan(math.pi * fc / sample_rate)
    Vh, Vb = 10 ** (G / 20), (10 ** (G / 20)) ** 0.499666774155
    a0 = 1 + K / Q + K * K
    shelf = ([(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0],
             [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])
    Q, fc = 0.5003270373253953, 38.13547087613982
    K = math.tan(math.pi * fc / sample_rate)
    a0 = 1 + K / Q + K * K
    high_pass = ([1.0, -2.0, 1.0], [1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0])
    return shelf, high_pass


def loudness(wav: np.ndarray, sample_rate: int) -> float:
    """Integrated loudness in LUFS (mono), -inf for silence or a short clip."""
    block, hop = int(0.4 * sample_rate), int(0.1 * sample_rate)
    if len(wav) < block:
        return -math.inf
    # Filtered in the frequency domain, padded so the filters' tails (well
    # under a second at these frequencies) do not wrap around.
    n = 1 << int(math.ceil(math.log2(len(wav) + sample_rate)))
    spectrum = np.fft.rfft(np.asarray(wav, dtype=np.float64), n)
    for b, a in _k_weighting(sample_rate):
        spectrum *= _biquad_response(b, a, n)
    weighted = np.fft.irfft(spectrum, n)[:len(wav)]
    starts = np.arange(0, len(weighted) - block + 1, hop)
    cumsum = np.concatenate(([0.0], np.cumsum(weighted ** 2)))
    power = (cumsum[starts + block] - cumsum[starts]) / block
    with np.errstate(divide='ignore'):
        levels = -0.691 + 10 * np.log10(power)
    gated = power[levels > -70]
    if not len(gated):
        return -math.inf
    relative = -0.691 + 10 * np.log10(gated.mean()) - 10
    gated = power[levels > max(-70, relative)]
    return float(-0.691 + 10 * np.log10(gated.mean())) if len(gated) else -math.inf


def normalize_loudness(wav: np.ndarray, sample_rate: int, target: float = TARGET_LUFS) -> np.ndarray:
    level = loudness(wav, sample_rate)
    if not math.isfinite(level):
        return wav
    return (wav * 10 ** ((target - level) / 20)).astype(np.float32)
