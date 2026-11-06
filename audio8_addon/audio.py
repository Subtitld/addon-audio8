"""Read and write PCM WAV with the standard library.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Subtitld's own reference clips are 16 kHz mono PCM WAV, and so are the preset
voices. Anything else is converted with ffmpeg first (see __main__).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class UnreadableAudio(Exception):
    """Not a PCM WAV file this reader takes."""


def read_wav(path) -> tuple[np.ndarray, int]:
    """(mono samples as float32 in [-1, 1], sample rate)."""
    try:
        with wave.open(str(path), 'rb') as clip:
            width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
            data = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise UnreadableAudio(f'{path}: {exc}') from exc
    if width == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = (np.where(value >= 1 << 23, value - (1 << 24), value) / float(1 << 23)).astype(np.float32)
    elif width == 4:
        samples = (np.frombuffer(data, dtype='<i4').astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        raise UnreadableAudio(f'{path}: {8 * width}-bit samples')
    if not len(samples):
        raise UnreadableAudio(f'{path}: no audio')
    return samples.reshape(-1, channels).mean(axis=1), rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    """16-bit mono PCM, written beside `path` and renamed into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype('<i2')
    with wave.open(str(part), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    part.replace(path)
