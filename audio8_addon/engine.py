"""Audio8 TTS 0.6B on ONNX Runtime: register a voice, speak a line.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The graphs
----------
Audio8's official ONNX release of Preview 0.6B (Apache-2.0) splits the model
in four graphs, and the export of Realistic 0.6B the add-on also offers has
the same four, the same layout and the same inputs and outputs (its codec
and tokenizer are Preview's files). The loop between them is a port of
upstream's onnx_runtime/arktts_runtime (runtime.py and registration.py,
Apache-2.0):

- codec_encoder (FP16, 414 MB): a 44.1 kHz reference clip → its codec codes,
  10 codebooks at 21.5 frames a second. Only loaded to register a clip; the
  codes are cached on disk by the clip's sha256, so a speaker's clip is
  encoded once.
- slow_ar (INT4 weights): a Qwen-style transformer, 24 layers, over the
  prompt (text tokens and the reference's codes) → per frame, the semantic
  token (the first codebook) and a hidden state. KV cache of 2,048 positions.
- fast_ar (INT4 weights): 4 layers → the frame's other nine codebooks, one by
  one, from the slow hidden state.
- codec_decoder (FP16, 260 MB): a line's codes → 44.1 kHz audio, in one run.

Speed (i7-13700T, 8 threads, a quiet machine): about 45 ms of the two
transformers and 37 ms of the decoder per 46 ms frame, so about 0.4 to 0.5
times real time. The decoder attends to the whole line, so decoding while
the transformers still run would mean decoding the past again for every
chunk, which costs more than it saves on lines of subtitle length.

Sampling
--------
Upstream's: temperature 0.7, top-p 0.9, top-k 50, and a semantic token that
repeats one of the last ten is drawn again at temperature 1.0. The add-on
adds a cap on a line's frames from the length of its text, so a line that
runs on stops instead of filling the 2,048-position context.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from audio8_addon import dsp
from audio8_addon.text import Tokenizer, clean_text

log = logging.getLogger('audio8_addon.engine')

SAMPLE_RATE = 44100
FRAME_SAMPLES = 2048
FRAMES_PER_SECOND = SAMPLE_RATE / FRAME_SAMPLES   # about 21.5
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
RECENT = 10                  # semantic tokens a new one must not repeat
MIN_REFERENCE_SECONDS = 0.5  # upstream's registration limits
MAX_REFERENCE_SECONDS = 30.0

# The files the add-on reads, as paths in the pinned repository.
ONLINE = ('runtime_manifest.json', 'tokenizer/tokenizer.json',
          'slow_ar_int4.onnx', 'slow_ar_int4.onnx.data',
          'fast_ar_int4.onnx', 'fast_ar_int4.onnx.data',
          'codec_decoder_fp16.onnx', 'codec_decoder_fp16.onnx.data')
REGISTRATION = ('registration/registration_manifest.json',
                'registration/codec_encoder_fp16.onnx', 'registration/codec_encoder_fp16.onnx.data')
FILES = ONLINE + REGISTRATION

_ORT_DTYPES = {'tensor(float)': np.float32, 'tensor(float16)': np.float16,
               'tensor(int64)': np.int64, 'tensor(bool)': np.bool_}

# Code points of scripts written without spaces, which take longer to say
# per character (see expected_frames).
_DENSE = ((0x1100, 0x11FF), (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
          (0xAC00, 0xD7A3), (0xF900, 0xFAFF), (0x20000, 0x2FA1F))


class Cancelled(Exception):
    """The host cancelled the line."""


@dataclass(frozen=True)
class Voice:
    codes: np.ndarray        # (10, frames) int64, the reference's codec codes
    transcript: str          # what the reference says; '' to clone from sound only


def default_threads() -> int:
    """Half the logical processors, at most 8: past that it got no faster."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _session(path: Path, threads: int, run_once: bool = False):
    """A CPU session, fully optimized: the two transformers need the full
    level (at EXTENDED they never stop a line).

    run_once is for the encoder, run once per speaker. Its full optimization
    makes FP32 copies of its 414 MB of FP16 weights (1.4 GB loaded, 1.8 GB
    at its peak on a 10 s clip); at EXTENDED the weights stay memory-mapped
    (1.26 GB at its peak) for codes that are 98 to 100% the same and clones
    as close. Without the arena, its working memory goes back to the
    system once it is released.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if run_once
                                        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    options.enable_cpu_mem_arena = not run_once
    options.enable_mem_pattern = not run_once
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def _sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int,
            rng: np.random.Generator) -> int:
    """Upstream's sampler: top-k and top-p, then the Gumbel-max trick."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1]
    base = np.exp(values[order] - values[order[0]])
    base /= base.sum()
    remove = (np.cumsum(base) > top_p) | (np.arange(base.size) >= top_k)
    remove[0] = False
    masked = values.copy()
    masked[order[remove]] = -np.inf
    scaled = masked / max(temperature, 1e-5)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= probs.sum()
    noise = -np.log(np.clip(rng.random(probs.size), 1e-12, 1.0))
    return int(np.argmax(probs / noise))


def expected_frames(text: str) -> float:
    """About how many frames a line takes to say: 14 characters a second for
    alphabetic scripts, 4.5 for Chinese, Japanese and Korean."""
    seconds = 0.0
    for char in clean_text(text):
        code = ord(char)
        seconds += 1 / 4.5 if any(lo <= code <= hi for lo, hi in _DENSE) else 1 / 14
    return max(1.0, seconds * FRAMES_PER_SECOND)


def max_frames(text: str) -> int:
    """The cap on a line's frames: three times the expected length, and a
    second and a half more, so slow speakers and pauses still fit."""
    return int(3 * expected_frames(text) + 1.5 * FRAMES_PER_SECOND)


class Audio8:
    """The loaded model: the online graphs, the tokenizer, the voice cache."""

    def __init__(self, files: dict[str, Path], threads: int, voices_dir: Path):
        self.files = files
        self.threads = threads
        self.voices_dir = Path(voices_dir)
        self.manifest = json.loads(files['runtime_manifest.json'].read_text(encoding='utf-8'))
        if self.manifest.get('slow_logits_layout') != 'semantic_then_eos':
            raise ValueError('unexpected runtime manifest: slow_logits_layout')
        self.fingerprint = self.manifest['model_fingerprint']
        self.tokenizer = Tokenizer(files['tokenizer/tokenizer.json'])
        self.slow = _session(files['slow_ar_int4.onnx'], threads)
        self.fast = _session(files['fast_ar_int4.onnx'], threads)
        self.decoder = _session(files['codec_decoder_fp16.onnx'], threads)
        m = self.manifest
        self.layers, self.fast_layers = int(m['num_layers']), int(m['num_fast_layers'])
        self.codebooks, self.codebook_size = int(m['num_codebooks']), int(m['codebook_size'])
        self.max_seq_len = int(m['max_seq_len'])
        self.semantic_begin, self.semantic_end = int(m['semantic_begin_id']), int(m['semantic_end_id'])
        self.stop = int(m['im_end_id'])
        self.allowed = np.concatenate([np.arange(self.semantic_begin, self.semantic_end + 1), [self.stop]])
        slow_inputs = {i.name: i for i in self.slow.get_inputs()}
        fast_inputs = {i.name: i for i in self.fast.get_inputs()}
        self._slow_cache_shape = (1, int(m['n_local_heads']), self.max_seq_len, int(m['head_dim']))
        self._slow_cache_dtype = _ORT_DTYPES[slow_inputs['cache_key_0'].type]
        self._fast_cache_shape = (1, int(m['fast_n_local_heads']), self.codebooks, int(m['fast_head_dim']))
        self._fast_cache_dtype = _ORT_DTYPES[fast_inputs['cache_key_0'].type]
        self._hidden_dtype = _ORT_DTYPES[fast_inputs['slow_hidden'].type]

    # -- voices ------------------------------------------------------------
    def voice(self, samples: np.ndarray, rate: int, transcript: str) -> Voice:
        """The voice of a reference clip, its codes encoded once and cached.

        A clip longer than 30 seconds is cut (near a pause); its transcript
        then no longer matches, so it is dropped and the voice is cloned from
        the sound alone.
        """
        samples = np.asarray(samples, dtype=np.float32)
        if len(samples) < MIN_REFERENCE_SECONDS * rate:
            raise ValueError('The reference clip is shorter than half a second.')
        if len(samples) > MAX_REFERENCE_SECONDS * rate:
            samples = dsp.cut(samples, rate, MAX_REFERENCE_SECONDS)
            if transcript:
                log.info('reference cut to %.1f s; its transcript no longer matches and is not used',
                         len(samples) / rate)
            transcript = ''
        audio = dsp.resample(samples, rate, SAMPLE_RATE)
        key = hashlib.sha256(self.fingerprint.encode('ascii') + audio.tobytes()).hexdigest()
        path = self.voices_dir / f'{key}.npy'
        try:
            codes = np.load(path, allow_pickle=False).astype(np.int64)
            if codes.ndim == 2 and codes.shape[0] == self.codebooks and codes.shape[1]:
                return Voice(codes, transcript.strip())
        except (OSError, ValueError):
            pass
        codes = self._encode(audio)
        try:
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + '.part')
            with part.open('wb') as out:
                np.save(out, codes.astype(np.uint16))
            os.replace(part, path)
        except OSError as exc:
            log.warning('could not cache the voice: %s', exc)
        return Voice(codes, transcript.strip())

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        """Codec codes of 44.1 kHz audio. The encoder (414 MB) is loaded for
        this and released after, as upstream's registration does."""
        manifest = json.loads(self.files['registration/registration_manifest.json'].read_text(encoding='utf-8'))
        if manifest.get('model_fingerprint') != self.fingerprint:
            raise ValueError('the registration encoder does not match the model')
        padding = (-audio.size) % FRAME_SAMPLES
        values = np.pad(audio, (0, padding)).reshape(1, 1, -1)
        session = _session(self.files['registration/codec_encoder_fp16.onnx'], self.threads, run_once=True)
        try:
            dtype = _ORT_DTYPES[session.get_inputs()[0].type]
            codes = np.asarray(session.run(None, {'audio': values.astype(dtype)})[0], dtype=np.int64)
        finally:
            del session
            gc.collect()
        if codes.ndim == 3:
            codes = codes[0]
        if codes.ndim != 2 or codes.shape[0] != self.codebooks or not codes.shape[1]:
            raise RuntimeError(f'the encoder returned codes of shape {codes.shape}')
        return codes

    # -- generation --------------------------------------------------------
    def prompt(self, text: str, voice: Voice) -> np.ndarray:
        """(1, 11, length) int64: the token row, and the reference's codebooks
        under its semantic tokens."""
        prefix, suffix = self.tokenizer.prompt(text, voice.transcript)
        codes = voice.codes
        row = np.asarray(prefix + (codes[0] + self.semantic_begin).tolist() + suffix, dtype=np.int64)
        values = np.zeros((self.codebooks + 1, row.size), dtype=np.int64)
        values[0] = row
        values[1:, len(prefix):len(prefix) + codes.shape[1]] = codes
        return values[np.newaxis]

    def _slow_step(self, codes: np.ndarray, positions: np.ndarray, caches: list[np.ndarray]):
        feeds = {'codes': codes, 'input_pos': positions}
        for i in range(self.layers):
            feeds[f'cache_key_{i}'] = caches[2 * i]
            feeds[f'cache_value_{i}'] = caches[2 * i + 1]
        outputs = self.slow.run(None, feeds)
        for index, delta in enumerate(outputs[2:]):
            caches[index][:, :, positions, :] = delta
        return np.asarray(outputs[0])[0, -1], np.asarray(outputs[1])[:, -1:, :]

    def _fast_step(self, hidden: np.ndarray, token: int, use_hidden: bool, position: int,
                   caches: list[np.ndarray]) -> np.ndarray:
        feeds = {'slow_hidden': hidden.astype(self._hidden_dtype, copy=False),
                 'token_id': np.asarray([[token]], dtype=np.int64),
                 'use_slow_hidden': np.asarray([use_hidden], dtype=np.bool_),
                 'input_pos': np.asarray([position], dtype=np.int64)}
        for i in range(self.fast_layers):
            feeds[f'cache_key_{i}'] = caches[2 * i]
            feeds[f'cache_value_{i}'] = caches[2 * i + 1]
        outputs = self.fast.run(None, feeds)
        at = np.asarray([position])
        for index, delta in enumerate(outputs[1:]):
            caches[index][:, :, at, :] = delta
        return np.asarray(outputs[0])[0, -1]

    def _semantic(self, logits: np.ndarray, recent: list[int], rng: np.random.Generator) -> int:
        values = np.asarray(logits).reshape(-1)
        if values.size != self.allowed.size:
            raise ValueError(f'slow logits of size {values.size}, expected {self.allowed.size}')
        normal = int(self.allowed[_sample(values, TEMPERATURE, TOP_P, TOP_K, rng)])
        # Drawn every step, as upstream does, so a seed gives upstream's tokens.
        high = int(self.allowed[_sample(values, 1.0, 0.9, TOP_K, rng)])
        if self.semantic_begin <= normal <= self.semantic_end and normal in recent:
            return high
        return normal

    def generate(self, text: str, voice: Voice, seed: int = 42,
                 progress: Callable[[float], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> np.ndarray:
        """The line's codec codes, (10, frames). progress(fraction) as frames come."""
        cancelled = cancelled or (lambda: False)
        prompt = self.prompt(text, voice)
        length = int(prompt.shape[2])
        if length >= self.max_seq_len - 16:
            raise ValueError(f'the line and its reference take {length} positions, '
                             f'over the model\'s {self.max_seq_len}; use a shorter reference or line')
        limit = min(max_frames(text), self.max_seq_len - length)
        expected = expected_frames(text)
        rng = np.random.default_rng(seed)
        slow_caches = [np.zeros(self._slow_cache_shape, self._slow_cache_dtype) for _ in range(2 * self.layers)]
        logits, hidden = self._slow_step(prompt, np.arange(length, dtype=np.int64), slow_caches)
        recent: list[int] = []
        frames: list[np.ndarray] = []
        for step in range(limit):
            if cancelled():
                raise Cancelled()
            semantic = self._semantic(logits, recent, rng)
            if semantic == self.stop:
                break
            recent = (recent + [semantic])[-RECENT:]
            fast_caches = [np.zeros(self._fast_cache_shape, self._fast_cache_dtype)
                           for _ in range(2 * self.fast_layers)]
            self._fast_step(hidden, 0, True, 0, fast_caches)
            token = min(max(semantic - self.semantic_begin, 0), self.codebook_size - 1)
            frame = [token]
            for position in range(1, self.codebooks):
                token = _sample(self._fast_step(hidden, token, False, position, fast_caches),
                                TEMPERATURE, TOP_P, TOP_K, rng)
                frame.append(token)
            frames.append(np.asarray(frame, dtype=np.int64))
            if progress is not None:
                progress(min(0.95, len(frames) / (expected * 1.1)))
            if step + 1 >= limit:
                log.warning('line stopped at the cap of %d frames (%.1f s)', limit, limit / FRAMES_PER_SECOND)
                break
            column = np.concatenate([[semantic], frame]).reshape(1, -1, 1)
            logits, hidden = self._slow_step(column, np.asarray([length + step], dtype=np.int64), slow_caches)
        if not frames:
            raise RuntimeError('the model produced no audio for this line')
        return np.stack(frames, axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        audio = self.decoder.run(None, {'codes': np.asarray(codes, dtype=np.int64)[np.newaxis]})[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize(self, text: str, voice: Voice, seed: int = 42,
                   progress: Callable[[float], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> np.ndarray:
        """44.1 kHz mono float32. progress(fraction): 0.8 at the end of the
        transformers, 1.0 once decoded."""
        codes = self.generate(text, voice, seed,
                              (lambda f: progress(0.8 * f)) if progress else None, cancelled)
        if cancelled is not None and cancelled():
            raise Cancelled()
        wav = self.decode(codes)
        if progress is not None:
            progress(1.0)
        return wav
