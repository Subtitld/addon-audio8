"""Audio8 TTS add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline voice-cloning text-to-speech with Audio8 TTS Preview 0.6B
(Apache-2.0), at 44.1 kHz, in eleven languages. Serves `tts.synthesize`.
ONNX Runtime runs the official INT4 ONNX release on the CPU (engine.py); the
model downloads on the first line (download.py), and after that nothing
leaves the machine.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:audio8]` and surfaces.

Options (the host sets them from Configure as AUDIO8_<KEY>)
-------------------------------------------------------------
threads: CPU threads, 0 for automatic; voice_ref_audio and voice_ref_text: a
clip, and what it says, the speaker's own voice falls back on when the host
sends none. skip_ref_text is read by the host, which then sends no transcript.

The reference transcript
------------------------
The model clones best with the exact words of the reference clip, which the
host sends as `voice_ref_text` (the subtitles of the clip). Without them it
clones from the clip's sound alone, a little less close to the speaker.

Limits
------
One line at a time. A cancel stops the line between frames, or stops the
download (which resumes on the next request).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from audio8_addon import ADDON_ID, PROTOCOL_VERSION, __version__
from audio8_addon import audio, download, tls, voices

ADDON_VERSION = __version__

ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_UNSUPPORTED_VOICE = 'unsupported_voice'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network_unavailable'
ERR_OOM = 'out_of_memory'

# No space left on the device, or over the user's disk quota.
_DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, 'EDQUOT', 122)}
# Beside a package module: frozen, this script itself runs from _internal/.
_PINS_FILE = Path(download.__file__).resolve().parent / 'pins.json'

log = logging.getLogger('audio8_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`, so nothing else can write a
    stray line into the protocol. Falls back to the Python-level swap if the
    fd dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _env_number(name: str, default: float, low: float, high: float) -> float:
    raw = _env(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        log.warning('%s=%r is not a number; using %s', name, raw, default)
        value = default
    return max(low, min(high, value))


def _options() -> dict:
    return {
        'threads': int(_env_number('AUDIO8_THREADS', 0, 0, 256)),
        'voice_ref_audio': _env('AUDIO8_VOICE_REF_AUDIO'),
        'voice_ref_text': _env('AUDIO8_VOICE_REF_TEXT'),
    }


def _resolve_language(raw) -> str:
    primary = voices.primary_language(raw)
    if primary in voices.LANGUAGES:
        return primary
    spoken = ', '.join(voices.LANGUAGE_NAMES[code] for code in voices.LANGUAGES)
    if not primary:
        raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                          f"Set the project's language first: Audio8 TTS speaks {spoken}.")
    raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                      f'Audio8 TTS does not speak {raw!r}. It speaks {spoken}; '
                      f'pick another voice engine for other languages.')


def _seed(params: dict) -> int:
    try:
        return int(params.get('seed', 42)) & 0xFFFFFFFF or 1
    except (TypeError, ValueError):
        return 42


# ---------------------------------------------------------------------------
# ffmpeg, for references that are not PCM WAV
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    for candidate in (_env('SUBTITLD_FFMPEG_EXECUTABLE'), _env('FFMPEG_EXECUTABLE')):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _child_env() -> dict | None:
    """Environment for ffmpeg, or None to inherit ours unchanged.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH, and a system ffmpeg that inherits it loads the bundle's
    libraries instead of its own. PyInstaller saves the host value in
    LD_LIBRARY_PATH_ORIG, so restore it for the child.
    """
    if not (getattr(sys, 'frozen', False) and sys.platform.startswith('linux')):
        return None
    env = dict(os.environ)
    original = env.get('LD_LIBRARY_PATH_ORIG')
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


def _read_reference(path: str) -> tuple:
    """(samples, rate): `path` itself if it is PCM WAV, else via ffmpeg."""
    try:
        return audio.read_wav(path)
    except audio.UnreadableAudio:
        pass
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise _AddonError(ERR_BAD_PARAMS,
                          f'The reference {Path(path).name} is not PCM WAV, and no ffmpeg was '
                          f'found to convert it. Use a WAV file.')
    scratch = Path(tempfile.mkdtemp(prefix='audio8-', dir=_env('SUBTITLD_TEMP_DIR') or None))
    try:
        out = scratch / 'reference.wav'
        cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', path,
               '-vn', '-ac', '1', '-ar', '44100', '-c:a', 'pcm_s16le', str(out)]
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, env=_child_env(),
                              startupinfo=_no_window_startupinfo())
        if done.returncode != 0 or not out.is_file():
            tail = ' / '.join(done.stderr.decode('utf-8', 'replace').strip().splitlines()[-3:])
            raise _AddonError(ERR_BAD_PARAMS, f'Could not read the reference {Path(path).name}: {tail}')
        try:
            return audio.read_wav(out)
        except audio.UnreadableAudio as exc:
            raise _AddonError(ERR_BAD_PARAMS, f'Could not read the reference {Path(path).name}: {exc}') from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# The model: downloaded and loaded on first use, shared by every request
# ---------------------------------------------------------------------------
_pins: dict | None = None
_loaded = None              # engine.Audio8
_loaded_threads = None
_model_lock = threading.Lock()

DISPLAY = 'Audio8 TTS Preview 0.6B'


def _pin_index() -> dict:
    global _pins
    if _pins is None:
        _pins = json.loads(_PINS_FILE.read_text(encoding='utf-8'))
    return _pins


def pins() -> list[dict]:
    """The pinned files the model reads, each with its repository and revision."""
    return [{**pin, 'key': key, 'repo': entry['repo'], 'revision': entry['revision']}
            for key, entry in _pin_index().items() for pin in entry['files']]


def model_files() -> dict:
    """{path in the repository: local path} of every pinned file."""
    return {pin['path']: download.local_path(pin) for pin in pins()}


def _megabytes(count: int) -> str:
    return f'{count / 1e6:,.0f} MB'


def _load(state: '_Request', options: dict):
    """Download (once) and load. The caller holds _model_lock."""
    global _loaded, _loaded_threads
    from audio8_addon import engine  # numpy and onnxruntime: only once a line comes

    threads = options['threads'] or engine.default_threads()
    if _loaded is not None and _loaded_threads == threads:
        return _loaded
    todo = download.missing(pins())
    if todo:
        total = sum(pin['size'] for pin in todo)
        message = f'Downloading {DISPLAY} ({_megabytes(total)})...'
        _emit_progress(state.req_id, 0.0, message)
        last = [0.0]

        def progress(done: int, total_bytes: int) -> None:
            now = time.monotonic()
            if now - last[0] >= 0.25 or done >= total_bytes:
                last[0] = now
                _emit_progress(state.req_id, 0.3 * done / max(1, total_bytes), message)

        started = time.monotonic()
        download.ensure(pins(), progress, state.cancel.is_set)
        log.info('downloaded %s in %.1fs', _megabytes(total), time.monotonic() - started)
    _emit_progress(state.req_id, 0.3, f'Loading {DISPLAY}...')
    started = time.monotonic()
    _loaded = _loaded_threads = None
    _loaded = engine.Audio8(model_files(), threads, download.cache_root() / 'voices')
    _loaded_threads = threads
    log.info('loaded with %d threads in %.1fs', threads, time.monotonic() - started)
    return _loaded


def _voice(model, path: str, transcript: str):
    """The voice of a reference clip; its codes are cached by the engine."""
    samples, rate = _read_reference(path)
    if len(samples) < rate * 0.5:
        raise _AddonError(ERR_BAD_PARAMS, 'The reference clip is shorter than half a second; '
                                          'Audio8 TTS needs 3 to 30 seconds of one person speaking.')
    return model.voice(samples, rate, transcript)


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()


def _handle_synthesize(state: _Request, params: dict) -> None:
    req_id = state.req_id
    text = ' '.join(str(params.get('text') or '').split())
    output_path = params.get('output_path')
    if not text:
        _emit_error(req_id, ERR_BAD_PARAMS, 'text is empty')
        return
    if not isinstance(output_path, str) or not output_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'output_path is required')
        return
    options = _options()
    try:
        language = _resolve_language(params.get('language'))
    except _AddonError as exc:
        _emit_error(req_id, exc.code, exc.message)
        return
    voice_id = str(params.get('voice') or '') or voices.CLONE
    if voice_id != voices.CLONE:
        _emit_error(req_id, ERR_UNSUPPORTED_VOICE, f'unknown voice {voice_id!r}')
        return
    seed = _seed(params)

    reference = str(params.get('voice_ref_audio') or '')
    transcript = ' '.join(str(params.get('voice_ref_text') or '').split())
    if not reference:
        reference, transcript = options['voice_ref_audio'], options['voice_ref_text']
    if not reference:
        _emit_error(req_id, ERR_BAD_PARAMS,
                    "The speaker's own voice needs a clip of it, and none came with the request. "
                    'Pick a reference clip in Configure.')
        return
    if not os.path.isfile(reference):
        _emit_error(req_id, ERR_BAD_PARAMS, f'The reference clip is missing: {reference}')
        return

    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        from audio8_addon import engine
        if state.cancel.is_set():
            raise engine.Cancelled()
        with _model_lock:
            model = _load(state, options)
            if state.cancel.is_set():
                raise engine.Cancelled()
            _emit_progress(req_id, 0.32, 'Listening to the reference...')
            voice = _voice(model, reference, transcript)
            if not voice.transcript:
                log.info('no transcript of the reference: cloning from its sound alone')
            _emit_progress(req_id, 0.35, 'Speaking...')

            def progress(fraction: float) -> None:
                _emit_progress(req_id, 0.35 + 0.65 * fraction, 'Speaking...')

            log.info('synthesize %s: language=%s seed=%d chars=%d reference=%.1fs transcript=%d chars',
                     req_id, language, seed, len(text), voice.codes.shape[1] / engine.FRAMES_PER_SECOND,
                     len(voice.transcript))
            started = time.monotonic()
            wav = model.synthesize(text, voice, seed=seed, progress=progress,
                                   cancelled=state.cancel.is_set)
            took = time.monotonic() - started
        if state.cancel.is_set():
            raise engine.Cancelled()
        audio.write_wav(Path(output_path), wav, engine.SAMPLE_RATE)
        duration = len(wav) / engine.SAMPLE_RATE
        log.info('synthesize %s: %.2fs of audio in %.2fs', req_id, duration, took)
        outcome = ('result', {
            'path': output_path,
            'duration_sec': round(duration, 3),
            'sample_rate': engine.SAMPLE_RATE,
            'channels': 1,
            'voice': voices.CLONE,
        })
    except (download.Cancelled, _cancelled_type()):
        log.info('synthesize %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except download.DownloadError as exc:
        log.warning('synthesize %s: download failed: %s', req_id, exc)
        outcome = ('error', ERR_NETWORK, f'Could not download the model: {exc}', True)
    except _AddonError as exc:
        log.warning('synthesize %s failed: [%s] %s', req_id, exc.code, exc.message)
        outcome = ('error', exc.code, exc.message, exc.retryable)
    except MemoryError:
        outcome = ('error', ERR_OOM, 'Out of memory. Audio8 TTS needs about 3 GB free.', False)
    except OSError as exc:
        log.exception('synthesize %s failed', req_id)
        if getattr(exc, 'errno', None) in _DISK_FULL_ERRNOS:
            outcome = ('error', ERR_DISK_FULL, f'Out of disk space: {exc}', False)
        else:
            outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)
    except ValueError as exc:
        log.warning('synthesize %s failed: %s', req_id, exc)
        outcome = ('error', ERR_BAD_PARAMS, str(exc), False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('synthesize %s failed', req_id)
        outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _cancelled_type():
    from audio8_addon import engine
    return engine.Cancelled


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'audio8-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': 'tts.synthesize',
            'languages': list(voices.LANGUAGES),
            'voices': voices.manifest_voices(),
            'voice_clone': True,
        }],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == 'tts.synthesize':
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('tts.synthesize without id')
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_synthesize, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Check what the frozen bundle needs, without the network or the model.

    ONNX Runtime must load and run a graph, the pins must be there, the
    Unicode database the tokenizer's pattern is built from, and the TLS and
    hashing modules the downloads use.
    """
    import ssl

    import numpy as np
    import onnxruntime as ort

    from audio8_addon import dsp, engine, text as audio8_text

    from onnxruntime import InferenceSession
    session = InferenceSession(_tiny_graph(), providers=['CPUExecutionProvider'])
    doubled = session.run(None, {'x': np.ones(3, dtype=np.float32)})[0]
    tone = np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000).astype(np.float32)
    up = dsp.resample(tone, 16000, engine.SAMPLE_RATE)
    # The Qwen pattern is built from the Unicode database the bundle carries.
    words = audio8_text._qwen_pattern().findall("It's 3 o'clock, 你好 Żółć!")
    ok = (bool(np.allclose(doubled, 2)) and len(pins()) == len(engine.FILES)
          and abs(len(up) - engine.SAMPLE_RATE) <= 1 and len(words) == 10)
    report = {
        'ok': ok,
        'version': ADDON_VERSION,
        'onnxruntime': ort.__version__,
        'numpy': np.__version__,
        'pins': len(pins()),
        'words': len(words),
        'openssl': ssl.OPENSSL_VERSION,
        'sha256': hashlib.sha256(b'').hexdigest()[:8],
        'ca_bundle': os.environ.get('SSL_CERT_FILE', ''),
    }
    # ASCII: a Windows console encodes cp1252.
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0 if ok else 1


def _tiny_graph() -> bytes:
    """A serialized ONNX model computing x + x, written without the onnx package."""
    def varint(n: int) -> bytes:
        out = b''
        while True:
            byte, n = n & 0x7F, n >> 7
            out += bytes([byte | (0x80 if n else 0)])
            if not n:
                return out

    def field(number: int, payload: bytes) -> bytes:  # length-delimited
        return varint(number << 3 | 2) + varint(len(payload)) + payload

    def number(field_no: int, value: int) -> bytes:   # varint
        return varint(field_no << 3) + varint(value)

    node = field(1, b'x') + field(1, b'x') + field(2, b'y') + field(4, b'Add')
    dim = field(1, number(1, 3))                                  # TensorShapeProto.Dimension
    tensor = number(1, 1) + field(2, dim)                         # elem_type FLOAT, shape
    value_type = field(1, tensor)                                 # TypeProto.tensor_type

    def value_info(name: bytes) -> bytes:
        return field(1, name) + field(2, value_type)

    graph = field(1, node) + field(2, b'g') + field(11, value_info(b'x')) + field(12, value_info(b'y'))
    opset = field(1, b'') + number(2, 13)
    return number(1, 8) + field(7, graph) + field(8, opset)


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'audio8-addon {ADDON_VERSION}')
        return 0
    logging.basicConfig(
        level=(_env('AUDIO8_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    tls.configure_ca_bundle()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('audio8-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        _stop_workers()
    return rc


if __name__ == '__main__':
    sys.exit(main())
