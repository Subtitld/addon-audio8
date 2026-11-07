"""The frozen binary, unpacked from its release archive the way Subtitld installs it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

$AUDIO8_TEST_BINARY names the binary; without it these tests skip (and the
release workflow refuses skipped tests). The binary downloads the model (1 GB)
into $SUBTITLD_MODELS_DIR on its first line, with nothing but what is in its
own bundle, then clones a clip in English and Japanese, and cancels a line.

    AUDIO8_TEST_BINARY=dist/audio8-addon/audio8-addon pytest tests/test_installed.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import wave
from pathlib import Path

import pytest

BINARY = os.environ.get('AUDIO8_TEST_BINARY', '')
CLIP = Path(__file__).resolve().parent / 'data' / 'en-woman.wav'
TRANSCRIPT = ('When one listens, by the side of honest men at the portals of society, '
              'one overhears the dialogues of those who are on the outside.')
pytestmark = pytest.mark.skipif(not BINARY, reason='AUDIO8_TEST_BINARY is not set')


class Addon:
    """The binary on the protocol, with a watchdog so a hang fails instead of stalling CI."""

    def __init__(self, tmp_path: Path, timeout: float, **options: str):
        env = dict(os.environ, SUBTITLD_TEMP_DIR=str(tmp_path), **options)
        env.setdefault('SUBTITLD_MODELS_DIR', str(tmp_path / 'models'))
        self.proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        self.timer = threading.Timer(timeout, self.proc.kill)
        self.timer.start()

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, 'the add-on closed its output (crashed or timed out)'
        return json.loads(line)

    def until_terminal(self, req_id: str, on_progress=None) -> dict:
        while True:
            frame = self.read()
            if frame.get('id') != req_id:
                continue
            if frame['type'] in ('result', 'error'):
                return frame
            if on_progress is not None:
                on_progress(frame)

    def close(self) -> int:
        try:
            self.send({'type': 'shutdown'})
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=15)
        finally:
            self.timer.cancel()
            self.proc.kill()


def test_version():
    out = subprocess.run([BINARY, '--version'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith('audio8-addon ')


def test_self_test_loads_onnx_runtime():
    out = subprocess.run([BINARY, '--self-test'], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['ok'] is True and report['pins'] == {'realistic-0.6b-int4': 11, 'preview-0.6b-int4': 11}


def test_clones_a_clip_then_cancels(tmp_path):
    addon = Addon(tmp_path, timeout=1800)
    try:
        hello = addon.read()
        assert hello['type'] == 'hello' and hello['addon'] == 'audio8'
        addon.send({'type': 'ready', 'host': 'test'})

        out = tmp_path / 'line.wav'
        addon.send({'id': 'a', 'type': 'tts.synthesize', 'params': {
            'text': 'Good morning. Let us start the recording.', 'voice': 'audio8-clone', 'language': 'en-US',
            'seed': 5, 'output_path': str(out), 'voice_ref_audio': str(CLIP), 'voice_ref_text': TRANSCRIPT}})
        result = addon.until_terminal('a')
        assert result['type'] == 'result', result
        assert result['data']['sample_rate'] == 44100 and result['data']['duration_sec'] > 1
        with wave.open(str(out)) as clip:
            assert clip.getframerate() == 44100 and clip.getnframes() > 44100

        # A clip that is not 16-bit PCM WAV goes through ffmpeg when there is one;
        # the line just written is one the bundle reads itself.
        ja = tmp_path / 'ja.wav'
        addon.send({'id': 'b', 'type': 'tts.synthesize', 'params': {
            'text': 'また明日会いましょう。', 'language': 'ja', 'seed': 5,
            'output_path': str(ja), 'voice_ref_audio': str(out)}})
        assert addon.until_terminal('b')['type'] == 'result'
        assert ja.stat().st_size > 1000

        cancelled = []

        def on_progress(frame):
            if frame.get('message') == 'Speaking...' and frame['value'] > 0.4 and not cancelled:
                cancelled.append(frame['value'])
                addon.send({'type': 'cancel', 'target': 'c'})

        addon.send({'id': 'c', 'type': 'tts.synthesize', 'params': {
            'text': 'This line is long enough to be interrupted before it ends, ' * 3,
            'language': 'en', 'output_path': str(tmp_path / 'c.wav'), 'voice_ref_audio': str(CLIP)}})
        error = addon.until_terminal('c', on_progress)
        assert cancelled and error['type'] == 'error' and error['code'] == 'cancelled'
        assert not (tmp_path / 'c.wav').exists()
    finally:
        assert addon.close() == 0
