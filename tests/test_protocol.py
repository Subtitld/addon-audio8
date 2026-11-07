"""The add-on spoken over its wire format, in a real process.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

These run `python -m audio8_addon` and talk JSON lines to it, the way the host
does. They cover every answer the add-on owes the host without the model or
the network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audio8_addon import voices  # noqa: E402

CLIP = REPO_ROOT / 'tests' / 'data' / 'en-woman.wav'


def _start(tmp_path, **options):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    env['SUBTITLD_MODELS_DIR'] = str(tmp_path / 'models')
    env['HF_ENDPOINT'] = 'http://127.0.0.1:9'  # nothing listens: no test reaches the network
    for key in ('AUDIO8_VOICE_REF_AUDIO', 'AUDIO8_VOICE_REF_TEXT', 'AUDIO8_THREADS', 'AUDIO8_MODEL'):
        env.pop(key, None)
    env.update(options)
    return subprocess.Popen([sys.executable, '-m', 'audio8_addon'], cwd=tmp_path, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture()
def proc(tmp_path):
    process = _start(tmp_path)
    yield process
    if process.poll() is None:
        process.kill()
    process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close()


def _send(proc, frame: dict) -> None:
    proc.stdin.write((json.dumps(frame) + '\n').encode('utf-8'))
    proc.stdin.flush()


def _recv(proc) -> dict:
    line = proc.stdout.readline()
    assert line, proc.stderr.read().decode('utf-8', 'replace')
    return json.loads(line)


def _terminal(proc, req_id: str) -> dict:
    while True:
        frame = _recv(proc)
        if frame.get('id') == req_id and frame['type'] in ('result', 'error'):
            return frame


def _synth(proc, req_id: str, tmp_path: Path, **params) -> dict:
    params.setdefault('output_path', str(tmp_path / f'{req_id}.wav'))
    _send(proc, {'id': req_id, 'type': 'tts.synthesize', 'params': params})
    return _terminal(proc, req_id)


def test_hello_lists_languages_and_the_voice(proc):
    hello = _recv(proc)
    assert hello['type'] == 'hello' and hello['addon'] == 'audio8' and hello['protocol'] == 1
    [capability] = hello['capabilities']
    assert capability['task'] == 'tts.synthesize'
    assert capability['languages'] == list(voices.LANGUAGES)
    assert capability['voice_clone'] is True
    assert capability['voices'] == [{'id': 'audio8-clone', 'language': '*', 'sample_rate': 44100,
                                     'display_name': "Speaker's own voice (cloned from the video)",
                                     'requires': ['voice_ref_audio']}]


def test_requests_it_cannot_serve_get_clear_errors(proc, tmp_path):
    _recv(proc)
    _send(proc, {'type': 'ready', 'host': 'subtitld', 'host_version': 'test'})
    ref = {'voice_ref_audio': str(CLIP)}
    assert _synth(proc, 'a', tmp_path, text='Ahoj.', language='cs', **ref)['code'] == 'unsupported_language'
    empty = _synth(proc, 'b', tmp_path, text='Hello.', language='', **ref)
    assert empty['code'] == 'unsupported_language' and "project's language" in empty['message']
    assert _synth(proc, 'c', tmp_path, text='Hello.', voice='alba', language='en', **ref)['code'] == 'unsupported_voice'
    assert _synth(proc, 'd', tmp_path, text='  ', language='en', **ref)['code'] == 'bad_params'
    _send(proc, {'id': 'e', 'type': 'tts.synthesize', 'params': {'text': 'Hi.', 'language': 'en', **ref}})
    assert _terminal(proc, 'e')['code'] == 'bad_params'  # no output_path
    _send(proc, {'id': 'f', 'type': 'asr.transcribe', 'params': {}})
    assert _terminal(proc, 'f')['code'] == 'bad_params'


def test_the_clone_needs_a_clip(proc, tmp_path):
    _recv(proc)
    none = _synth(proc, 'a', tmp_path, text='Hello.', voice='audio8-clone', language='en')
    assert none['code'] == 'bad_params' and 'Configure' in none['message']
    gone = _synth(proc, 'b', tmp_path, text='Hello.', language='en', voice_ref_audio=str(tmp_path / 'missing.wav'))
    assert gone['code'] == 'bad_params' and 'missing' in gone['message']


def test_a_download_that_cannot_reach_the_server_is_retryable(proc, tmp_path):
    _recv(proc)
    frame = _synth(proc, 'g', tmp_path, text='你好。', language='zh-CN', voice_ref_audio=str(CLIP))
    assert frame['code'] == 'network_unavailable' and frame['retryable'] is True


def test_the_configured_clip_is_the_fallback(tmp_path):
    """No clip in the request: Configure's clip is used (and the download is tried)."""
    process = _start(tmp_path, AUDIO8_VOICE_REF_AUDIO=str(CLIP), AUDIO8_VOICE_REF_TEXT='Words.')
    try:
        _recv(process)
        frame = _synth(process, 'h', tmp_path, text='Hello.', language='en')
        assert frame['code'] == 'network_unavailable'
    finally:
        process.kill()
        process.communicate()


@pytest.mark.parametrize('model, language, used', [
    ('preview-0.6b-int4', 'fr', 'Preview'),
    ('preview-0.6b-int4', 'pt-BR', 'Realistic'),   # Preview doesn't speak it
    ('', 'fr', 'Realistic'),                         # the default
    ('nonsense', 'fr', 'Realistic'),
])
def test_each_line_downloads_the_model_it_needs(tmp_path, model, language, used):
    process = _start(tmp_path, AUDIO8_MODEL=model)
    try:
        _recv(process)
        progress = []
        _send(process, {'id': 'm', 'type': 'tts.synthesize', 'params': {
            'text': 'Bonjour.', 'language': language, 'voice_ref_audio': str(CLIP),
            'output_path': str(tmp_path / 'm.wav')}})
        while True:
            frame = _recv(process)
            if frame['type'] == 'progress':
                progress.append(frame['message'])
            elif frame.get('id') == 'm':
                break
        assert frame['code'] == 'network_unavailable'
        assert any(f'Downloading Audio8 TTS {used} 0.6B' in message for message in progress), progress
    finally:
        process.kill()
        process.communicate()


def test_shutdown_exits_cleanly(proc):
    _recv(proc)
    _send(proc, {'type': 'shutdown'})
    assert proc.wait(timeout=10) == 0
