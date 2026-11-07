"""Real synthesis with the pinned model, from source.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The model (1 GB) downloads into $SUBTITLD_MODELS_DIR, or
~/.cache/subtitld/models, on the first run, checked against pins.json.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audio8_addon import __main__ as addon, audio, download, engine, voices  # noqa: E402

pytestmark = pytest.mark.model

CLIP = REPO_ROOT / 'tests' / 'data' / 'en-woman.wav'
TRANSCRIPT = ('When one listens, by the side of honest men at the portals of society, '
              'one overhears the dialogues of those who are on the outside.')


@pytest.fixture(scope='module', params=list(voices.MODEL_LANGUAGES))
def model(request, tmp_path_factory):
    model = request.param
    download.ensure(addon.pins(model), lambda *_: None, lambda: False)
    return engine.Audio8(addon.model_files(model), engine.default_threads(), tmp_path_factory.mktemp('voices'))


@pytest.fixture(scope='module')
def voice(model):
    samples, rate = audio.read_wav(CLIP)
    return model.voice(samples, rate, TRANSCRIPT)


def _check_speech(wav, low=0.8, high=8):
    seconds = len(wav) / engine.SAMPLE_RATE
    assert low < seconds < high, seconds
    assert wav.dtype == np.float32 and np.isfinite(wav).all()
    assert 0.01 < float(np.sqrt(np.mean(wav ** 2))) < 0.5


def test_a_voice_is_encoded_once(model, voice):
    assert voice.codes.shape[0] == 10
    # 11.4 s at 21.5 frames a second
    assert abs(voice.codes.shape[1] - 11.4 * engine.FRAMES_PER_SECOND) < 3
    assert voice.transcript == TRANSCRIPT
    cached = list(model.voices_dir.glob('*.npy'))
    assert len(cached) == 1
    samples, rate = audio.read_wav(CLIP)
    calls = []
    real = model._encode
    model._encode = lambda audio_: calls.append(1) or real(audio_)
    try:
        again = model.voice(samples, rate, '')
    finally:
        del model._encode
    assert not calls and np.array_equal(again.codes, voice.codes) and again.transcript == ''


@pytest.mark.parametrize('text', [
    'We have to leave before sunrise.',
    '我们必须在日出之前出发。',
    '日の出前に出発しなければなりません。',
    'Musimy wyjść przed świtem.',
])
def test_speaks(model, voice, text):
    _check_speech(model.synthesize(text, voice, seed=3))


def test_speaks_without_a_transcript(model, voice):
    _check_speech(model.synthesize('We have to leave before sunrise.', engine.Voice(voice.codes, ''), seed=3))


def test_the_same_seed_gives_the_same_line(model, voice):
    a = model.generate('Same line, same voice.', voice, seed=11)
    b = model.generate('Same line, same voice.', voice, seed=11)
    assert np.array_equal(a, b)


def test_progress_rises_and_ends_at_one(model, voice):
    seen = []
    model.synthesize('We have to leave before sunrise.', voice, seed=2, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == 1.0 and len(seen) > 10


def test_cancel_stops_between_frames(model, voice):
    flag = threading.Event()
    calls = []

    def progress(value):
        calls.append(value)
        flag.set()

    with pytest.raises(engine.Cancelled):
        model.synthesize('A line that is cancelled as soon as it starts to be spoken.', voice,
                         progress=progress, cancelled=flag.is_set)
    assert len(calls) == 1


def test_a_line_that_runs_on_stops_at_the_cap(model, voice, monkeypatch):
    """The stop token never comes: the cap ends the line."""
    stop_index = model.allowed.size - 1

    def never_stop(logits, recent, rng):
        values = np.asarray(logits).reshape(-1).copy()
        values[stop_index] = -np.inf
        return int(model.allowed[engine._sample(values, engine.TEMPERATURE, engine.TOP_P, engine.TOP_K, rng)])

    monkeypatch.setattr(model, '_semantic', never_stop)
    codes = model.generate('Short.', voice, seed=1)
    assert codes.shape[1] == engine.max_frames('Short.')


def test_a_long_reference_is_cut_and_loses_its_transcript(model):
    samples, rate = audio.read_wav(CLIP)
    long = np.concatenate([samples] * 3)        # 34 s
    voice = model.voice(long, rate, TRANSCRIPT)
    assert voice.transcript == ''
    assert voice.codes.shape[1] <= engine.MAX_REFERENCE_SECONDS * engine.FRAMES_PER_SECOND + 1
