"""manifest.json, pins.json and the code agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audio8_addon import ADDON_ID, __version__, voices  # noqa: E402
from audio8_addon import __main__ as addon  # noqa: E402
from audio8_addon.engine import FILES  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
PINS = json.loads((REPO_ROOT / 'audio8_addon' / 'pins.json').read_text(encoding='utf-8'))
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['voice_clone'] is True
    assert MANIFEST['compute'] == 'cpu' and 'accelerators' not in MANIFEST
    fields = {f['key']: f for f in MANIFEST['config_schema']['fields']}
    assert set(fields) == {'model', 'threads', 'skip_ref_text', 'voice_ref_audio', 'voice_ref_text'}
    assert fields['skip_ref_text']['type'] == 'bool'   # the host reads it, by this key
    assert fields['model']['type'] == 'select' and fields['model']['default'] == voices.REALISTIC
    assert [o['value'] for o in fields['model']['options']] == list(voices.MODEL_LANGUAGES)


def test_realistic_speaks_every_preview_language():
    assert set(voices.MODEL_LANGUAGES[voices.PREVIEW]) < set(voices.LANGUAGES)
    assert len(voices.LANGUAGES) == 23 and set(voices.LANGUAGES) == set(voices.LANGUAGE_NAMES)
    assert voices.model_for('fr', voices.PREVIEW) == voices.PREVIEW
    assert voices.model_for('pt', voices.PREVIEW) == voices.REALISTIC
    assert voices.model_for('fr', voices.REALISTIC) == voices.REALISTIC


def test_the_models_match_the_pins():
    assert set(PINS) == {'shared', *voices.MODEL_LANGUAGES}
    assert [m['id'] for m in MANIFEST['models']] == list(voices.MODEL_LANGUAGES)
    assert [m['id'] for m in MANIFEST['models'] if m['default']] == [voices.REALISTIC]
    for model in MANIFEST['models']:
        pinned = addon.pins(model['id'])
        assert sorted(p['path'] for p in pinned) == sorted(FILES)
        assert model['size_mb'] == round(sum(p['size'] for p in pinned) / 1e6)
        entry = PINS[model['id']]
        assert model['revision'] == entry['revision'] and model['repo'] == entry['repo']
    both = {(p['repo'], p['path']) for m in voices.MODEL_LANGUAGES for p in addon.pins(m)}
    assert MANIFEST['disk_required_mb'] >= sum(
        p['size'] for p in PINS['shared']['files'] + PINS[voices.PREVIEW]['files']
        + PINS[voices.REALISTIC]['files']) / 1e6 and len(both) == 17


def test_each_model_has_its_own_fingerprint():
    """The voice cache is keyed by the fingerprint in each model's manifests."""
    own = {m: {p['path'] for p in PINS[m]['files']} for m in voices.MODEL_LANGUAGES}
    for paths in own.values():
        assert {'runtime_manifest.json', 'registration/registration_manifest.json'} <= paths
    assert {p['path'] for p in PINS['shared']['files']}.isdisjoint(set().union(*own.values()))


def test_every_pin_is_a_checked_file():
    for model in voices.MODEL_LANGUAGES:
        paths = {p['path'] for p in addon.pins(model)}
        for pin in addon.pins(model):
            assert _SHA256.match(pin['sha256']) and pin['size'] > 0, pin
            assert re.fullmatch(r'[0-9a-f]{40}', pin['revision']), pin
            if pin['path'].endswith('.onnx'):
                assert pin['path'] + '.data' in paths   # each graph's weights come with it


def test_model_files_are_in_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    for model in voices.MODEL_LANGUAGES:
        files = addon.model_files(model)
        assert set(files) == set(FILES)
        assert all(str(path).startswith(str(tmp_path / 'audio8')) for path in files.values())
    # Preview's files stay where 26.11.1 downloaded them.
    preview = addon.model_files(voices.PREVIEW)
    assert '/Edge0--Audio8-TTS-Preview-0.6B-ONNX-INT4/818569c6b832118ad68d61bbd873abe250fcd68a/' in \
        preview['slow_ar_int4.onnx'].as_posix()


@pytest.mark.parametrize('tag, expected', [
    ('en-US', 'en'), ('EN_gb', 'en'), ('zh-CN', 'zh'), ('zh-Hans', 'zh'), ('zh-HK', 'yue'),
    ('yue', 'yue'), ('cmn', 'zh'), ('pt-BR', 'pt'), ('nb-NO', 'no'), ('in', 'id'), ('', ''),
])
def test_language_tags(tag, expected):
    assert voices.primary_language(tag) == expected


def test_test_clip_is_credited():
    credits = (REPO_ROOT / 'tests' / 'data' / 'README.md').read_text(encoding='utf-8')
    assert '`en-woman.wav`' in credits and (REPO_ROOT / 'tests' / 'data' / 'CC-BY-4.0.txt').is_file()
