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
    assert set(fields) == {'threads', 'skip_ref_text', 'voice_ref_audio', 'voice_ref_text'}
    assert fields['skip_ref_text']['type'] == 'bool'   # the host reads it, by this key


def test_the_model_matches_the_pins():
    [model] = MANIFEST['models']
    [entry] = PINS.values()
    pinned = addon.pins()
    assert [p['path'] for p in pinned] == list(FILES)
    assert model['size_mb'] == round(sum(p['size'] for p in pinned) / 1e6)
    assert model['revision'] == entry['revision'] and model['repo'] == entry['repo']
    assert MANIFEST['disk_required_mb'] >= model['size_mb']


def test_every_pin_is_a_checked_file():
    paths = {p['path'] for p in addon.pins()}
    for pin in addon.pins():
        assert _SHA256.match(pin['sha256']) and pin['size'] > 0, pin
        assert re.fullmatch(r'[0-9a-f]{40}', pin['revision']), pin
        if pin['path'].endswith('.onnx'):
            assert pin['path'] + '.data' in paths   # each graph's weights come with it


def test_model_files_are_in_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    files = addon.model_files()
    assert set(files) == set(FILES)
    assert all(str(path).startswith(str(tmp_path / 'audio8')) for path in files.values())


@pytest.mark.parametrize('tag, expected', [
    ('en-US', 'en'), ('EN_gb', 'en'), ('zh-CN', 'zh'), ('zh-Hans', 'zh'), ('zh-HK', 'yue'),
    ('yue', 'yue'), ('cmn', 'zh'), ('', ''),
])
def test_language_tags(tag, expected):
    assert voices.primary_language(tag) == expected


def test_test_clip_is_credited():
    credits = (REPO_ROOT / 'tests' / 'data' / 'README.md').read_text(encoding='utf-8')
    assert '`en-woman.wav`' in credits and (REPO_ROOT / 'tests' / 'data' / 'CC-BY-4.0.txt').is_file()
