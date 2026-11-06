"""The voice the add-on offers, and the languages it speaks.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Audio8 TTS has no voices of its own and ships no preset clips: every line
clones a reference clip, the speaker's own from the video (or a clip named in
Configure).
"""

from __future__ import annotations

CLONE = 'audio8-clone'
SAMPLE_RATE = 44100

# Upstream's eleven recommended languages. The model takes no language tag:
# this list only says which lines to accept.
LANGUAGE_NAMES = {
    'zh': 'Chinese', 'yue': 'Cantonese', 'nl': 'Dutch', 'en': 'English', 'fr': 'French',
    'de': 'German', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'pl': 'Polish',
    'es': 'Spanish',
}
LANGUAGES = tuple(LANGUAGE_NAMES)
# Tags Subtitld or a user may send for a language under another code.
ALIASES = {'cmn': 'zh', 'zho': 'zh', 'chi': 'zh', 'jpn': 'ja', 'kor': 'ko', 'nld': 'nl', 'dut': 'nl'}


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    return [{
        'id': CLONE,
        'language': '*',
        'display_name': "Speaker's own voice (cloned from the video)",
        'sample_rate': SAMPLE_RATE,
        'requires': ['voice_ref_audio'],
    }]


def primary_language(tag) -> str:
    """`en-US` -> `en`, `zh-HK` -> `yue`... ; '' when there is none."""
    parts = str(tag or '').strip().lower().replace('_', '-').split('-')
    primary = ALIASES.get(parts[0], parts[0])
    if primary == 'zh' and len(parts) > 1 and parts[-1] in ('hk', 'mo', 'yue'):
        return 'yue'
    return primary
