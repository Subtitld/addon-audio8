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

# The languages each model speaks, from its model card. The model takes no
# language tag: these lists only say which lines to accept, and which model a
# line needs.
LANGUAGE_NAMES = {
    'ar': 'Arabic', 'yue': 'Cantonese', 'zh': 'Chinese', 'nl': 'Dutch', 'en': 'English',
    'fi': 'Finnish', 'fr': 'French', 'de': 'German', 'hi': 'Hindi', 'id': 'Indonesian',
    'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'ms': 'Malay', 'no': 'Norwegian',
    'pl': 'Polish', 'pt': 'Portuguese', 'ru': 'Russian', 'es': 'Spanish', 'sv': 'Swedish',
    'th': 'Thai', 'tr': 'Turkish', 'vi': 'Vietnamese',
}
REALISTIC = 'realistic-0.6b-int4'
PREVIEW = 'preview-0.6b-int4'
MODEL_LANGUAGES = {
    REALISTIC: tuple(LANGUAGE_NAMES),
    PREVIEW: ('yue', 'zh', 'nl', 'en', 'fr', 'de', 'it', 'ja', 'ko', 'pl', 'es'),
}
LANGUAGES = MODEL_LANGUAGES[REALISTIC]
# Tags Subtitld or a user may send for a language under another code.
ALIASES = {
    'cmn': 'zh', 'zho': 'zh', 'chi': 'zh', 'jpn': 'ja', 'kor': 'ko', 'nld': 'nl', 'dut': 'nl',
    'ara': 'ar', 'fin': 'fi', 'hin': 'hi', 'ind': 'id', 'in': 'id', 'msa': 'ms', 'may': 'ms',
    'zsm': 'ms', 'nor': 'no', 'nb': 'no', 'nob': 'no', 'nn': 'no', 'nno': 'no', 'por': 'pt',
    'rus': 'ru', 'swe': 'sv', 'tha': 'th', 'tur': 'tr', 'vie': 'vi', 'eng': 'en', 'fra': 'fr',
    'fre': 'fr', 'deu': 'de', 'ger': 'de', 'ita': 'it', 'pol': 'pl', 'spa': 'es',
}


def model_for(language: str, chosen: str) -> str:
    """The model that speaks `language`: the chosen one if it can, else Realistic."""
    return chosen if language in MODEL_LANGUAGES.get(chosen, ()) else REALISTIC


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
