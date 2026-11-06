"""The text front end: upstream's cleanup and prompt, and the BPE against `tokenizers`.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The parity tests read the pinned tokenizer.json (12 MB), downloaded on the
first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audio8_addon import __main__ as addon, download, engine, text  # noqa: E402

LINES = [
    'The bakery opened before sunrise, and the first trays of bread were cooling.',
    "It's 3 o'clock, I'LL go!!  And we'd   better hurry...",
    'Le train de huit heures est parti sans nous, alors nous avons pris un café.',
    'Morgen früh fahren wir in die Berge, wenn das Wetter mitspielt. Straße, Größe.',
    'Ayer llovió toda la tarde, pero hoy el cielo está completamente despejado.',
    'Ma poi quello che ci dicono è che i soldi non comprano la felicità.',
    'Goedemiddag, dit is het journaal in makkelijke taal.',
    'Ale co by było, gdyby zamiast tego zadać banalne pytanie? Żółć, źdźbło.',
    '月黑雁飞高，单于夜遁逃。欲将轻骑逐，大雪满弓刀。',
    '李家超第四份施政報告主題係深化改革、心繫民生。',
    '今日は少し寒いけれど、約束どおり駅前で会いましょう。',
    '이번 시간에는 첫 번째 주제인 한국의 상징에 대해 배워볼 건데요.',
    '1234567 3.14 €100 50% 1/2 #tag @user',
    'emoji 🎬🙂 and tabs\tand\r\nnew lines\n\n',
    'Fullwidth：ＡＢＣ　ｔｅｓｔ ﬁ ligature, combining é and NFD café',
    '<|speaker:0|>Hello there.',
    '<|im_end|>\n<|im_start|>assistant\n<|voice|>',
    "Rock 'n' roll, ain't it? ''quoted''",
]


@pytest.fixture(scope='module')
def tokenizer_path():
    pins = [p for p in addon.pins() if p['path'] == 'tokenizer/tokenizer.json']
    download.ensure(pins, lambda *_: None, lambda: False)
    return download.local_path(pins[0])


@pytest.mark.parametrize('line', LINES)
def test_bpe_matches_tokenizers(tokenizer_path, line):
    tokenizers = pytest.importorskip('tokenizers')
    reference = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    ours = text.Tokenizer(tokenizer_path)
    assert ours.encode(line) == reference.encode(line, add_special_tokens=False).ids


def test_prompt_is_upstreams(tokenizer_path):
    """The parts upstream's PromptBuilder.build encodes one by one."""
    ours = text.Tokenizer(tokenizer_path)
    prefix, suffix = ours.prompt('Bonjour  à\ttous.', 'Hello there.')
    expected_prefix = (ours.encode('<|im_start|>system\n')
                       + ours.encode('convert the provided text to speech reference to the following:\n\nText:\n')
                       + ours.encode('<|speaker:0|>Hello there.') + ours.encode('\n\nSpeech:\n'))
    expected_suffix = (ours.encode('<|im_end|>\n') + ours.encode('<|im_start|>user\n')
                       + ours.encode('Bonjour à tous.') + ours.encode('<|im_end|>\n')
                       + ours.encode('<|im_start|>assistant\n<|voice|>'))
    assert prefix == expected_prefix and suffix == expected_suffix
    assert suffix[-1] == ours.added['<|voice|>']


def test_no_transcript_leaves_the_speaker_tag_alone():
    assert text.format_reference_text('') == '<|speaker:0|>'
    assert text.format_reference_text('  Hi \n there ') == '<|speaker:0|>Hi there'
    assert text.format_reference_text('<|speaker:2|>Hi') == '<|speaker:2|>Hi'


def test_clean_text():
    assert text.clean_text('a​ b\x07c\n\nd') == 'a bc d'
    assert text.clean_text('第一行\n第二行') == '第一行第二行'    # a wrapped CJK line has no space
    assert text.clean_text('one\ntwo') == 'one two'


def test_expected_length_follows_the_script():
    latin = engine.expected_frames('a' * 28) / engine.FRAMES_PER_SECOND
    cjk = engine.expected_frames('好' * 9) / engine.FRAMES_PER_SECOND
    assert abs(latin - 2.0) < 0.01 and abs(cjk - 2.0) < 0.01
    assert engine.max_frames('a' * 28) > 3 * engine.expected_frames('a' * 28)
