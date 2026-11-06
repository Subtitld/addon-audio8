"""Text front end: upstream's text cleanup, the prompt, and Qwen's BPE.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Upstream (Audio8-AI/Audio8_TTS, onnx_runtime/arktts_runtime/prompt.py,
Apache-2.0) tokenizes with Hugging Face's `tokenizers` library. The add-on
reads the same tokenizer.json with the small byte-level BPE below instead, so
no native library (and no licence notice for its statically linked crates)
ships for it. tests/test_text.py checks it against `tokenizers`.

The tokenizer is Qwen2's: NFC, then a regex pre-tokenizer, then GPT-2's
byte-to-unicode mapping and BPE. Its added tokens (<|im_start|>, <|voice|>,
the 4,096 <|semantic:N|> tokens...) are matched in the raw text first.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Upstream's text cleanup (prompt.py)
# ---------------------------------------------------------------------------
_CJK_RANGES = (
    'ᄀ-ᇿ⺀-⿟　-〿぀-ヿ㄀-ㇿ'
    '㐀-䶿一-鿿ꥠ-꥿가-힣ힰ-퟿豈-﫿'
    '︰-﹏！-ﾟ\U00020000-\U0002fa1f'
)
_CJK_CHARACTER_RE = re.compile(f'[{_CJK_RANGES}]')
_LINE_BREAK_RE = re.compile('[\r\n\v\f\x1c-\x1e\x85  ]')
_SPEAKER_TAG_RE = re.compile(r'<\|speaker:\d+\|>')


def _normalize_whitespace(text: str) -> str:
    """Runs of whitespace become one space, or nothing for a line break
    between two CJK characters (a wrapped Chinese line has no space)."""
    def replace(match: re.Match) -> str:
        left = text[match.start() - 1] if match.start() else ''
        right = text[match.end()] if match.end() < len(text) else ''
        if (_LINE_BREAK_RE.search(match.group())
                and _CJK_CHARACTER_RE.fullmatch(left) and _CJK_CHARACTER_RE.fullmatch(right)):
            return ''
        return ' '

    return re.sub(r'\s+', replace, text).strip()


def clean_text(text: str) -> str:
    """Control and format characters dropped, whitespace collapsed."""
    value = ''.join(char if char.isspace() else '' if unicodedata.category(char).startswith('C') else char
                    for char in str(text))
    return _normalize_whitespace(value)


def format_reference_text(text: str) -> str:
    """The reference transcript behind its speaker tag. Without a transcript,
    the tag alone: the model then clones from the clip's sound only."""
    text = clean_text(text)
    return text if _SPEAKER_TAG_RE.search(text) else f'<|speaker:0|>{text}'


PROMPT_PREFIX = ('<|im_start|>system\n',
                 'convert the provided text to speech reference to the following:\n\nText:\n')
PROMPT_SPEECH = '\n\nSpeech:\n'
PROMPT_SUFFIX = ('<|im_end|>\n', '<|im_start|>user\n')
PROMPT_END = ('<|im_end|>\n', '<|im_start|>assistant\n<|voice|>')


# ---------------------------------------------------------------------------
# Byte-level BPE
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """GPT-2's printable stand-in for each byte."""
    keep = (list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1))
            + list(range(ord('®'), ord('ÿ') + 1)))
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars)))


def _char_class(points: list[int]) -> str:
    out, start, prev = [], points[0], points[0]
    for cp in points[1:] + [-1]:
        if cp == prev + 1:
            prev = cp
            continue
        out.append(re.escape(chr(start)) if start == prev
                   else f'{re.escape(chr(start))}-{re.escape(chr(prev))}')
        start = prev = cp
    return ''.join(out)


@lru_cache(maxsize=1)
def _qwen_pattern() -> re.Pattern:
    r"""Qwen2's pre-tokenizer regex. Python's re has no \p{L} or \p{N}, so the
    two classes are built from the Unicode database (once, about 0.3 s)."""
    letters, numbers = [], []
    for cp in range(sys.maxunicode + 1):
        category = unicodedata.category(chr(cp))
        if category[0] == 'L':
            letters.append(cp)
        elif category[0] == 'N':
            numbers.append(cp)
    L, N = _char_class(letters), _char_class(numbers)
    return re.compile(
        rf"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n{L}{N}]?[{L}]+|[{N}]| ?[^\s{L}{N}]+[\r\n]*"
        rf"|\s*[\r\n]+|\s+(?!\S)|\s+")


class Tokenizer:
    """tokenizer.json of Audio8 TTS Preview (Qwen2's BPE plus its added tokens)."""

    def __init__(self, path: Path):
        spec = json.loads(Path(path).read_text(encoding='utf-8'))
        model = spec['model']
        if model.get('type') != 'BPE':
            raise ValueError(f'{path}: not a BPE tokenizer')
        self.vocab: dict[str, int] = model['vocab']
        merges = [tuple(m.split(' ', 1)) if isinstance(m, str) else tuple(m) for m in model['merges']]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.added = {t['content']: t['id'] for t in spec.get('added_tokens', [])}
        # Longest first, so a longer added token wins over its prefix.
        names = sorted(self.added, key=len, reverse=True)
        self._split = re.compile('(' + '|'.join(re.escape(n) for n in names) + ')') if names else None
        self._cache: dict[str, list[int]] = {}

    def _merge(self, word: str) -> list[str]:
        parts = list(word)
        while len(parts) > 1:
            best = min(range(len(parts) - 1),
                       key=lambda i: self.ranks.get((parts[i], parts[i + 1]), sys.maxsize))
            pair = (parts[best], parts[best + 1])
            if pair not in self.ranks:
                break
            merged, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts

    def _word_ids(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = []
        for token in self._merge(word):
            token_id = self.vocab.get(token)
            if token_id is None:
                raise ValueError(f'no token for {token!r}')
            ids.append(token_id)
        if len(self._cache) < 50000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Ids of `text` without special tokens added around it, as
        `tokenizers`' encode(text, add_special_tokens=False).ids."""
        pattern, to_unicode = _qwen_pattern(), _byte_to_unicode()
        pieces = self._split.split(text) if self._split is not None else [text]
        ids: list[int] = []
        for index, piece in enumerate(pieces):
            if not piece:
                continue
            if index % 2 == 1:
                ids.append(self.added[piece])
                continue
            for word in pattern.findall(unicodedata.normalize('NFC', piece)):
                ids.extend(self._word_ids(''.join(to_unicode[b] for b in word.encode('utf-8'))))
        return ids

    def prompt(self, target_text: str, reference_text: str) -> tuple[list[int], list[int]]:
        """(ids before the reference's speech, ids after it): upstream's
        PromptBuilder.build, without the speech codes in between."""
        prefix = [*PROMPT_PREFIX, format_reference_text(reference_text), PROMPT_SPEECH]
        suffix = [*PROMPT_SUFFIX, clean_text(target_text), *PROMPT_END]
        return ([i for part in prefix for i in self.encode(part)],
                [i for part in suffix for i in self.encode(part)])
