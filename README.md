# Audio8 TTS add-on for Subtitld

Offline **voice-cloning text-to-speech** for
[Subtitld](https://github.com/Subtitld/subtitld), powered by
[Audio8 TTS Preview 0.6B](https://github.com/Audio8-AI/Audio8_TTS)
(Apache-2.0), run on [ONNX Runtime](https://onnxruntime.ai).

- Serves the `tts.synthesize` task over Subtitld's add-on protocol.
- **Clones a voice from a few seconds of speech.** Subtitld sends a clip of
  the speaker from the video, with its subtitles, and the dub speaks in their
  voice, in any of the add-on's languages, whatever language the clip is in.
- **Eleven languages, including Chinese, Cantonese, Japanese and Korean.**
  44.1 kHz audio.
- **CPU only, and slow.** About 0.4 to 0.5 times real time on a laptop CPU
  (a 5-second line takes 10 to 12 seconds), with about 1.6 GB of memory and
  up to 3 GB while it encodes a new speaker's clip. No GPU and no torch.
- **Offline after the first line.** The model downloads on the first line
  (1 GB), then nothing leaves the machine: no telemetry, no account.

## Languages

Chinese, Cantonese, Dutch, English, French, German, Italian, Japanese, Korean,
Polish and Spanish: the languages upstream recommends for this preview. The
model takes no language tag, so the language only decides which lines the
add-on accepts. A project in another language is refused with a message
that says so.

## The voice

There is one voice, the **speaker's own**: every line clones the clip of the
speaker Subtitld sends with it, or the fallback clip in Configure. The model
ships no preset voices.

A good clip is 3 to 30 seconds of one person speaking, with no music or other
voices; a longer one is cut to 30 seconds at a pause. Each clip is encoded
once (a few seconds) and its codes are cached, so later lines of the same
speaker start right away.

The model clones closest with the **exact words of the clip**. Subtitld sends
the subtitles spoken in it; without them the add-on clones from the clip's
sound alone, a little less close to the speaker. If the subtitles paraphrase
the speech, turn on *Don't send the subtitles of the reference clip* in
Configure.

## Options

Set these in **Settings ▸ Add-ons ▸ Audio8 TTS ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| CPU threads | automatic | `AUDIO8_THREADS` | 0 picks half the logical processors, at most 8. |
| Don't send the subtitles of the reference clip | off | (read by Subtitld) | Clone from the clip's sound alone. |
| Fallback reference clip | none | `AUDIO8_VOICE_REF_AUDIO` | Cloned when Subtitld has no clip of the speaker to send. |
| What the fallback clip says | none | `AUDIO8_VOICE_REF_TEXT` | Its exact words; empty to clone from its sound alone. |

Subtitld stretches a dub to fit its subtitle itself, so the add-on ignores
the rate.

## How it runs

The official ONNX release splits the model in four graphs: a codec encoder
for the reference clip, a "slow" transformer (24 layers, 4-bit weights) that
gives each frame's first codebook, a "fast" one (4 layers) that gives the
other nine, and a codec decoder from the codes to the waveform. The add-on
runs the loop between them, a port of upstream's `onnx_runtime`
(`audio8_addon/engine.py`), with upstream's sampling. With the same voice,
line and seed it gives the same codes as upstream's runtime.

- `audio8_addon/text.py`: upstream's text cleanup and prompt, and the Qwen2
  byte-level BPE, read from the model's tokenizer.json. `tests/test_text.py`
  checks it against Hugging Face's `tokenizers`.
- `audio8_addon/engine.py`: the graphs, the sampling loop, a cap on a line's
  length so a line that runs on stops, and the voice cache.

Measured on an Intel i7-13700T (8 threads): about 45 ms of the two
transformers and 37 ms of the decoder per 46 ms of audio. The decoder attends
to the whole line, so it cannot usefully run while the transformers do.

Quality, with one line in each of the eleven languages, scored with Whisper
(medium) and Resemblyzer: cloned from a speaker of the same language, a word
error rate of 0.045 and a speaker similarity of 0.89; cloned from an English
speaker, as when dubbing, 0.064 and 0.88. Whisper hears Cantonese as written
Chinese, so its Cantonese score (0.42) says little. None of it was checked by
ear in every language.

## What is bundled, what is downloaded

| What | Size | Licence | Source |
|------|------|---------|--------|
| Add-on, ONNX Runtime, numpy (bundled) | see the release | Apache-2.0; MIT; BSD-3-Clause | this repository, PyPI |
| Audio8 TTS Preview 0.6B, ONNX INT4 (downloaded) | 1,015 MB | Apache-2.0 | [Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4) |

The model goes to Subtitld's model folder (`$SUBTITLD_MODELS_DIR/audio8`), at
the revision in `audio8_addon/pins.json`, each file checked by its sha256. An
interrupted download resumes. `$HF_ENDPOINT` points it at a mirror.

Each release archive carries `THIRD_PARTY_LICENSES/README.md`, which names
the notice for every native file in it (the build fails if one has none), and
upstream's LICENSE and NOTICE. The Linux archive carries libquadmath
(LGPL-2.1), from numpy's wheel; its source is attached to the same release.

Clone a voice only with its owner's consent, and say that the dub is
synthetic where that matters.

## Limits

- **Slow.** Plan for two to three seconds of work per second of speech.
- **A preview model.** Upstream calls it a preview with limited language
  coverage.
- **Long lines.** A line and its reference share 2,048 positions: about 60
  seconds of speech with a 10-second reference.
- **Platforms.** Releases ship Linux x86_64, macOS arm64 and Windows x86_64.
- **One line at a time.** A cancel stops the line between frames, or stops
  the download, which resumes on the next line.

## Build

```bash
pip install -c constraints.txt -e '.[build,dev]'
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/audio8-addon/audio8-addon --self-test
AUDIO8_TEST_BINARY=$PWD/dist/audio8-addon/audio8-addon pytest tests/test_installed.py
```

`tools/build_linux.sh` builds the Linux archive in AlmaLinux 8 (glibc 2.28),
as the release does. To move to another revision of the model, run
`tools/pin_model.py` and check that the tests still pass.

A tag `vYY.MM.N` builds, tests and publishes every platform
(`.github/workflows/release.yml`).
