"""Write audio8_addon/pins.json: every file the add-on reads, pinned.

    python tools/pin_model.py [PREVIEW_REVISION [REALISTIC_REVISION]]

Without revisions, pins each repository's current head. Sizes and sha256 come
from Hugging Face's LFS metadata; the small files that are not in LFS are
downloaded and hashed.

The two models share their codec and tokenizer: the Realistic export reuses
the Preview release's files byte for byte. Those are pinned once, from
Edge0's official repository (`shared`), and each model pins only its two
transformers and the manifests that carry its fingerprint. The script fails
if the Realistic repository's copies of the shared files ever differ.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from audio8_addon.engine import FILES  # noqa: E402
from audio8_addon.voices import PREVIEW, REALISTIC  # noqa: E402

# The official ONNX release; Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4 redirects here.
PREVIEW_REPO = 'Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4'
# Edge0 published no ONNX build of Realistic. This independent export follows
# the Preview release's recipe and layout; tools/verify_export.py checks it
# against Realistic's checkpoint and the official Preview graphs.
REALISTIC_REPO = 'stephvax/audio8-tts-realistic-0.6b-onnx-int4'
# Each model's own files; the rest of FILES is shared.
OWN = ('runtime_manifest.json', 'registration/registration_manifest.json',
       'slow_ar_int4.onnx', 'slow_ar_int4.onnx.data', 'fast_ar_int4.onnx', 'fast_ar_int4.onnx.data')


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def describe(repo: str, revision: str | None) -> tuple[str, dict]:
    """(revision, {path: {path, size, sha256}}) of every file in FILES."""
    revision = revision or json.loads(get(f'https://huggingface.co/api/models/{repo}'))['sha']
    tree = {item['path']: item for item in json.loads(get(
        f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1'))}
    files = {}
    for path in FILES:
        meta = tree[path]
        lfs = meta.get('lfs') or {}
        sha = lfs.get('oid') or lfs.get('sha256')
        if not sha:
            data = get(f'https://huggingface.co/{repo}/resolve/{revision}/{path}')
            if len(data) != meta['size']:
                raise SystemExit(f'{repo}/{path}: got {len(data)} bytes, the tree says {meta["size"]}')
            sha = hashlib.sha256(data).hexdigest()
        files[path] = {'path': path, 'size': meta['size'], 'sha256': sha}
    return revision, files


def main() -> int:
    args = sys.argv[1:] + [None, None]
    preview_rev, preview = describe(PREVIEW_REPO, args[0])
    realistic_rev, realistic = describe(REALISTIC_REPO, args[1])
    shared = [path for path in FILES if path not in OWN]
    for path in shared:
        if preview[path] != realistic[path]:
            raise SystemExit(f'{path} differs between {PREVIEW_REPO} and {REALISTIC_REPO}')
    out = {
        'shared': {'repo': PREVIEW_REPO, 'revision': preview_rev, 'files': [preview[p] for p in shared]},
        PREVIEW: {'repo': PREVIEW_REPO, 'revision': preview_rev, 'files': [preview[p] for p in OWN]},
        REALISTIC: {'repo': REALISTIC_REPO, 'revision': realistic_rev, 'files': [realistic[p] for p in OWN]},
    }
    for key, entry in out.items():
        size = sum(f['size'] for f in entry['files'])
        print(f"{key}: {entry['repo']}@{entry['revision'][:12]}, {len(entry['files'])} files, {size / 1e6:.0f} MB")
    (HERE / 'audio8_addon' / 'pins.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
