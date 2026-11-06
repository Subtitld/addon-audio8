"""Write audio8_addon/pins.json: every file the add-on reads, pinned.

    python tools/pin_model.py [REVISION]

Without a revision, pins the repository's current head. Sizes and sha256 come
from Hugging Face's LFS metadata; the small files that are not in LFS are
downloaded and hashed.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from audio8_addon.engine import FILES  # noqa: E402

# The official ONNX release; Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4 redirects here.
REPO = 'Edge0/Audio8-TTS-Preview-0.6B-ONNX-INT4'


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main() -> int:
    revision = sys.argv[1] if len(sys.argv) > 1 else \
        json.loads(get(f'https://huggingface.co/api/models/{REPO}'))['sha']
    tree = {item['path']: item for item in json.loads(get(
        f'https://huggingface.co/api/models/{REPO}/tree/{revision}?recursive=1'))}
    files = []
    for path in FILES:
        meta = tree[path]
        lfs = meta.get('lfs') or {}
        sha = lfs.get('oid') or lfs.get('sha256')
        if not sha:
            data = get(f'https://huggingface.co/{REPO}/resolve/{revision}/{path}')
            if len(data) != meta['size']:
                raise SystemExit(f'{REPO}/{path}: got {len(data)} bytes, the tree says {meta["size"]}')
            sha = hashlib.sha256(data).hexdigest()
        files.append({'path': path, 'size': meta['size'], 'sha256': sha})
    out = {'audio8': {'repo': REPO, 'revision': revision, 'files': files}}
    print(f"{REPO}: {len(files)} files, {sum(f['size'] for f in files) / 1e6:.0f} MB at {revision}")
    (HERE / 'audio8_addon' / 'pins.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
