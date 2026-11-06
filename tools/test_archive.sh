#!/usr/bin/env bash
# Run tests/test_installed.py against the release archive, in a clean container.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# The archive is unpacked the way Subtitld's installer does it (zipfile's
# extractall, then the executable bit), into a distribution that has none of the
# build's packages. The container gets only what the tests drive the binary
# with: Python and pytest. Without AUDIO8_TEST_MODELS_DIR the binary downloads
# the model itself (1 GB), which also checks that it finds the system's CA
# certificates.
#
#     tools/test_archive.sh fedora:latest
#     AUDIO8_TEST_MODELS_DIR=~/.cache/subtitld/models tools/test_archive.sh ubuntu:22.04
#
# Needs Docker and the network.

set -euo pipefail

IMAGE="${1:?usage: tools/test_archive.sh <image> [archive]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = .\(.*\).$/\1/p' "$ROOT/audio8_addon/__init__.py")"
ARCHIVE="$(realpath "${2:-$ROOT/audio8-${VERSION}-linux-x86_64.zip}")"
[ -f "$ARCHIVE" ] || { echo "no archive at $ARCHIVE; run tools/build_linux.sh" >&2; exit 1; }

MOUNTS=(-v "$ARCHIVE:/artifact/addon.zip:ro" -v "$ROOT/tests:/repo/tests:ro")
MODELS=/tmp/models
if [ -n "${AUDIO8_TEST_MODELS_DIR:-}" ]; then
    MOUNTS+=(-v "$(realpath "$AUDIO8_TEST_MODELS_DIR"):/models:ro")
    MODELS=/models
fi

docker run --rm -e "SUBTITLD_MODELS_DIR=$MODELS" "${MOUNTS[@]}" "$IMAGE" bash -euo pipefail -c '
    if command -v apt-get >/dev/null; then
        apt-get update -qq >/dev/null
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates python3 python3-pytest >/dev/null
    elif command -v dnf >/dev/null; then
        dnf install -y -q python3 python3-pytest >/dev/null
    elif command -v pacman >/dev/null; then
        pacman -Sy --noconfirm archlinux-keyring >/dev/null
        pacman -Syu --noconfirm --needed python python-pytest >/dev/null
    else
        echo "no supported package manager in this image" >&2; exit 1
    fi
    . /etc/os-release
    echo "image: $PRETTY_NAME, $(ldd --version | head -1)"
    python3 - <<PY
import os, stat, zipfile
zipfile.ZipFile("/artifact/addon.zip").extractall("/opt/addon")
exe = "/opt/addon/bin/audio8-addon"
os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
PY
    cd /tmp
    AUDIO8_TEST_BINARY=/opt/addon/bin/audio8-addon python3 -m pytest -q -rs -p no:cacheprovider /repo/tests/test_installed.py
'
