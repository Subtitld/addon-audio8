#!/usr/bin/env bash
# Build the Linux frozen add-on, its archive and the libquadmath source the
# release attaches, inside an AlmaLinux 8 container.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# The bundle carries the build interpreter's libpython, so it runs on any
# glibc at least as new as the build's: AlmaLinux 8, glibc 2.28, as for the
# other Subtitld add-ons, with the distribution's Python 3.12
# (tools/linux-build.Dockerfile).
#
#     tools/build_linux.sh
#
# writes, into the repository, owned by you:
#     audio8-<version>-linux-x86_64.zip              (+ .sha256)
#     audio8-<version>-libquadmath-source.tar.gz     (+ .sha256)  LGPL-2.1, numpy's
#
# Needs Docker and the network.

set -euo pipefail

BUILD_IMAGE="${BUILD_IMAGE:-subtitld-addon-build:almalinux8-py312}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = .\(.*\).$/\1/p' "$ROOT/audio8_addon/__init__.py")"
ARCHIVE="audio8-${VERSION}-linux-x86_64.zip"
QUADMATH_SRC="audio8-${VERSION}-libquadmath-source.tar.gz"
# pip's cache and the downloaded source packages survive between builds.
PIP_CACHE="${PIP_CACHE:-$HOME/.cache/subtitld-addon-build-pip}"
SOURCES_CACHE="${SOURCES_CACHE:-$HOME/.cache/subtitld-addon-build-sources}"
mkdir -p "$PIP_CACHE" "$SOURCES_CACHE"

# No build context: the Dockerfile copies nothing in.
docker build -q -t "$BUILD_IMAGE" - < "$ROOT/tools/linux-build.Dockerfile" >/dev/null

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$ROOT:/src" \
    -v "$PIP_CACHE:/tmp/.cache/pip" \
    -v "$SOURCES_CACHE:/tmp/.cache/sources" \
    -w /src \
    "$BUILD_IMAGE" \
    bash -euo pipefail -c "
        command -v nm >/dev/null || { echo 'nm is required for the libexpat check' >&2; exit 1; }
        # The distribution's interpreter: /usr/local/bin/python3.12 is the image's static one.
        /usr/bin/python3.12 -m venv /tmp/build-venv
        . /tmp/build-venv/bin/activate
        pip install -q --upgrade pip
        pip install -q -c constraints.txt '.[build]'

        # A failed build must not leave older outputs that look current.
        rm -rf build/pyinstaller dist release-staging '$ARCHIVE' '$ARCHIVE.sha256' \\
            '$QUADMATH_SRC' '$QUADMATH_SRC.sha256'

        pyinstaller pyinstaller.spec --noconfirm --log-level WARN --workpath build/pyinstaller
        dist/audio8-addon/audio8-addon --self-test

        # Every LGPL library in the bundle must be a build whose source is attached.
        python tools/build_libquadmath_source.py --check-bundle dist/audio8-addon

        mkdir -p release-staging/bin
        cp -R dist/audio8-addon/. release-staging/bin/
        cp manifest.json LICENSE README.md release-staging/
        python tools/collect_licences.py --bundle dist/audio8-addon \\
            --out release-staging/THIRD_PARTY_LICENSES --needs-source /tmp/sources.txt
        python tools/make_archive.py release-staging '$ARCHIVE'
        sha256sum '$ARCHIVE' > '$ARCHIVE.sha256'

        python tools/build_libquadmath_source.py --srpm-dir /tmp/.cache/sources --out '$QUADMATH_SRC'
        sha256sum '$QUADMATH_SRC' > '$QUADMATH_SRC.sha256'
    "

echo "built $ARCHIVE and $QUADMATH_SRC"
cat "$ROOT/$ARCHIVE.sha256" "$ROOT/$QUADMATH_SRC.sha256"
