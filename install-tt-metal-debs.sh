#!/bin/bash
set -e

VERSION="0.66.0"
UBUNTU_VERSION=$(lsb_release -rs)
ARCH="amd64"
BASE_URL="https://github.com/tenstorrent/tt-metal/releases/download/v${VERSION}"

PACKAGES=(
    "tt-metalium_${VERSION}.ubuntu${UBUNTU_VERSION}_${ARCH}.deb"
    "tt-metalium-dev_${VERSION}.ubuntu${UBUNTU_VERSION}_${ARCH}.deb"
    "tt-metalium-jit_${VERSION}.ubuntu${UBUNTU_VERSION}_${ARCH}.deb"
    "tt-nn_${VERSION}.ubuntu${UBUNTU_VERSION}_${ARCH}.deb"
)

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading tt-metal v${VERSION} packages for Ubuntu ${UBUNTU_VERSION}..."
for pkg in "${PACKAGES[@]}"; do
    echo "  -> $pkg"
    wget -q --show-progress -P "$TMPDIR" "${BASE_URL}/${pkg}"
done

echo "Installing packages..."
sudo apt-get install -y "${PACKAGES[@]/#/$TMPDIR/}"

echo "Done."
