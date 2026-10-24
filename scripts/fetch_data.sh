#!/usr/bin/env bash
# Download the benchmark datasets into data/ (next to this script's parent directory).
# Usage: scripts/fetch_data.sh [mnist] [fashion] [iris]    (no arguments fetches all three)
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
data="$root/data"
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(mnist fashion iris)
idx=(train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte)

# fetch URL DEST: download unless DEST already exists
fetch() {
    if [ -s "$2" ]; then echo "have $2"; return; fi
    echo "get  $1"
    curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

# idxset NAME BASEURL: the four gzipped IDX files of an MNIST-format dataset
idxset() {
    mkdir -p "$data/$1"
    for f in "${idx[@]}"; do
        if [ -s "$data/$1/$f" ]; then echo "have $data/$1/$f"; continue; fi
        fetch "$2/$f.gz" "$data/$1/$f.gz"
        gunzip -f "$data/$1/$f.gz"
    done
}

for w in "${want[@]}"; do
    case "$w" in
        mnist)   idxset mnist https://storage.googleapis.com/cvdf-datasets/mnist ;;
        fashion) idxset fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com ;;
        iris)    mkdir -p "$data/iris"; fetch https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data "$data/iris/iris.data" ;;
        *)       echo "unknown dataset: $w (expected mnist, fashion or iris)" >&2; exit 1 ;;
    esac
done
