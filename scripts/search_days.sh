#!/usr/bin/env bash
# Search whole days of GWOSC bulk data with a trained CNN (open-search plan): fetch and convert each day, then run
# bench/gw_bulk_scan.q on it. The next day downloads while the current one is searched.
#   scripts/search_days.sh MODEL SLIDES RUN:GPS_START [RUN:GPS_START ...]
#   e.g. scripts/search_days.sh models/gw_cnn_4.qsm 10 O3a:1246492800 O2:1184601600
# Each day is [GPS_START, GPS_START + 86400). Logs and results go to data/gw/search/.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
model=$1; slides=$2; shift 2
out="$root/data/gw/search"; mkdir -p "$out"
name=$(basename "$model" .qsm)

fetch() {
    local run=${1%%:*} start=${1##*:}
    "$root/.venv/bin/python" -I "$root/scripts/gwosc_bulk.py" fetch --run "$run" --start "$start" \
        --end $((start + 86400)) > "$out/fetch-$run-$start.log" 2>&1
}

search() {
    local run=${1%%:*} start=${1##*:}
    echo "$(date +%T) searching $run $start"
    (cd "$root" && q bench/gw_bulk_scan.q -s 4 -q -run "$run" -start "$start" -stop $((start + 86400)) \
        -slides "$slides" -top 25 -model "$model" > "$out/$run-$start-$((start + 86400))-$name.log" 2>&1) ||
        echo "  search failed: see $out/$run-$start-$((start + 86400))-$name.log"
}

prev=""
for day in "$@"; do
    echo "$(date +%T) fetching $day"
    fetch "$day" &
    pid=$!
    if [ -n "$prev" ]; then search "$prev"; fi
    wait "$pid" || echo "  fetch failed: see $out/fetch-${day%%:*}-${day##*:}.log"
    prev=$day
done
search "$prev"
echo "$(date +%T) done"
