#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
data="$root/data/gw"
catalogue="$data/allevents.json"
duration=4096
mode=noise
dry_run=0
duration_set=0
events_file=
declare -a requested=()

usage() {
    cat <<'EOF'
Usage:
  scripts/fetch_gw.sh [--dry-run] [--events-file FILE] [EVENT ...]
  scripts/fetch_gw.sh --test-events [--dry-run] [--duration 32]
  scripts/fetch_gw.sh --duration 32 [--dry-run] EVENT ...

No arguments downloads 4096-second H1/L1 noise files for the six events in
scripts/gw_noise_events.txt. --test-events selects catalogue BBHs with
p_astro >= 0.9 and source component masses from 10 to 80 solar masses, then
downloads their 32-second H1/L1 files. Event files contain one common event
name per line; blank lines and lines beginning with # are ignored.
Files are cached under data/gw/strain/<event>/ and incomplete downloads resume.
EOF
}

while (($#)); do
    case "$1" in
        --duration)
            (($# >= 2)) || { usage >&2; exit 2; }
            duration="$2"
            duration_set=1
            shift 2
            ;;
        --events-file)
            (($# >= 2)) || { usage >&2; exit 2; }
            events_file="$2"
            shift 2
            ;;
        --test-events)
            mode=test
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            requested+=("$1")
            shift
            ;;
    esac
done

case "$duration" in
    32|4096) ;;
    *) echo "duration must be 32 or 4096 seconds" >&2; exit 2 ;;
esac

if [[ "$mode" == test && $duration_set -eq 0 ]]; then duration=32; fi
if [[ "$mode" == noise && ( -n "$events_file" || ${#requested[@]} -gt 0 ) && $duration_set -eq 0 ]]; then duration=32; fi

mkdir -p "$data"
if [[ ! -s "$catalogue" ]]; then
    echo "get  $catalogue"
    curl -fsSL --retry 3 --retry-all-errors -o "$catalogue.part" \
        "https://gwosc.org/eventapi/json/allevents/"
    mv "$catalogue.part" "$catalogue"
fi

if [[ -n "$events_file" ]]; then
    [[ -r "$events_file" ]] || { echo "cannot read event list: $events_file" >&2; exit 2; }
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line//[[:space:]]/}"
        [[ -n "$line" ]] && requested+=("$line")
    done < "$events_file"
elif [[ ${#requested[@]} -eq 0 && "$mode" == noise ]]; then
    mapfile -t requested < "$root/scripts/gw_noise_events.txt"
elif [[ ${#requested[@]} -eq 0 && "$mode" == test ]]; then
    mapfile -t requested < <(jq -r '
        [.events | to_entries[] | .value]
        | group_by(.commonName)
        | .[]
        | select(any(.[]; .p_astro != null and .p_astro >= 0.9
            and .mass_1_source != null and .mass_1_source >= 10 and .mass_1_source <= 80
            and .mass_2_source != null and .mass_2_source >= 10 and .mass_2_source <= 80))
        | .[0].commonName
    ' "$catalogue")
fi

((${#requested[@]} > 0)) || { echo "event list is empty" >&2; exit 2; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

download_one() {
    local event="$1" detector="$2" gps="$3" url="$4"
    local destination="$data/strain/$event/${detector}-${duration}-${gps}.txt.gz"
    local partial="$destination.part"

    if [[ -s "$destination" ]]; then
        if gzip -t "$destination" 2>/dev/null; then
            echo "have $destination"
            return
        fi
        echo "cached file is not a valid gzip stream: $destination" >&2
        return 1
    fi
    if ((dry_run)); then
        printf 'would get %s\n  %s\n' "$destination" "$url"
        return
    fi

    mkdir -p "$(dirname "$destination")"
    echo "get  $url"
    if ! curl -fL --retry 3 --retry-all-errors -C - -o "$partial" "$url"; then
        gzip -t "$partial" 2>/dev/null || return 1
    fi
    gzip -t "$partial"
    mv "$partial" "$destination"
}

for event in "${requested[@]}"; do
    mapfile -t detail_urls < <(jq -r --arg name "$event" '
        [.events | to_entries[] | select(.value.commonName == $name)]
        | sort_by(.value.version) | reverse | .[].value.jsonurl
    ' "$catalogue")
    if ((${#detail_urls[@]} == 0)); then
        echo "skip $event: not found in cached GWOSC catalogue" >&2
        continue
    fi

    products=()
    for detail_url in "${detail_urls[@]}"; do
        [[ -n "$detail_url" ]] || continue
        detail="$tmp/detail.json"
        if ! curl -fsSL --retry 2 --retry-all-errors --max-time 30 -o "$detail" "$detail_url"; then
            continue
        fi
        mapfile -t products < <(jq -r --argjson duration "$duration" '
            [.events[] | .strain[]?
                | select(.sampling_rate == 4096 and .duration == $duration and .format == "txt")
                | select(.detector == "H1" or .detector == "L1")]
            | sort_by(.detector) | group_by(.detector)[] | last
            | [.detector, .GPSstart, .url] | @tsv
        ' "$detail")
        detectors="$(printf '%s\n' "${products[@]}" | cut -f1 | sort -u | paste -sd, -)"
        if [[ "$detectors" == "H1,L1" ]]; then break; fi
        products=()
    done

    if ((${#products[@]} != 2)); then
        echo "skip $event: no matching 4096 Hz, ${duration}-second H1/L1 text pair" >&2
        continue
    fi

    for product in "${products[@]}"; do
        IFS=$'\t' read -r detector gps url <<< "$product"
        download_one "$event" "$detector" "$gps" "$url"
    done
done