#!/usr/bin/env python3
"""Fetch GWOSC bulk strain (HDF5) and convert it into raw files q reads directly.

For each 4096 s file of a run, detector and GPS range this writes, under data/gw/bulk/<run>/:
  <det>-<gps>-<duration>.f32   strain, little-endian float32 (NaN where there is no data)
  <det>-<gps>-<duration>.dq    data-quality bitmask, one little-endian int32 per second
  <det>-<gps>-<duration>.inj   hardware-injection bitmask, one little-endian int32 per second
  <det>-<gps>-<duration>.json  detector, gps, duration, rate, and the names of the mask bits

Usage (run with the project's venv, which has h5py and numpy):
  .venv/bin/python scripts/gwosc_bulk.py fetch --run O4a --start GPS --end GPS [--detectors H1 L1]
        [--keep-hdf5] [--dry-run]
  .venv/bin/python scripts/gwosc_bulk.py convert FILE.hdf5 [--out DIR]
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://gwosc.org/api/v2/runs/{run}/strain-files?detector={det}"
# bulk files start on multiples of 4096 s and sit in directories named after the GPS time rounded down to a
# multiple of 2^20 s, so their URLs can be built directly (the API ignores GPS ranges and pages 20 files at a time)
URL = "https://gwosc.org/archive/data/{run}_4KHZ_R1/{dir}/{site}-{det}_GWOSC_{run}_4KHZ_R1-{gps}-4096.hdf5"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def names(dataset):
    return [b.decode() if isinstance(b, bytes) else str(b) for b in dataset[()]]


def convert(path, out_dir):
    """Convert one GWOSC HDF5 file; returns the base path of the outputs."""
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        det = f["meta/Detector"][()]
        det = det.decode() if isinstance(det, bytes) else str(det)
        gps = int(f["meta/GPSstart"][()])
        duration = int(f["meta/Duration"][()])
        strain = f["strain/Strain"]
        rate = int(round(1.0 / float(strain.attrs["Xspacing"])))
        h = strain[()].astype("<f4")
        dq = f["quality/simple/DQmask"][()].astype("<i4")
        inj = f["quality/injections/Injmask"][()].astype("<i4")
        info = {"detector": det, "gps": gps, "duration": duration, "rate": rate,
                "samples": int(h.size), "nan": int(np.isnan(h).sum()),
                "dqBits": names(f["quality/simple/DQShortnames"]),
                "injBits": names(f["quality/injections/InjShortnames"]),
                "source": os.path.basename(path)}
    if h.size != duration * rate or dq.size != duration or inj.size != duration:
        raise ValueError(f"{path}: unexpected sizes {h.size} {dq.size} {inj.size}")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{det}-{gps}-{duration}")
    for ext, arr in ((".f32", h), (".dq", dq), (".inj", inj)):
        tmp = base + ext + ".part"
        arr.tofile(tmp)
        os.replace(tmp, base + ext)
    with open(base + ".json.part", "w") as fh:
        json.dump(info, fh)
    os.replace(base + ".json.part", base + ".json")
    return base


def guessed(run, det, start, end):
    """The 4 kHz files a detector would have overlapping [start, end), by URL pattern (missing ones 404)."""
    first = (start // 4096) * 4096
    return [{"gps_start": g, "hdf5_url": URL.format(run=run, dir=(g >> 20) << 20, site=det[0], det=det, gps=g)}
            for g in range(first, end, 4096)]


def listing(run, det, start, end, cache_dir):
    """Every 4 kHz file of a detector overlapping [start, end) from the API (all pages, cached per run and detector)."""
    cache = os.path.join(cache_dir, f"listing-{det}.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            files = json.load(fh)
    else:
        url, files = API.format(run=run, det=det), []
        while url:
            with urllib.request.urlopen(url, timeout=60) as r:
                page = json.load(r)
            files += [x for x in page["results"] if x.get("sample_rate_kHz") == 4]
            url = page.get("next")
        with open(cache, "w") as fh:
            json.dump(files, fh)
    return [x for x in files if x["gps_start"] < end and x["gps_start"] + 4096 > start]


def download(url, dest):
    """Download with resumption into dest (via dest.part)."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(part, dest)


def fetch(args):
    out = os.path.join(ROOT, "data", "gw", "bulk", args.run)
    os.makedirs(out, exist_ok=True)
    for det in args.detectors:
        files = (listing(args.run, det, args.start, args.end, out) if args.use_api
                 else guessed(args.run, det, args.start, args.end))
        for x in files:
            base = os.path.join(out, f"{det}-{x['gps_start']}-4096")
            if os.path.exists(base + ".json"):
                print("have", base)
                continue
            hdf5 = os.path.join(out, os.path.basename(x["hdf5_url"]))
            if args.dry_run:
                print("would get", x["hdf5_url"])
                continue
            if not os.path.exists(hdf5):
                try:
                    download(x["hdf5_url"], hdf5)
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print("none", x["hdf5_url"], flush=True)
                        continue
                    raise
                print("got ", x["hdf5_url"], flush=True)
            print("conv", convert(hdf5, out), flush=True)
            if not args.keep_hdf5:
                os.remove(hdf5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--run", required=True)
    f.add_argument("--start", type=int, required=True)
    f.add_argument("--end", type=int, required=True)
    f.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    f.add_argument("--keep-hdf5", action="store_true")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--use-api", action="store_true", help="list files through the API (slow; cached) instead of by URL")
    c = sub.add_parser("convert")
    c.add_argument("file")
    c.add_argument("--out")
    a = p.parse_args()
    if a.cmd == "fetch":
        fetch(a)
    else:
        print(convert(a.file, a.out or os.path.dirname(os.path.abspath(a.file))))


if __name__ == "__main__":
    sys.exit(main())
