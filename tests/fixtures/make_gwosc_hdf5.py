#!/usr/bin/env python3
"""Write a small HDF5 file in GWOSC's bulk layout for tests.
Usage: make_gwosc_hdf5.py OUT.hdf5 DETECTOR GPS DURATION RATE BAD_FROM BAD_TO
Strain sample i is (i+1)*1e-21 except NaN (and DATA off) in seconds [10,20); CBC_CAT2 fails in seconds
[BAD_FROM,BAD_TO); a CBC hardware injection (NO_CBC_HW_INJ off) in seconds [DURATION-5,DURATION-3)."""
import sys
import h5py
import numpy as np

out, det, gps, dur, rate, bad0, bad1 = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:8])
h = (np.arange(dur * rate) + 1) * 1e-21
h[10 * rate:20 * rate] = np.nan
dqbits = ["DATA", "CBC_CAT1", "CBC_CAT2", "CBC_CAT3", "BURST_CAT1", "BURST_CAT2", "BURST_CAT3", "STOCH_CAT1", "CW_CAT1"]
injbits = ["NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_DETCHAR_HW_INJ", "NO_CW_HW_INJ", "NO_STOCH_HW_INJ"]
dq = np.full(dur, 0b111, dtype="u4")
dq[10:20] = 0
dq[bad0:bad1] &= ~np.uint32(0b100)
inj = np.full(dur, 0b11111, dtype="u4")
inj[dur - 5:dur - 3] &= ~np.uint32(1)
with h5py.File(out, "w") as f:
    f["meta/Detector"] = det
    f["meta/GPSstart"] = gps
    f["meta/Duration"] = dur
    s = f.create_dataset("strain/Strain", data=h, compression="gzip", chunks=(rate,))
    s.attrs["Xspacing"] = 1.0 / rate
    f.create_dataset("quality/simple/DQmask", data=dq, compression="gzip")
    f["quality/simple/DQShortnames"] = np.array(dqbits, dtype=object)
    f.create_dataset("quality/injections/Injmask", data=inj, compression="gzip")
    f["quality/injections/InjShortnames"] = np.array(injbits, dtype=object)
