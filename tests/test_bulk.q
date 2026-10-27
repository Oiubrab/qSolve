/ open-search plan S1: GWOSC bulk HDF5 -> converted files -> q

.t.add[`bulk_segments;{
    ok:01111001110111b;
    s:.qs.gw.segments[ok;100f;2];
    .t.eq[s`start;101 107 111f];
    .t.eq[s`duration;4 3 3];
    .t.eq[count .qs.gw.segments[ok;100f;4];1];
    .t.eq[count .qs.gw.segments[000b;0f;1];0];
    .t.eq[.qs.gw.bit[5 4 1i;`a`b`c;`a];101b];
    .t.eq[.qs.gw.bit[5 4 1i;`a`b`c;`c];110b];
 }];

/ a fixture HDF5 in GWOSC's layout, converted by scripts/gwosc_bulk.py, read back by q; then H1 and L1 fixtures with
/ different vetoed stretches, intersected into analysable segments
.t.add[`bulk_convert_and_segments;{
    py:.t.root,"/.venv/bin/python";
    if[()~key hsym `$py; .t.skip "needs the project's .venv with h5py (python3 -m venv .venv; .venv/bin/pip install h5py numpy)"];
    tmp:getenv`TMPDIR; tmp:$[""~tmp;"/tmp";tmp],"/qsolve_bulk_",string .z.i;
    run:`TESTRUN; out:tmp,"/gw/bulk/TESTRUN";
    system "env mkdir -p ",out;
    mk:{[py;out;det;gps;b0;b1]
        f:out,"/",det,".hdf5";
        system "env ",py," -I ",.t.root,"/tests/fixtures/make_gwosc_hdf5.py ",f," ",det," ",(string gps)," 100 16 ",(string b0)," ",string b1;
        system "env ",py," -I ",.t.root,"/scripts/gwosc_bulk.py convert ",f;
        system "env rm -f ",f}[py;out];
    mk["H1";1000000000;40;50];
    mk["L1";1000000000;60;65];
    / point .qs.gw.dataDir at the temporary tree for this test
    d:.qs.gw.dataDir; .qs.gw.dataDir:{[tmp;x] tmp,"/gw"}[tmp];
    r:@[{
        b:.qs.gw.bulk x;
        .t.eq[b`detector;`H1]; .t.eq[b`gps;1000000000f]; .t.eq[b`rate;16];
        .t.eq[count b`h;1600];
        .t.eq[sum null b`h;160i];
        / float32: about 7 significant digits
        .t.near[((b`h) 0 1 1599)%1e-21*1 2 1600f;3#1f;1e-6];
        .t.eq["j"$sum .qs.gw.usable b;100-10+10+2];
        s:.qs.gw.analysable[`TESTRUN;1000000000f;1000000100f;5];
        / usable in both: H1 loses 10-20 (gap), 40-50 (CAT2), 95-97 (injection); L1 loses 10-20, 60-65, 95-97
        / (97-100 s is only 3 s, under the 5 s minimum)
        .t.eq[(s`start)-1000000000f;0 20 50 65f];
        .t.eq[s`duration;10 20 10 30];
        ix:.qs.gw.bulkIndex`TESTRUN;
        .t.eq[ix`detector;`H1`L1];
        `ok}; out,"/H1-1000000000-100"; {(`err;x)}];
    .qs.gw.dataDir:d;
    system "env rm -rf ",tmp;
    if[`err~first r; e:r 1; 'e];
 }];

/ the real O4a file, if fetched: matches what h5py reported
.t.add[`bulk_real_o4a;{
    base:.t.root,"/data/gw/bulk/O4a/H1-1368195072-4096";
    if[()~key hsym `$base,".json"; .t.skip "fetch with .venv/bin/python scripts/gwosc_bulk.py fetch --run O4a --start 1368195072 --end 1368199168 --detectors H1"];
    b:.qs.gw.bulk base;
    .t.eq[count b`h;16777216];
    .t.eq[sum null b`h;13000704i];
    .t.eq[sum .qs.gw.usable b;922i];
 }];
