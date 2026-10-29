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

/ the search's whitening (S2.1) matches the noise library's, chunk for chunk, and zeroes chunks with missing data
.t.add[`bulk_whiten_stream_matches_noise;{
    system "S 3"; r:256; n:r*200;
    h:1e-21*(n?1f)-0.5; h:@[h;(150*r)+til r;:;0n];
    s:`gps`rate`h!(1e9;r;h);
    o:`rate`chunk`margin`guard!(256;16;2;0);
    a:.qs.gw.whitenStream[s;o];
    b:.qs.gw.noiseChunks[s;`float$();o];
    .t.eq[a`gps;1e9+2];
    .t.eq[a`rate;256];
    keep:r*12;
    .t.eq[count a`w;keep*1+(n-r*16) div keep];
    / the chunks the library kept appear in the stream at their own times; the others are zero and not ok
    i:"j"$r*(b`gps)-a`gps;
    .t.near[raze {[a;keep;i] a[`w] i+til keep}[a;keep] each i;"f"$raze b`w;1e-6];
    .t.eq["j"$sum not a`ok;keep*(count[a`w] div keep)-count b];
    .t.ok[all 0=(a`w) where not a`ok;"zeroed chunks"];
 }];

/ the stream search (S2.1): skips windows touching unusable data, finds a burst present in both detectors at zero
/ lag, and a slide of L1 moves where the coincidence appears
.t.add[`bulk_search_streams;{
    system "S 8"; r:256; n:r*40;
    mk:{[r;n] `gps`rate`w`ok!(1e9;r;.qs.data.normal n;n#1b)}[r];
    a:mk n; b:mk n;
    burst:(20*r)+til r div 4;
    a[`w;burst]+:6*.qs.data.normal count burst; b[`w;burst]+:6*.qs.data.normal count burst;
    a[`ok;(30*r)+til 2*r]:0b;
    / stand in for the model with each window's energy (a stub .qs.predictRaw), restored afterwards
    p:.qs.predictRaw; .qs.predictRaw:{[m;X] enlist each 0.001*{x$x} each X};
    res:.qs.gw.searchStreams[();(a;b);`step`sep`merger!(0.25;2;0.5)]; t:res`triggers;
    ts:.qs.gw.searchStreams[();(a;b);`step`sep`merger`shift!(0.25;2;0.5;5)]`triggers;
    / scoring in small blocks gives the same triggers
    bk:.qs.gw.searchBlock; .qs.gw.searchBlock:7;
    tb:.qs.gw.searchStreams[();(a;b);`step`sep`merger!(0.25;2;0.5)]`triggers;
    .qs.gw.searchBlock:bk;
    .qs.predictRaw:p;
    / no window overlaps 30-32 s: of the 157 windows 0-39 s, 11 (29.25-31.75 s) are skipped
    .t.near[res`seconds;0.25*157-11;1e-9];
    .t.ok[not any ((t`gps)-1e9) within 29.01 31.99;"windows touching unusable data were scored"];
    best:t first idesc t`logit;
    .t.ok[((best`gps)-1e9) within 19.25 20;"loudest window starts at ",string (best`gps)-1e9];
    .t.near[best`merger;0.5+best`gps;1e-9];
    .t.near[best`score;reciprocal 1+exp neg best`logit;1e-12];
    / the amplitudes are each detector's largest |w| in the window
    i:"j"$256*(best`gps)-1e9;
    .t.near[best`ampH1`ampL1;{[i;s] max abs s[`w] i+til 256}[i] each (a;b);1e-12];
    .t.ok[all 2<=1_deltas t`gps;"triggers at least 2 s apart"];
    .t.ok[(max ts`logit)<0.8*best`logit;"a slid L1 should lose the coincidence"];
    .t.eq[tb;t];
 }];

/ false-alarm rates from the background, without vetoed triggers or an excluded event slid against noise
.t.add[`bulk_far;{
    bg:([] merger:100 200 300 400 500f; shift:7 7 7 14 14f; logit:1 2 3 9 4f; vetoed:00001b);
    trig:([] merger:10 20 30f; logit:2.5 0.5 10f);
    f:.qs.gw.far[trig;bg;3600f;`float$()];
    .t.eq[f`far;2 4 1f];
    .t.eq[f`bound;001b];
    / the 9 is an event at GPS 414 seen through L1 (merger+shift): excluded, so 3 is the loudest left
    g:.qs.gw.far[trig;bg;3600f;enlist 414.5];
    .t.eq[g`far;1 3 1f];
    .t.eq[.qs.gw.far[trig;bg;3600f;enlist 400.2]`far;1 3 1f];
 }];

/ the matched-filter follow-up (S4.1) on synthetic data: a template-shaped signal in white noise, in L1 3 ms after
/ H1, is found by the matching template with the right time difference and SNRs near the injected 12 and 9
/ (noise moves a measured SNR by about 1)
.t.add[`bulk_follow_up;{
    rate:1024; span:16; n:rate*span; st:1e9; gps:st+9;
    t:.qs.gw.template[n;rate;30f;30f;0 0f;20];
    / in unit-variance white noise a signal's optimal SNR is the square root of its sum of squares
    sh:{[t;rate;n;at;rho] h:rotate[neg "j"$rate*at-t`peak] t`h; h*rho%sqrt h$h}[t;rate;n];
    system "S 9";
    / followUp reads span s centred on gps, so the signal sits at span/2 s into the stretch
    sig:(sh[8;12f];sh[8.003;9f]);
    noise:(.qs.data.normal n;.qs.data.normal n);
    data:`H1`L1!((noise 0)+sig 0;(noise 1)+sig 1);
    b:.qs.gw.bulkStretch; .qs.gw.bulkStretch:{[data;rate;run;det;start;stop] `gps`rate`h!(start;rate;data det)}[data;rate];
    r:@[.qs.gw.followUp[`X;gps;];`span`margin`window`bank!(span;2;0.2;(30 30f;10 10f));{(`err;x)}];
    .qs.gw.bulkStretch:b;
    if[`err~first r; e:r 1; 'e];
    best:first r;
    .t.eq[best`m1`m2;30 30f];
    .t.near[best`dt;0.003;0.0015];
    .t.near[best`merger;gps;0.002];
    .t.ok[(best`snrH1) within 10 14;"H1 SNR ",string best`snrH1];
    .t.ok[(best`snrL1) within 7 11;"L1 SNR ",string best`snrL1];
    .t.ok[(best`bgH1)<6;"H1 background ",string best`bgH1];
 }];
