/ GWOSC event catalogue and detector metadata

.t.add[`gw_catalogue_sample;{
    json:raze read0 hsym .t.path "tests/fixtures/gw_events.json";
    events:.qs.gw.parseEvents json;
    .t.eq[string events`name;("GW150914";"GW170817";"GWTEST")];
    .t.eq[events`version;3 3 1];
    .t.eq[events`catalogue;.qs.gw.symbols ("GWTC-1-confident";"GWTC-1-confident";"GWTC-1-confident")];
    .t.near[events[0 1;`gps];1126259462.4 1187008882.4;1e-6];
    .t.near[events[0 1;`snr];25.2 32.4;1e-12];
    .t.ok[null events[2;`mass1];"missing mass should remain null"];
    selected:.qs.gw.confidentBBH[events;10;80;0.9];
    .t.eq[count selected;1];
    .t.eq[first selected`name;`GW150914];
    details:`GW150914`GW170817`GWTEST!(
        "{\"events\":{\"GW150914-v3\":{\"strain\":[{\"detector\":\"H1\"},{\"detector\":\"L1\"}]}}}";
        "{\"events\":{\"GW170817-v3\":{\"strain\":[{\"detector\":\"H1\"}]}}}";
        "{\"events\":{\"GWTEST-v1\":{\"strain\":[]}}}");
    selected:.qs.gw.confidentWithDetectors[events;details;10;80;0.9;`H1`L1];
    .t.eq[count selected;1];
    .t.eq[first selected`name;`GW150914];
 }];

.t.add[`gw_detector_metadata;{
    json:"{\"events\":{\"GW150914-v3\":{\"strain\":[{\"detector\":\"H1\"},{\"detector\":\"L1\"}]}}}";
    .t.ok[.qs.gw.hasDetectors[json;`H1`L1];"H1 and L1 should be available"];
    .t.ok[not .qs.gw.hasDetectors[json;`H1`V1];"missing detector should not be available"];
 }];

.t.add[`gw_taylorf2_chirp_track;{
     frequencies:10+til 8;
     waveform:.qs.gw.taylorF2[frequencies;35.6;30.6;0 0f;440;0f;(0f;0f)];
     phase:waveform[`phase];
     observed:neg (phase[1]-phase[0])%(2*.qs.sig.pi*(frequencies[1]-frequencies[0]));
     total:35.6+30.6;
     eta:(35.6*30.6)%(total*total);
     chirp:total*(eta xexp (3%5));
     expected:.qs.gw.newtonianTime[avg frequencies[0 1];chirp];
     .t.ok[abs observed-expected<0.1*expected;"TaylorF2 time ",string observed," vs Newtonian ",string expected];
     .t.ok[0.5<observed and observed<10;"unexpected low-frequency chirp time ",string observed];
 }];

.t.add[`gw_taylorf2_amplitude_cutoff;{
     frequencies:20+til 100;
    nearer:.qs.gw.taylorF2[frequencies;35.6;30.6;-0.2 0.3f;400;0.7f;(0f;0f)];
     twiceDistance:.qs.gw.taylorF2[frequencies;35.6;30.6;-0.2 0.3f;800;0.7f;(0f;0f)];
    .t.near[nearer[`amp];2*twiceDistance[`amp];1e-12];
    .t.ok[last nearer[`freq]<=nearer[`fisco];"frequencies exceeded ISCO cutoff"];
    .t.ok[nearer[`fisco] within 60 70;"unexpected GW150914-like ISCO ",string nearer[`fisco]];
    .t.ok[all abs nearer[`phase]<1e8;"non-finite inspiral phase"];
 }];

/ every TaylorF2 coefficient against values from a direct Python transcription of LALSuite's
/ XLALSimInspiralPNPhasing_F2 (aligned spins, black holes) for 36 + 29 Msun, chi 0.3 and -0.2
.t.add[`gw_taylorf2_coefficients;{
    c:.qs.gw.f2coeffs[36;29;0.3;-0.2];
    lal:`c2`c3`c4`c5`c6`c6log`c7!(6.424080335618798;-47.681040642841026;45.69134958776763;133.83635571232392;
        -932.4991716846503;-326.0952380952381;694.1722626044312);
    ours:@[c;`c6;:;(c`c6)+(c`c6log)*log 4];
    .t.near[(value ours)%value lal;7#1f;1e-12];
    / no spin, no spin terms
    n:.qs.gw.f2coeffs[36;29;0f;0f];
    .t.near[n`c3;-16*.qs.sig.pi;1e-12];
 }];

/ the leading-order amplitude against LAL's form: 4 m1 m2 / r MRSUN MTSUN sqrt(pi/12) sqrt(5/(32 eta)) v^(-7/2)
.t.add[`gw_taylorf2_amplitude_lal;{
    f:30 50 70f;
    w:.qs.gw.taylorF2[f;30;25;0 0f;500;0f;(0f;0f)];
    mrsun:1476.6250614046494; mtsun:4.9254909476412675e-6;
    total:55f; eta:(30*25)%total*total;
    r:500*3.085677581491367e22;
    v:(.qs.sig.pi*total*mtsun*f) xexp 1%3;
    lal:(4*30*25%r)*mrsun*mtsun*sqrt[.qs.sig.pi%12]*sqrt[5%32*eta]*v xexp -3.5;
    / LAL's MRSUN is a slightly different rounding of MTSUN times c, a 1.6e-8 relative difference
    .t.near[w[`amp]%lal;3#1f;1e-7];
    / face-on: plus has that amplitude, cross is -i times it
    .t.near[(sqrt (w[`plus;0] xexp 2)+w[`plus;1] xexp 2)%lal;3#1f;1e-7];
    .t.near[w[`cross;0];w[`plus;1];1e-35];
    .t.near[w[`cross;1];neg w[`plus;0];1e-35];
 }];

.t.add[`gw_taylorf2_symmetric;{
    f:20+til 100;
    a:.qs.gw.taylorF2[f;36;29;0.4 -0.3;400;0.5;(0.01;0.2)];
    b:.qs.gw.taylorF2[f;29;36;-0.3 0.4;400;0.5;(0.01;0.2)];
    .t.near[a`phase;b`phase;1e-9];
    .t.near[a`amp;b`amp;1e-30];
 }];

/ in the time domain the chirp rises in frequency and ends at the coalescence time: the check that would
/ catch the conjugate (time-reversed) convention, since our FFT uses exp(-2 pi i f t) as LAL does
.t.add[`gw_taylorf2_time_domain;{
    rate:4096; seconds:4; tc:3f;
    n:rate*seconds; df:1%seconds; f:df*til 1+n div 2;
    w:.qs.gw.taylorF2[f where f>=20;30;25;0.2 0.1;400;0f;(tc;0f)];
    bins:"j"$w[`freq]%df;
    re:@[(count f)#0f;bins;:;w[`plus;0]]; im:@[(count f)#0f;bins;:;w[`plus;1]];
    h:rate*.qs.sig.irfft (re;im);
    t:(til n)%rate;
    peak:t first idesc abs h;
    .t.ok[peak within (tc-0.05;tc+0.002);"chirp peaks at ",(string peak)," s, not just before ",string tc];
    crossings:{[h;t;a;b] i:where (t>=a)&t<b; sum 0>(1_h i)*-1_h i}[h;t];
    .t.ok[(crossings[tc-0.8;tc-0.4])<crossings[tc-0.4;tc];"frequency should rise towards coalescence"];
    / almost nothing well before the chirp starts (about 1 s before tc from 20 Hz)
    early:{avg x*x} h where t<1.5;
    late:{avg x*x} h where (t>=tc-0.5)&t<tc;
    .t.ok[early<0.01*late;"energy before the chirp: ",string early%late];
 }];

/ PhenomB against a Python transcription of LALSuite's ComputeIMRPhenomBParams and IMRPhenomBGenerateFD
/ (36 + 29 Msun, chi1 0.3, chi2 -0.2, 400 Mpc), across the inspiral, merger and ringdown pieces
.t.add[`gw_phenomb_lal;{
    f:20 50 100 150 200 300f;
    w:.qs.gw.phenomB[f;36;29;0.3 -0.2;400;0f;(0f;0f)];
    .t.near[(w`fmerger`fring`sigma`fcut)%115.15875686756915 259.25665947125896 58.9673055776804 349.4790648716245;4#1f;1e-12];
    lalAmp:9.037441195641006e-23 2.966322528674908e-23 1.2468879296632372e-23 8.546289209772345e-24 6.99034843116213e-24 2.0384849411233392e-24;
    lalPhase:-917.5590141532003 -968.045747118472 -991.4395602398799 -1009.3234021210112 -1025.6277805621257 -1055.8785698716417;
    .t.near[w[`amp]%lalAmp;6#1f;1e-12];
    .t.near[w`phase;lalPhase;1e-9];
 }];

.t.add[`gw_phenomb_continuous_scaling;{
    w:.qs.gw.phenomB[enlist 50f;36;29;0.2 0.2;400;0f;(0f;0f)];
    at:{[f] .qs.gw.phenomB[f;36;29;0.2 0.2;400;0f;(0f;0f)]`amp};
    / no jump at the merger and ringdown frequencies
    {[at;f] a:at f*(1-1e-9;1+1e-9); .t.near[a 1;a 0;1e-6*a 0]}[at] each w`fmerger`fring;
    / at fixed eta and spin, the frequencies scale as 1/total mass
    d:.qs.gw.phenomB[enlist 20f;72;58;0.2 0.2;400;0f;(0f;0f)];
    .t.near[(d`fmerger`fring`fcut)*2;w`fmerger`fring`fcut;1e-9];
    / an aligned spin raises the merger frequency
    .t.ok[(.qs.gw.phenomB[enlist 20f;36;29;0.6 0.6;400;0f;(0f;0f)]`fmerger)>.qs.gw.phenomB[enlist 20f;36;29;-0.6 -0.6;400;0f;(0f;0f)]`fmerger;"spin should raise fmerger"];
 }];

/ in the early inspiral PhenomB follows TaylorF2: after removing a free time and phase offset, phases agree to
/ a few tenths of a radian over 20-40 Hz; amplitudes are a few percent lower (PhenomB's PN amplitude terms)
.t.add[`gw_phenomb_matches_taylorf2;{
    check:{[m1;m2;c]
        f:20+0.25*til 81;
        a:.qs.gw.phenomB[f;m1;m2;c;400;0f;(0f;0f)]; b:.qs.gw.taylorF2[f;m1;m2;c;400;0f;(0f;0f)];
        d:a[`phase]-b`phase; X:flip (count[f]#1f;f);
        r:d-X mmu (inv (flip X) mmu X) mmu (flip X) mmu d;
        .t.ok[0.3>max abs r;"phase residual ",string max abs r];
        .t.ok[all (a[`amp]%b`amp) within 0.85 1;"amplitude ratio ",-3!(a[`amp]%b`amp) 0 80];
    };
    check .' ((36;29;0 0f);(15;10;0.3 0.3);(50;40;0 0f));
 }];

/ in the time domain: rising frequency, peak shortly before tc (tc is a linear phase reference; the peak falls
/ 20-80 ms earlier for these masses), and nothing well before the chirp
.t.add[`gw_phenomb_time_domain;{
    rate:4096; seconds:4; tc:3f;
    n:rate*seconds; df:1%seconds; f:df*til 1+n div 2;
    w:.qs.gw.phenomB[f where f>=20;36;29;0.3 -0.2;400;0f;(tc;0f)];
    bins:"j"$w[`freq]%df;
    h:rate*.qs.sig.irfft (@[(count f)#0f;bins;:;w[`plus;0]];@[(count f)#0f;bins;:;w[`plus;1]]);
    t:(til n)%rate;
    peak:t first idesc abs h;
    .t.ok[peak within (tc-0.1;tc);"peak at ",string peak];
    crossings:{[h;t;a;b] i:where (t>=a)&t<b; sum 0>(1_h i)*-1_h i}[h;t];
    .t.ok[(crossings[peak-0.4;peak-0.2])<crossings[peak-0.2;peak];"frequency should rise towards the merger"];
    .t.ok[({avg x*x} h where t<1.5)<0.01*{avg x*x} h where (t>=peak-0.3)&t<peak;"energy before the chirp"];
 }];

/ the response tensors against the matrices printed in LALSuite's CreateDetector.c documentation
.t.add[`gw_detector_tensors;{
    .t.near[.qs.gw.tensor`H1;(-0.3926141 -0.0776130 -0.2473886;-0.0776130 0.3195244 0.2279981;-0.2473886 0.2279981 0.0730903);1e-6];
    .t.near[.qs.gw.tensor`L1;(0.4112809 0.1402097 0.2472943;0.1402097 -0.1090056 -0.1816157;0.2472943 -0.1816157 -0.3022755);1e-6];
    / arms are orthogonal unit vectors
    / (LAL's arm vectors, printed to 11 digits, are orthogonal unit vectors to about 2e-7)
    {d:.qs.gw.detectors x; .t.near[(d[`x]$d`x;d[`y]$d`y;d[`x]$d`y);1 1 0f;1e-6]} each `H1`L1;
 }];

.t.add[`gw_sidereal_time;{
    / 2000-01-01 12:00 UTC (GPS 630763213, 13 leap seconds then): GMST 18h 41m 50.548s = 280.46061837 degrees
    .t.near[.qs.gw.gmst 630763213;280.46061837*.qs.sig.pi%180;1e-5];
    .t.eq["j"$.qs.gw.leaps each 600000000 1126259462 1300000000;13 17 18];
    / one sidereal day later the angle comes back (to the rounding of the day's length)
    .t.near[.qs.gw.gmst 1126259462+86164.0905;.qs.gw.gmst 1126259462;1e-4];
 }];

.t.add[`gw_antenna_patterns;{
    system "S 4"; n:500;
    ra:n?2*.qs.sig.pi; dec:asin -1+n?2f; psi:n?.qs.sig.pi; gps:1126259462f;
    a:{[d;r;e;p;g] .qs.gw.antenna[d;r;e;p;g]}[`H1;;;;gps]'[ra;dec;psi];
    p2:{(x[`plus] xexp 2)+x[`cross] xexp 2} each a;
    .t.ok[all p2<=1+1e-9;"F+^2 + Fx^2 above 1"];
    / that sum doesn't depend on the polarisation angle, and a quarter turn swaps F+ and Fx (up to sign)
    b:{[r;e;p] .qs.gw.antenna[`H1;r;e;p;1126259462f]}'[ra;dec;psi+0.7];
    .t.near[p2;{(x[`plus] xexp 2)+x[`cross] xexp 2} each b;1e-9];
    c:{[r;e;p] .qs.gw.antenna[`H1;r;e;p;1126259462f]}'[ra;dec;psi+.qs.sig.pi%4];
    .t.near[abs c@\:`plus;abs a@\:`cross;1e-9];
    / a source straight above the detector (along its vertex vector) sees nearly the full response
    v:.qs.gw.detectors[`H1]`vertex; r:sqrt v$v;
    dec0:asin v[2]%r; ra0:.qs.gw.gmst[gps]+.qs.gw.atan2[v 1;v 0];
    top:.qs.gw.antenna[`H1;ra0;dec0;0.3;gps];
    .t.ok[0.99<(top[`plus] xexp 2)+top[`cross] xexp 2;"overhead response ",-3!top];
 }];

.t.add[`gw_arrival_delays;{
    .t.ok[(.qs.gw.lightTime[`H1;`L1]) within 0.01 0.01002;"H1-L1 light travel time"];
    system "S 5"; n:2000; gps:1126259462f;
    ra:n?2*.qs.sig.pi; dec:asin -1+n?2f;
    dt:{[r;e] .qs.gw.delay[`H1;r;e;1126259462f]-.qs.gw.delay[`L1;r;e;1126259462f]}'[ra;dec];
    lt:.qs.gw.lightTime[`H1;`L1];
    .t.ok[all (abs dt)<=lt+1e-12;"delay beyond the light travel time"];
    .t.ok[(max abs dt)>0.98*lt;"delays should reach close to the light travel time"];
    / the projection's phase factor is exp(-2 pi i f dt)
    w0:.qs.gw.phenomB[30 60 90f;36;29;0 0f;400;0.5;(0f;0f)];
    p0:.qs.gw.project[w0;`L1;1.1;0.2;0.4;gps];
    re:(p0[`fplus]*w0[`plus;0])+p0[`fcross]*w0[`cross;0]; im:(p0[`fplus]*w0[`plus;1])+p0[`fcross]*w0[`cross;1];
    ang:neg 2*.qs.sig.pi*30 60 90f*p0`delay;
    .t.near[p0[`strain;0];(re*cos ang)-im*sin ang;1e-30];
    / an edge-on source has no cross part, so both detectors see the same chirp up to scale and its peak moves
    / by exactly the delay difference (with a cross part the loudest cycle can move by part of a period)
    rate:4096; seconds:4; nn:rate*seconds; df:1%seconds; f:df*til 1+nn div 2;
    w:.qs.gw.phenomB[f where f>=20;36;29;0 0f;400;.qs.sig.pi%2;(3f;0f)];
    peak:{[w;f;df;rate;det;r;e]
        p:.qs.gw.project[w;det;r;e;0.4;1126259462f]; bins:"j"$p[`freq]%df;
        h:rate*.qs.sig.irfft (@[(count f)#0f;bins;:;p[`strain;0]];@[(count f)#0f;bins;:;p[`strain;1]]);
        (first idesc abs h)%rate}[w;f;df;rate];
    r0:ra first idesc abs dt; e0:dec first idesc abs dt;
    shift:peak[`H1;r0;e0]-peak[`L1;r0;e0];
    expect:.qs.gw.delay[`H1;r0;e0;gps]-.qs.gw.delay[`L1;r0;e0;gps];
    .t.ok[0.0005>abs shift-expect;"peak shift ",(string shift)," vs delay difference ",string expect];
 }];

.t.add[`gw_catalogue_cached;{
     file:.t.path "data/gw/allevents.json";
     if[()~key file; .t.skip "fetch the GWOSC catalogue with .qs.gw.events[]"];
     events:.qs.gw.parseEvents raze read0 file;
     .t.ok[300<count events;"unexpectedly few de-duplicated GWOSC events"];
     .t.ok[all `name`catalogue`gps`mass1`mass2`snr`pAstro in cols events;"catalogue fields missing"];
 }];

.t.gwTmpFile:{[name] temp:getenv`TMPDIR; base:$[""~temp;"/tmp";temp]; hsym `$raze (base;"/";name)};

.t.add[`gw_strain_parse_cache;{
     compressed:.t.gwTmpFile "qsolve_gw_strain.txt.gz";
    cachePath:raze (1 _ string compressed;".qcache");
    cache:hsym `$raze (":";cachePath);
     raw:.t.gwTmpFile "qsolve_gw_strain.txt";
     raw 0:("# sample strain";"# This file has 4096 samples per second";"# starting GPS 1234567890 duration 1";"1.25";"nan";"-2.5");
    system "env gzip -f ",.qs.gw.shellQuote[1 _ string raw];
     strain:.qs.gw.strain compressed;
     .t.eq[strain`gps;1234567890];
     .t.eq[strain`rate;4096];
    .t.near[strain[`h][0 2];1.25 -2.5;1e-12];
    .t.ok[null strain[`h] 1;"nan sample was not preserved as null"];
    hdel compressed;
     cached:.qs.gw.strain compressed;
     .t.eq[cached`gps;strain`gps];
    .t.eq[cached`rate;strain`rate];
    .t.eq[cached`h;strain`h];
     hdel cache;
 }];

.t.add[`gw_strain_gw150914_real;{
     file:.t.path "data/gw/strain/GW150914/H1-32-1126259447.txt.gz";
     if[()~key hsym file; .t.skip "run scripts/fetch_gw.sh --duration 32 GW150914"];
     strain:.qs.gw.strain file;
     .t.eq[strain`gps;1126259447];
     .t.eq[strain`rate;4096];
     .t.eq[count strain`h;131072];
     .t.ok[all not null strain`h;"unexpected NaN in GW150914 H1 data"];
 }];