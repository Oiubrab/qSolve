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