/ GWOSC catalogue access and event metadata

.qs.gw.empty:{[v] $[(::)~v;1b;0<=type v;0=count v;null v]}
.qs.gw.field:{[row;fieldName;default] $[not fieldName in key row; default; .qs.gw.empty v:row fieldName; default; v]}
.qs.gw.column:{[rows;fieldName;default] .qs.gw.field[;fieldName;default] each rows}
.qs.gw.symbols:{[items] {$[10h=type x; `$x; `$enlist x]} each items}

/ Parse GWOSC's flat, versioned events map and retain the newest row per common name.
.qs.gw.parseEvents:{[json]
    doc:.j.k json;
    if[not `events in key doc; '"GWOSC JSON has no events object"];
    events:doc`events;
    if[0=count events; :([] id:`symbol$(); name:`symbol$(); catalogue:`symbol$(); version:`int$(); gps:`float$(); mass1:`float$(); mass2:`float$(); distance:`float$(); snr:`float$(); pAstro:`float$(); jsonurl:`symbol$())];
    ids:key events;
    rows:value events;
    t:([]
        id:ids;
        name:.qs.gw.symbols .qs.gw.column[rows;`commonName;""];
        catalogue:.qs.gw.symbols .qs.gw.column[rows;`catalog.shortName;""];
        version:"j"$.qs.gw.column[rows;`version;0N];
        gps:"f"$.qs.gw.column[rows;`GPS;0n];
        mass1:"f"$.qs.gw.column[rows;`mass_1_source;0n];
        mass2:"f"$.qs.gw.column[rows;`mass_2_source;0n];
        distance:"f"$.qs.gw.column[rows;`luminosity_distance;0n];
        snr:"f"$.qs.gw.column[rows;`network_matched_filter_snr;0n];
        pAstro:"f"$.qs.gw.column[rows;`p_astro;0n];
        jsonurl:.qs.gw.symbols .qs.gw.column[rows;`jsonurl;""]);
    t:`name`version xasc t;
    0!select last id,last catalogue,last version,last gps,last mass1,last mass2,last distance,
        last snr,last pAstro,last jsonurl by name from t
 }

/ where gravitational-wave data lives: data/gw under .qs.data.dir (data.q's setting, relative to the working
/ directory unless a script such as dashboard.q sets it)
.qs.gw.dataDir:{[] $[`dir in key `.qs.data; .qs.data.dir; "data"],"/gw"}

/ Load the cached catalogue, downloading it once when absent.
.qs.gw.events:{[]
    file:`:data/gw/allevents.json;
    if[()~key hsym file;
        system "env mkdir -p data/gw";
        system "env curl -fsSL --retry 3 -o data/gw/allevents.json.part https://gwosc.org/eventapi/json/allevents/";
        if[()~key hsym `:data/gw/allevents.json.part; '"GWOSC catalogue download failed"];
        system "env mv data/gw/allevents.json.part data/gw/allevents.json"];
    .qs.gw.parseEvents raze read0 hsym file
 }

/ Select confident binary black holes whose component masses are within the inclusive range.
.qs.gw.confidentBBH:{[events;minMass;maxMass;minPAstro]
    select from events where pAstro>=minPAstro, mass1>=minMass, mass1<=maxMass,
        mass2>=minMass, mass2<=maxMass
 }

/ Check detector availability in a per-event GWOSC JSON response.
.qs.gw.hasDetectors:{[json;detectors]
    doc:.j.k json;
    if[not `events in key doc; :0b];
    events:doc`events;
    if[0=count events; :0b];
    event:first value events;
    if[not `strain in key event; :0b];
    strain:event`strain;
    available:.qs.gw.symbols .qs.gw.column[strain;`detector;""];
    all detectors in available
 }

/ Filter catalogue events by confidence/mass, then by per-event JSON keyed on common name.
.qs.gw.confidentWithDetectors:{[events;details;minMass;maxMass;minPAstro;detectors]
     selected:.qs.gw.confidentBBH[events;minMass;maxMass;minPAstro];
    selected where {[details;detectors;name] .qs.gw.hasDetectors[details name;detectors]}[details;detectors] each selected`name
 }

/ TaylorF2 aligned-spin phasing coefficients, from LALSuite's LALSimInspiralPNCoefficients.c
/ (XLALSimInspiralTaylorF2Phasing_*Coeff, which cites arXiv:1303.7412 for the 3 and 3.5PN spin-orbit terms),
/ with its decimal literals written as exact fractions. m is one body's mass over the total; for black holes
/ the quadrupole-monopole parameter is 1, as LAL's default.
.qs.gw.so3:{[m] m*(25+(38%3)*m)}
/ (q evaluates right to left, so terms with signs are summed as lists to keep each sign where it is written)
.qs.gw.so5:{[m] e:m*1-m; neg m*sum ((13915%84);neg e*10%3;m*sum ((12760%81);e*170%9))}
.qs.gw.so6:{[m] .qs.sig.pi*m*((1490%3)+260*m)}
.qs.gw.so7:{[m] e:m*1-m; m*((-170978035%48384)+(e*2876425%672)+(e*e*4735%144)+m*((-7189233785%1524096)+(e*458555%3024)-(e*e*5345%72)))}
/ 2PN spin-spin: S1S2 (247/4.8) plus S1S2-orbit (-721/4.8) per eta chi1 chi2, and per body
/ quadrupole-monopole plus self-spin, (-720 + 1 + 240 - 7)/9.6 m^2 chi^2
.qs.gw.ss4:{[eta;m1;m2;c1;c2] ((-474%4.8)*eta*c1*c2)+(-486%9.6)*((m1*m1*c1*c1)+m2*m2*c2*c2)}
/ 3PN spin-spin: S1S2-orbit per eta, and per body quadrupole-monopole plus self-spin
.qs.gw.ss6:{[eta;m1;m2;c1;c2]
    self:{[m] m*m*sum ((47035%84);(2935%6)*m;-120*m*m;(-410825%672);neg (1085%12)*m;(1255%36)*m*m)};
    (((32675%112)+(5575%18)*eta)*eta*c1*c2)+(self[m1]*c1*c1)+self[m2]*c2*c2
 }

/ TaylorF2 phase coefficients of v^2 .. v^7 for masses m1, m2 and aligned spins chi1, chi2 (before the
/ overall 3/(128 eta) factor); c5 multiplies (1 + 3 log v) and the v^6 term is c6 + c6log log(4v)
.qs.gw.f2coeffs:{[m1;m2;chi1;chi2]
    total:m1+m2; mu1:m1%total; mu2:m2%total; eta:mu1*mu2;
    c2:(3715%756)+(55%9)*eta;
    c3:(-16*.qs.sig.pi)+(.qs.gw.so3[mu1]*chi1)+.qs.gw.so3[mu2]*chi2;
    c4:(15293365%508032)+((27145%504)*eta)+((3085%72)*eta*eta)+.qs.gw.ss4[eta;mu1;mu2;chi1;chi2];
    c5:sum ((.qs.sig.pi*(38645%756)-(65%9)*eta);.qs.gw.so5[mu1]*chi1;.qs.gw.so5[mu2]*chi2);
    pi2:.qs.sig.pi*.qs.sig.pi;
    c6:sum ((11583231236531%4694215680);neg (640%3)*pi2;neg (6848%21)*0.5772156649015329;
        eta*(-15737765635%3048192)+(2255%12)*pi2;(76055%1728)*eta*eta;neg (127825%1296)*eta*eta*eta;
        .qs.gw.so6[mu1]*chi1;.qs.gw.so6[mu2]*chi2;.qs.gw.ss6[eta;mu1;mu2;chi1;chi2]);
    c6log:neg 6848%21;
    c7:sum ((.qs.sig.pi*sum ((77096675%254016);(378515%1512)*eta;neg (74045%756)*eta*eta));.qs.gw.so7[mu1]*chi1;.qs.gw.so7[mu2]*chi2);
    `c2`c3`c4`c5`c6`c6log`c7!(c2;c3;c4;c5;c6;c6log;c7)
 }

/ TaylorF2 stationary-phase inspiral of the dominant mode, aligned spins, phase to 3.5PN (spin terms
/ included), leading-order amplitude, cut at the Schwarzschild ISCO frequency. Masses in solar masses,
/ distance in Mpc, frequencies in Hz; spins (chi1;chi2) along the orbital angular momentum; coalescence is
/ (time in s;phase). The convention follows LALSuite's TaylorF2 so it suits an FFT with exp(-2 pi i f t):
/   psi(f) = 2 pi f tc - phic + 3/(128 eta v^5) (sum of PN terms), v = (pi M f)^(1/3)
/   h+(f) = A (1+cos^2 i)/2 exp(-i (psi - pi/4)),  hx(f) = -i A cos i exp(-i (psi - pi/4))
/ so the chirp's frequency rises towards tc. phase is psi - pi/4; plus and cross are (real;imaginary)
.qs.gw.taylorF2:{[frequencies;m1;m2;spins;distance;inclination;coalescence]
    chi1:spins 0; chi2:spins 1;
    coalescenceTime:coalescence 0; coalescencePhase:coalescence 1;
    if[(m1<=0) or (m2<=0) or distance<=0; '"masses and distance must be positive"];
    if[not (chi1 within -1 1) and chi2 within -1 1; '"aligned spins must be between -1 and 1"];
    total:m1+m2; mu1:m1%total; mu2:m2%total;
    eta:mu1*mu2;
    chirp:total*eta xexp 3%5;
    solarTime:4.9254909476412675e-6;
    megaParsec:3.085677581491367e22;
    lightSpeed:299792458f;
    massSeconds:solarTime*total;
    fisco:1%((6 xexp 1.5)*.qs.sig.pi*massSeconds);
    frequencies:frequencies where (frequencies>0)&frequencies<=fisco;
    v:(massSeconds*.qs.sig.pi*frequencies) xexp 1%3;
    v2:v*v; v3:v2*v; v4:v2*v2; v5:v4*v; v6:v3*v3; v7:v6*v;
    c:.qs.gw.f2coeffs[m1;m2;chi1;chi2];
    c2:c`c2; c3:c`c3; c4:c`c4; c5:c`c5; c6:c`c6; c6log:c`c6log; c7:c`c7;
    / the 2.5PN term's log(v) part has three times its coefficient, and 3PN carries -6848/21 log(4v)
    series:1+(c2*v2)+(c3*v3)+(c4*v4)+(c5*v5*1+3*log v)+(v6*c6+c6log*log 4*v)+c7*v7;
    psi:(2*.qs.sig.pi*frequencies*coalescenceTime)+(neg coalescencePhase)+(3%128)*series%eta*v5;
    phase:psi-.qs.sig.pi%4;
    chirpSeconds:solarTime*chirp;
    amplitude:sqrt[5%24]*(.qs.sig.pi xexp -2%3)*(chirpSeconds xexp 5%6)*
        (lightSpeed%distance*megaParsec)*frequencies xexp -7%6;
    cosine:cos inclination;
    plusScale:0.5*1+cosine*cosine;
    / plus = A p exp(-i phase); cross = -i A cos(i) exp(-i phase)
    `freq`amp`phase`plus`cross`fisco!(frequencies;amplitude;phase;
        (amplitude*plusScale*cos phase;neg amplitude*plusScale*sin phase);
        (neg amplitude*cosine*sin phase;neg amplitude*cosine*cos phase);fisco)
 }

/ IMRPhenomB (Ajith et al. 2011, PRL 106, 241101): inspiral, merger and ringdown in closed form for aligned
/ spins, transcribed from LALSuite's LALSimIMRPhenom.c (ComputeIMRPhenomBParams, IMRPhenomBGenerateFD).
/ Each phenomenological parameter is a base term plus a fitted polynomial in eta and the mass-weighted spin chi,
/ with coefficients of (eta; eta chi; eta chi^2; eta^2; eta^2 chi; eta^3) in this table
.qs.gw.pbTable:`psi2`psi3`psi4`psi6`psi7`psi8`fMerger`fRing`sigma`fCut!(
    -920.91 492.13 135.03 6741.9 -1053.4 -13397;
    17022 -9565.9 -2182.1 -121370 20752 238590;
    -125440 75066 13382 873540 -165730 -1693600;
    -889770 631020 50676 5980800 -1414800 -11280000;
    869600 -670980 -30082 -5837900 1514500 10891000;
    -366000 306700 631.76 2426500 -721800 -4552400;
    0.64365 0.82696 -0.27063 -0.058218 -3.9346 -7.0916;
    0.1469 -0.12281 -0.026091 -0.0249 0.17013 2.3252;
    -0.40979 -0.035226 0.10082 1.8286 -0.020169 -2.8698;
    -0.13313 -0.081719 0.14512 -0.2714 0.12788 4.922)

/ PhenomB parameters for total mass in solar masses, eta and chi: phase coefficients and the merger, ringdown
/ (with width sigma) and cut-off frequencies in Hz
.qs.gw.pbParams:{[total;eta;chi]
    terms:(eta;eta*chi;eta*chi*chi;eta*eta;eta*eta*chi;eta*eta*eta);
    fit:key[.qs.gw.pbTable]!("f"$value .qs.gw.pbTable) mmu terms;
    s:1-chi; ring:1-0.63*s xexp 0.3;
    base:`psi2`psi3`psi4`psi6`psi7`psi8`fMerger`fRing`sigma`fCut!(
        3715%756;(-16*.qs.sig.pi)+113*chi%3;(15293365%508032)-405*chi*chi%8;0f;0f;0f;
        sum (1f;-4.4547*s xexp 0.217;3.521*s xexp 0.26);ring%2;ring*(s xexp 0.45)%4;
        sum (0.32361;0.048935*chi;0.013463*chi*chi));
    p:base+fit;
    piM:.qs.sig.pi*total*4.9254909476412675e-6;
    @[p;`fMerger`fRing`sigma`fCut;%;piM]
 }

/ IMRPhenomB waveform with the same arguments and result as .qs.gw.taylorF2 (spins are the two aligned spins,
/ combined into chi = (m1 chi1 + m2 chi2)/M), from the lowest frequency given up to PhenomB's cut-off.
/ LAL's form plus the 2 pi f tc term that places the merger: h(f) = A(f) exp(-i psi), with
/   psi = 2 pi f tc - phic + 3/(128 eta v^5)(1 + psi2 v^2 + psi3 v^3 + psi4 v^4 + psi6 v^6 + psi7 v^7 + psi8 v^8)
.qs.gw.phenomB:{[frequencies;m1;m2;spins;distance;inclination;coalescence]
    if[(m1<=0) or (m2<=0) or distance<=0; '"masses and distance must be positive"];
    if[not all (spins 0 1) within -1 1; '"aligned spins must be between -1 and 1"];
    total:m1+m2; eta:(m1*m2)%total*total;
    chi:((m1*spins 0)+m2*spins 1)%total;
    p:.qs.gw.pbParams[total;eta;chi];
    piM:.qs.sig.pi*total*4.9254909476412675e-6;
    frequencies:frequencies where (frequencies>0)&frequencies<p`fCut;
    f:frequencies;
    v:(piM*f) xexp 1%3; v2:v*v; v3:v2*v;
    / amplitude: PN-corrected inspiral, then a power law with spin corrections, then a Lorentzian ringdown,
    / continuous at fMerger and fRing by the choice of w1 and w2
    fm:p`fMerger; fr:p`fRing; sigma:p`sigma;
    alpha2:(-323%224)+451*eta%168; alpha3:chi*(27%8)-11*eta%6;
    eps1:(1.4547*chi)-1.8897; eps2:1.6557-1.8153*chi;
    vm:(piM*fm) xexp 1%3; vr:(piM*fr) xexp 1%3;
    w1:(1+(alpha2*vm*vm)+alpha3*piM*fm)%1+(eps1*vm)+eps2*vm*vm;
    w2:w1*(.qs.sig.pi*sigma%2)*((fr%fm) xexp -2%3)*1+(eps1*vr)+eps2*vr*vr;
    lorentz:sigma%2*.qs.sig.pi*((f-fr)*(f-fr))+sigma*sigma%4;
    shape:?[f<=fm;((f%fm) xexp -7%6)*1+(alpha2*v2)+alpha3*v3;
        ?[f>fr;w2*lorentz;w1*((f%fm) xexp -2%3)*1+(eps1*v)+eps2*v2]];
    lightSpeed:299792458f; megaParsec:3.085677581491367e22;
    amp0:(((4.9254909476412675e-6*total) xexp 5%6)*(fm xexp -7%6)*sqrt[5*eta%24]*lightSpeed%distance*megaParsec)%.qs.sig.pi xexp 2%3;
    amplitude:amp0*shape;
    v4:v2*v2; v5:v4*v; v6:v3*v3; v7:v6*v; v8:v7*v;
    series:1+sum (p[`psi2]*v2;p[`psi3]*v3;p[`psi4]*v4;p[`psi6]*v6;p[`psi7]*v7;p[`psi8]*v8);
    phase:(2*.qs.sig.pi*f*coalescence 0)+(neg coalescence 1)+(3%128)*series%eta*v5;
    cosine:cos inclination; plusScale:0.5*1+cosine*cosine;
    `freq`amp`phase`plus`cross`fcut`fmerger`fring`sigma!(f;amplitude;phase;
        (amplitude*plusScale*cos phase;neg amplitude*plusScale*sin phase);
        (neg amplitude*cosine*sin phase;neg amplitude*cosine*cos phase);p`fCut;fm;fr;sigma)
 }

/ Detectors, from LALSuite's LALDetectors.h: vertex position (Earth-centred, Earth-fixed, metres) and unit
/ vectors along the x and y arms in the same frame
.qs.gw.detectors:`H1`L1!(
    `vertex`x`y!(-2.16141492636e6 -3.83469517889e6 4.60035022664e6;-0.22389266154 0.79983062746 0.55690487831;
        -0.91397818574 0.02609403989 -0.40492342125);
    `vertex`x`y!(-7.42760447238e4 -5.49628371971e6 3.22425701744e6;-0.95457412153 -0.14158077340 -0.26218911324;
        0.29774156894 -0.48791033647 -0.82054461286))

/ the detector response tensor D = (x x' - y y')/2, as LAL's CreateDetector.c builds it
.qs.gw.tensor:{[det] d:.qs.gw.detectors det; 0.5*(d[`x]*/:d`x)-d[`y]*/:d`y}

/ GPS - UTC in seconds at a GPS time: LAL's XLALLeapSeconds.h table (GPS second of each leap; TAI-UTC - 19)
.qs.gw.leapGps:46828800 78364801 109900802 173059203 252028804 315187205 346723206 393984007 425520008 457056009 504489610 551750411 599184012 820108813 914803214 1025136015 1119744016 1167264017
.qs.gw.leaps:{[gps] sum gps>=.qs.gw.leapGps}

/ Greenwich mean sidereal time in radians, as LAL's XLALGreenwichMeanSiderealTime: UTC from GPS, Julian centuries
/ from J2000.0, then the IAU 1982 polynomial in seconds of sidereal time
.qs.gw.gmst:{[gps]
    jd:2444244.5+(gps-.qs.gw.leaps floor gps)%86400;
    t:(jd-2451545.0)%36525;
    seconds:sum ((0.093104-6.2e-6*t)*t*t;67310.54841;(8640184.812866+3155760000.0)*t);
    (seconds*.qs.sig.pi%43200) mod 2*.qs.sig.pi
 }

/ antenna patterns F+ and Fx of a detector for a source at (ra, dec) with polarisation psi at a GPS time,
/ following LAL's XLALComputeDetAMResponse (Anderson et al. 2001, Eqs. B4-B7)
.qs.gw.antenna:{[det;ra;dec;psi;gps]
    gha:.qs.gw.gmst[gps]-ra;
    X:(neg (cos psi)*sin gha)-(sin psi)*(cos gha)*sin dec;
    X:(X;(neg (cos psi)*cos gha)+(sin psi)*(sin gha)*sin dec;(sin psi)*cos dec);
    Y:((sin psi)*(sin gha))-(cos psi)*(cos gha)*sin dec;
    Y:(Y;((sin psi)*cos gha)+(cos psi)*(sin gha)*sin dec;(cos psi)*cos dec);
    D:.qs.gw.tensor det;
    `plus`cross!((X$D$X)-Y$D$Y;(X$D$Y)+Y$D$X)
 }

/ seconds by which a signal from (ra, dec) reaches the detector after the Earth's centre, as LAL's
/ XLALTimeDelayFromEarthCenter: minus the vertex's projection on the direction to the source, over c
.qs.gw.delay:{[det;ra;dec;gps]
    gha:.qs.gw.gmst[gps]-ra;
    toSource:((cos dec)*cos gha;neg (cos dec)*sin gha;sin dec);
    neg (toSource$.qs.gw.detectors[det]`vertex)%299792458f
 }

/ (q's atan takes one argument; .qs.sig.atan2 is the two-argument form)
.qs.gw.atan2:{[y;x] .qs.sig.atan2[y;x]}

/ the light travel time between two detectors, seconds
.qs.gw.lightTime:{[a;b] d:.qs.gw.detectors[a][`vertex]-.qs.gw.detectors[b]`vertex; (sqrt d$d)%299792458f}

/ a waveform (from taylorF2 or phenomB) as one detector sees it: F+ h+ + Fx hx, delayed by the arrival time at
/ the detector relative to the Earth's centre (a factor exp(-2 pi i f dt)); returns `freq`strain`fplus`fcross`delay
.qs.gw.project:{[w;det;ra;dec;psi;gps]
    a:.qs.gw.antenna[det;ra;dec;psi;gps]; dt:.qs.gw.delay[det;ra;dec;gps];
    re:(a[`plus]*w[`plus;0])+a[`cross]*w[`cross;0];
    im:(a[`plus]*w[`plus;1])+a[`cross]*w[`cross;1];
    ang:neg 2*.qs.sig.pi*w[`freq]*dt;
    `freq`strain`fplus`fcross`delay!(w`freq;((re*cos ang)-im*sin ang;(re*sin ang)+im*cos ang);a`plus;a`cross;dt)
 }

/ a face-on PhenomB template as a time series on an n-sample grid at a sample rate, from flow Hz, with its
/ coalescence reference at t=0 (the chirp wraps to the end of the buffer); returns `h`peak, peak being the
/ time of its largest sample relative to that reference (negative: the merger comes just before tc)
.qs.gw.template:{[n;rate;m1;m2;spins;flow]
    df:rate%n; f:df*til 1+n div 2;
    w:.qs.gw.phenomB[f where f>=flow;m1;m2;spins;400;0f;(0f;0f)];
    bins:"j"$w[`freq]%df;
    h:rate*.qs.sig.irfft (@[(count f)#0f;bins;:;w[`plus;0]];@[(count f)#0f;bins;:;w[`plus;1]]);
    p:(first idesc abs h)%rate;
    `h`peak!(h;$[p>n%2*rate;p-n%rate;p])
 }

/ matched-filter search of one detector's strain (from .qs.gw.strain) for a PhenomB template: PSD by Welch
/ (4 s Hann segments, 50% overlap), a 10% Tukey taper, phase-maximised SNR, and only lags at least margin
/ seconds from either end (the whitening filter is about a PSD segment long, so the edges are unreliable).
/ Returns the best SNR, the GPS time of the merger (the template's peak), its phase, and the largest SNR more
/ than half a second from it
.qs.gw.search:{[strain;m1;m2;spins;margin]
    x:strain`h; rate:strain`rate; n:count x;
    psd:.qs.sig.welch[x;rate;4*rate;2*rate];
    t:.qs.gw.template[n;rate;m1;m2;spins;20];
    mf:.qs.sig.matchedFilterMax[x*.qs.sig.tukey[n;0.1];t`h;rate;psd];
    ok:where (mf[`lag]>=margin)&mf[`lag]<(n%rate)-margin;
    best:ok first idesc mf[`snr] ok;
    time:strain[`gps]+mf[`lag][best]+t`peak;
    away:ok where 0.5<abs (strain[`gps]+mf[`lag] ok)+t[`peak]-time;
    `snr`gps`phase`background!(mf[`snr] best;time;mf[`phase] best;max mf[`snr] away)
 }

/ the noise library (G4.1). Each detector's long strain is cut into overlapping chunks; each chunk is
/ resampled, whitened with its own PSD, and only its middle is kept (the whitening corrupts about a PSD
/ segment at each end, see G3.4), so the kept parts tile the stream without gaps. Chunks with missing samples,
/ or kept parts within `guard` seconds of a catalogued event, are dropped.
.qs.gw.noiseDefaults:`rate`chunk`margin`guard!(2048;64;6;8)

/ which events' noise is used for what: no noise from a test event is ever used for training (G-D9), and
/ GW150914, the headline test detection, is a test event
.qs.gw.noiseSplit:`GW151226`GW170104`GW190412`GW200322_091133`GW170814`GW150914!`train`train`train`train`validation`test

/ whitened noise from one detector's strain dictionary (`gps`rate`h) at o`rate Hz, avoiding the GPS times in
/ avoid. Returns a table: gps (start of the kept samples), w (whitened samples as reals) and psd (`freq`psd)
.qs.gw.noiseChunks:{[strain;avoid;o]
    o:.qs.gw.noiseDefaults,o;
    x:strain`h; rate:strain`rate; n:count x;
    len:rate*o`chunk; stride:rate*o[`chunk]-2*o`margin;
    starts:stride*til 1+0|(n-len) div stride;
    starts:starts where (starts+len)<=n;
    one:{[x;rate;o;len;s]
        c:x s+til len;
        if[any null c; :()];
        c:$[rate=o`rate; c; .qs.sig.resample[c;rate;o`rate]];
        r:o`rate; m:r*o`margin;
        psd:.qs.sig.welch[c;r;4*r;2*r];
        w:.qs.sig.whiten[c*.qs.sig.tukey[count c;0.1];r;`freq`asd!(psd`freq;sqrt psd`psd)];
        enlist `start`w`psd!(s;"e"$w m+til (count w)-2*m;psd)
    }[x;rate;o;len] peach starts;
    t:raze one;
    if[not count t; :([] gps:`float$(); w:(); psd:())];
    gps:strain[`gps]+(t[`start]%rate)+o`margin;
    span:o[`chunk]-2*o`margin;
    far:{[avoid;g;guard;span] not any (avoid>=g-guard)&avoid<=g+span+guard}[avoid;;o`guard;span] each gps;
    select from ([] gps; w:t`w; psd:t`psd) where far
 }

/ an event's noise for both detectors (its 4096 s files under data/gw/strain/<event>/), avoiding every catalogued
/ event in the stretch, keeping only times both detectors have; returns a table event, det, gps, w, psd
.qs.gw.noiseEvent:{[event;o]
    dir:.qs.gw.dataDir[],"/strain/",string event;
    / (like rejects "*-4096-*.txt.gz" with nyi: two wildcards and more text after them, so match the parts)
    f:key hsym `$dir; files:f where (f like "*-4096-*") and f like "*.txt.gz";
    if[not 2=count files; '"need the H1 and L1 4096 s files for ",string event];
    ev:.qs.gw.events[];
    one:{[dir;ev;o;file]
        s:.qs.gw.strain hsym `$dir,"/",string file;
        span:"f"$s[`gps]+0,(count s`h)%s`rate;
        avoid:exec gps from ev where gps within span;
        t:.qs.gw.noiseChunks[s;avoid;o];
        update det:`$2#string file from t}[dir;ev;o];
    t:raze one each files;
    both:exec gps from (select n:count i by gps from t) where n=2;
    `event`det`gps xcols update event:event from select from t where gps in both
 }

/ build and save the noise library: one file per event in data/gw/noise/, returning a summary per event
.qs.gw.buildNoise:{[events;o]
    {[o;e]
        t0:.z.p; t:.qs.gw.noiseEvent[e;o];
        (hsym `$.qs.gw.dataDir[],"/noise/",string e) set t;
        w:"f"$raze t`w;
        / seconds of coincident noise (each chunk is one detector, so half the rows per detector)
        `event`split`chunks`seconds`variance`kurtosis`maxAbs`buildSeconds!(e;.qs.gw.noiseSplit e;count t;
            ((count t) div 2)*(.qs.gw.noiseDefaults,o)[`chunk]-2*(.qs.gw.noiseDefaults,o)`margin;avg w*w;
            (avg w xexp 4)%(avg w*w) xexp 2;max abs w;1e-9*.z.p-t0)}[o] each events
 }

.qs.gw.loadNoise:{[event] get hsym `$.qs.gw.dataDir[],"/noise/",string event}

/ injections (G4.2). Parameters are drawn per injection; the waveform is made in the frequency domain on a
/ buffer long enough for the lightest systems, projected onto H1 and L1 (antenna patterns and delays), divided
/ by each detector's ASD (that of the noise chunk it goes into) and inverse-FFT'd once: whitening is linear, so
/ this equals whitening the strain, and the whitened signal's sum of squares is its optimal SNR squared.
.qs.gw.injectDefaults:`window`buffer`flow`mass`spin`snr`merger!(1;8;20;10 80f;-0.8 0.8;5 30f;0.6 0.9)

/ n random injection parameter sets: component masses uniform in mass (m1 >= m2), aligned spins uniform in spin,
/ isotropic sky and orientation, uniform polarisation and phase, network SNR uniform in snr, and the merger's
/ position in the window as a fraction, uniform in merger
.qs.gw.drawParams:{[n;o]
    o:.qs.gw.injectDefaults,o;
    a:o[`mass;0]+n?o[`mass;1]-o[`mass;0]; b:o[`mass;0]+n?o[`mass;1]-o[`mass;0];
    u:{[o;n;r] r[0]+n?r[1]-r[0]}[o;n];
    ([] m1:a|b; m2:a&b; chi1:u o`spin; chi2:u o`spin; ra:n?2*.qs.sig.pi; dec:asin -1+n?2f; psi:n?.qs.sig.pi;
        inclination:acos -1+n?2f; phase:n?2*.qs.sig.pi; snr:u o`snr; merger:u o`merger)
 }

/ the whitened signal of one parameter set in both detectors over the buffer, for a pair of noise PSDs at a GPS
/ time: `H1`L1 time series (unit-variance-noise units), each detector's optimal SNR and the merger's sample
.qs.gw.whitenedSignal:{[p;psds;gps;o]
    o:.qs.gw.injectDefaults,o;
    rate:.qs.gw.noiseDefaults`rate; n:rate*o`buffer; df:rate%n; f:df*til 1+n div 2;
    w:.qs.gw.phenomB[f where f>=o`flow;p`m1;p`m2;p`chi1`chi2;400;p`inclination;(0f;p`phase)];
    bins:"j"$w[`freq]%df;
    one:{[w;f;bins;rate;p;gps;det;psd]
        pr:.qs.gw.project[w;det;p`ra;p`dec;p`psi;gps];
        asd:sqrt .qs.sig.interpolate[psd`freq;psd`psd;f bins];
        / the DFT of the strain is rate times its continuous transform; whiten as .qs.sig.whiten does
        scale:rate*sqrt[2%rate]%asd;
        .qs.sig.irfft (@[(count f)#0f;bins;:;scale*pr[`strain;0]];@[(count f)#0f;bins;:;scale*pr[`strain;1]])
    }[w;f;bins;rate;p;gps];
    h:`H1`L1!one'[`H1`L1;psds];
    snrs:{sqrt sum x*x} each h;
    `h`snr`peak!(h;snrs;first idesc (h[`H1]*h`H1)+h[`L1]*h`L1)
 }

/ one training window from a noise library (rows for both detectors at the same times, as .qs.gw.loadNoise) and
/ its pairs (.qs.gw.pairs), with a signal injected at parameters p, or none when p is empty. Returns `x (H1 then
/ L1, as one vector) and, for signals, the network SNR over the whole signal and inside the window
.qs.gw.sample:{[lib;pairs;p;o]
    o:.qs.gw.injectDefaults,o;
    rate:.qs.gw.noiseDefaults`rate; len:rate*o`window;
    rows:pairs first 1?count pairs;
    span:count lib[rows 0;`w];
    start:first 1?1+span-len;
    noise:{[lib;start;len;r] "f"$lib[r;`w] start+til len}[lib;start;len] each rows;
    if[not count p; :`x`snr`snrWindow!(raze noise;0n;0n)];
    s:.qs.gw.whitenedSignal[p;lib[rows;`psd];lib[rows 0;`gps];o];
    scale:p[`snr]%sqrt sum s[`snr] xexp 2;
    / start the window so the merger falls at the requested fraction of it (the buffer is circular)
    n:count s[`h;`H1];
    idx:(s[`peak]+(til len)-"j"$p[`merger]*len) mod n;
    sig:{[s;idx;scale;d] scale*s[`h;d] idx}[s;idx;scale] each `H1`L1;
    `x`snr`snrWindow!(raze noise+sig;p`snr;sqrt sum {x$x} each sig)
 }

/ the coincident chunk pairs of a library: for each GPS time present in both detectors, its (H1 row; L1 row)
.qs.gw.pairs:{[lib]
    g:exec i by gps from select gps,det from lib;
    {[lib;r] r iasc lib[r;`det]}[lib] each g where 2=count each g
 }

/ glitches (G4.3): a sine-Gaussian exp(-(t-t0)^2/tau^2) sin(2 pi f0 (t-t0) + phase), tau = Q/(2 pi f0), in one
/ detector only, scaled to an SNR in whitened units (its sum of squares). "blip" glitches are short, broadband
/ (Q 1-3); "sineGaussian" ones ring longer (Q 4-20). Frequencies log-uniform in freq, SNR uniform in snr
.qs.gw.glitchDefaults:`freq`snr`blipQ`sgQ!(30 500f;5 40f;1 3f;4 20f)
.qs.gw.glitch:{[len;rate;o]
    o:.qs.gw.glitchDefaults,o;
    blip:first 1?01b;
    q:{[r] r[0]+first 1?r[1]-r 0} o[$[blip;`blipQ;`sgQ]];
    f0:exp {[r] r[0]+first 1?r[1]-r 0} log o`freq;
    tau:q%2*.qs.sig.pi*f0;
    t:(til len)%rate; t0:first 0.1+1?0.8*len%rate;
    g:(exp neg ((t-t0)*(t-t0))%tau*tau)*sin (first 1?2*.qs.sig.pi)+2*.qs.sig.pi*f0*t-t0;
    snr:{[r] r[0]+first 1?r[1]-r 0} o`snr;
    `g`det`snr`f0`q`t0`kind!(g*snr%sqrt g$g;first 1?`H1`L1;snr;f0;q;t0;$[blip;`blip;`sineGaussian])
 }

/ real glitches in a noise library (hard negatives): in each chunk, samples whose whitened amplitude exceeds thr
/ (Gaussian noise reaches about 4.5 in a 52 s chunk), merged into one glitch per `sep` seconds. Returns chunk row,
/ detector, sample index of the loudest sample and its amplitude
.qs.gw.mine:{[lib;thr;sep]
    gap:"j"$sep*.qs.gw.noiseDefaults`rate;
    one:{[thr;gap;w] a:abs "f"$w; i:where a>thr;
        if[not count i; :([] at:`long$(); peak:`float$())];
        grp:value group sums 1b,gap<1_deltas i;
        top:{[a;i;k] i k first idesc a i k}[a;i] each grp;
        ([] at:top; peak:a top)}[thr;gap];
    t:raze {[lib;one;r] t:one lib[r;`w]; ([] row:(count t)#r; det:(count t)#lib[r;`det]),'t}[lib;one] each til count lib;
    $[count t; t; ([] row:`long$(); det:`symbol$(); at:`long$(); peak:`float$())]
 }

/ the noise a model mistakes for signals (hard negatives by score): every pair of a library scanned in 1 s windows
/ every `step` seconds, keeping local maxima of the logit (at least `sep` s apart) above `logit`. Returns rows as
/ .qs.gw.mine's (row is the pair's H1 row, at the window's middle sample, peak the logit)
.qs.gw.mineScores:{[m;lib;pairs;o]
    o:(`step`sep`logit!(0.25;1;0f)),o;
    r:.qs.gw.noiseDefaults`rate; len:r; st:"j"$o[`step]*r; w:"j"$o[`sep]%o`step;
    t:raze {[m;lib;len;st;w;o;p]
        span:count lib[p 0;`w]; s:st*til 1+(span-len) div st;
        X:{[lib;p;len;s] "f"$raze lib[p;`w]@\:s+til len}[lib;p;len] each s;
        z:first each .qs.predictRaw[m;X];
        keep:where (z>o`logit)&{[z;w;i] z[i]=max z (0|i-w)+til 1+(2*w)&(count z)-0|i-w}[z;w] each til count z;
        ([] row:(count keep)#p 0; det:(count keep)#`both; at:(s keep)+len div 2; peak:z keep)}[m;lib;len;st;w;o] each pairs;
    $[count t; t; ([] row:`long$(); det:`symbol$(); at:`long$(); peak:`float$())]
 }

/ a window of len samples from both detectors containing a mined glitch g (a row of .qs.gw.mine's table) at a
/ random place in its middle 80%; owner maps a library row to its pair
.qs.gw.glitchWindow:{[lib;pairs;owner;len;g]
    rows:pairs owner g`row; span:count lib[rows 0;`w];
    start:0|(span-len)&g[`at]-"j"$len*0.1+first 1?0.8;
    `x`snr`kind!(raze {[lib;start;len;r] "f"$lib[r;`w] start+til len}[lib;start;len] each rows;g`peak;`real)
 }

/ a batch of n windows from a library for training: a share sig of them with an injected chirp (label 1), and of
/ the rest a share glitchy with a glitch in one detector (label 0), the remainder clean noise (label 0).
/ With o`mined (a .qs.gw.mine table of the same library) a share `real` of the glitchy windows hold a real glitch
/ instead of a simulated one (kind `real, snr its peak amplitude).
/ Returns `X`Y`meta, meta holding each row's kind and SNRs. Signals are made in parallel (peach)
.qs.gw.batch:{[lib;pairs;n;o]
    o:(`signal`glitchy`real!(0.5;0.25;0.5)),o;
    ns:"j"$n*o`signal; ng:"j"$(n-ns)*o`glitchy; nn:n-ns+ng;
    owner:(raze pairs)!raze {2#x} each til count pairs;
    mined:$[`mined in key o; select from o`mined where row in key owner; ()];
    nr:$[count mined; "j"$ng*o`real; 0];
    params:.qs.gw.drawParams[ns;o];
    rate:.qs.gw.noiseDefaults`rate; len:rate*(.qs.gw.injectDefaults,o)`window;
    sig:{[lib;pairs;o;p] .qs.gw.sample[lib;pairs;p;o]}[lib;pairs;o] peach params;
    clean:{[lib;pairs;o;k] .qs.gw.sample[lib;pairs;()!();o]}[lib;pairs;o] each til nn+ng;
    gl:{[len;rate;o;c] g:.qs.gw.glitch[len;rate;o]; x:c`x; off:$[`H1=g`det;0;len]; x[off+til len]+:g`g;
        `x`snr`kind!(x;g`snr;g`kind)}[len;rate;o] each (ng-nr)#clean;
    gl,:.qs.gw.glitchWindow[lib;pairs;owner;len] each mined nr?count mined;
    rows:(sig@\:`x),(nn#clean@\:`x),gl@\:`x;
    / (meta is a keyword, so the row descriptions are called info here)
    info:([] kind:(ns#`signal),(nn#`noise),gl@\:`kind; snr:(sig@\:`snr),(nn#0n),gl@\:`snr;
        snrWindow:(sig@\:`snrWindow),(nn+ng)#0n);
    order:neg[n]?n;
    `X`Y`meta`params!(rows order;enlist each "f"$order<ns;info order;params)
 }

/ scanning real data with a trained detector (G7.1): whiten each detector's strain as the noise library does
/ (resampled to the library rate, its own Welch PSD, a Tukey taper, only the middle kept), then score 1 s windows
/ stepping by `step` seconds. Returns a table of window start time (GPS), the time a merger would have if it sat
/ at the middle of the trained range (`merger` fraction of the window), and the score
.qs.gw.whitenStrain:{[strain;margin]
    r:.qs.gw.noiseDefaults`rate; x:strain`h;
    x:$[strain[`rate]=r; x; .qs.sig.resample[x;strain`rate;r]];
    psd:.qs.sig.welch[x;r;4*r;2*r];
    w:.qs.sig.whiten[x*.qs.sig.tukey[count x;0.1];r;`freq`asd!(psd`freq;sqrt psd`psd)];
    m:r*margin;
    `gps`w!(strain[`gps]+margin;w m+til (count w)-2*m)
 }
.qs.gw.scan:{[m;h1;l1;step;o]
    o:(`margin`merger!(6;0.75)),o;
    r:.qs.gw.noiseDefaults`rate; len:r*(.qs.gw.injectDefaults,o)`window;
    a:.qs.gw.whitenStrain[h1;o`margin]; b:.qs.gw.whitenStrain[l1;o`margin];
    if[not a[`gps]=b`gps; '"H1 and L1 strain must start at the same GPS time"];
    n:min count each (a`w;b`w);
    starts:("j"$step*r)*til 1+(n-len) div "j"$step*r;
    X:{[a;b;len;s] (a[`w] s+til len),b[`w] s+til len}[a;b;len] each starts;
    sc:first each .qs.predict[m;X];
    ([] start:a[`gps]+starts%r; merger:a[`gps]+(starts%r)+o[`merger]*len%r; score:sc)
 }

/ scan one event's 32 s H1/L1 files (data/gw/strain/<event>/) with model m: the best score within `near` seconds
/ of the event's GPS time (on the implied merger time) and the best more than `far` seconds away (the local
/ background in the same data). Events with missing samples in the scanned stretch are reported as gaps
.qs.gw.scanEvent:{[m;event;gps;o]
    o:(`step`near`far!(1%16;0.3;1.5)),o;
    dir:.qs.gw.dataDir[],"/strain/",string event;
    f:key hsym `$dir; f:f where (f like "*-32-*") and f like "*.txt.gz";
    hf:f where f like "H1*"; lf:f where f like "L1*";
    if[not (1=count hf) and 1=count lf; :`event`status!(event;`missing)];
    h:.qs.gw.strain hsym `$dir,"/",string first hf; l:.qs.gw.strain hsym `$dir,"/",string first lf;
    if[any null h[`h],l`h; :`event`status!(event;`gap)];
    t:.qs.gw.scan[m;h;l;o`step;o];
    near:select from t where (o`near)>=abs merger-gps;
    away:select from t where (o`far)<abs merger-gps;
    best:near first idesc near`score;
    `event`status`score`offset`background!(event;`ok;best`score;best[`merger]-gps;max away`score)
 }

/ background in an event's own data by time slides (G6.1 in O3-like conditions): L1 is shifted against H1 by each
/ of `shifts` seconds, circularly within the clean stretch, and every 1 s window is scored; windows touching either
/ detector's event (within `guard` s) are left out, so only noise remains. Triggers are local maxima of the score
/ at least `sep` seconds apart. Returns the trigger scores and the seconds of background searched
.qs.gw.peaks:{[score;w] keep:{[score;w;i] score[i]=max score (0|i-w)+til 1+(2*w)&(count score)-0|i-w}[score;w] each til count score; score where keep}
.qs.gw.slides:{[m;event;gps;shifts;o]
    o:(`step`margin`merger`guard`sep!(1%16;6;0.75;1;1)),o;
    dir:.qs.gw.dataDir[],"/strain/",string event;
    f:key hsym `$dir; f:f where (f like "*-32-*") and f like "*.txt.gz";
    hf:f where f like "H1*"; lf:f where f like "L1*";
    if[not (1=count hf) and 1=count lf; :`triggers`seconds!(`float$();0f)];
    h:.qs.gw.strain hsym `$dir,"/",string first hf; l:.qs.gw.strain hsym `$dir,"/",string first lf;
    if[any null h[`h],l`h; :`triggers`seconds!(`float$();0f)];
    a:.qs.gw.whitenStrain[h;o`margin]; b:.qs.gw.whitenStrain[l;o`margin];
    r:.qs.gw.noiseDefaults`rate; len:r*(.qs.gw.injectDefaults,o)`window; n:min count each (a`w;b`w);
    stepN:"j"$o[`step]*r; starts:stepN*til 1+(n-len) div stepN;
    ev:"j"$r*gps-a`gps;
    / (q functions take at most 8 parameters, so the fixed values travel in one dictionary)
    c:`a`b`r`len`n`starts`ev`o`m!(a;b;r;len;n;starts;ev;o;m);
    one:{[c;k]
        sh:"j"$k*c`r; len:c`len; n:c`n;
        / L1 sample j pairs with H1 sample j+sh (mod n): a window is clean if neither detector's span holds the event
        ok:c[`starts] where {[len;n;ev;g;sh;s] not any (abs ((s,s+sh) mod n)+(len div 2)-ev)<g+len%2}[len;n;c`ev;c[`o;`guard]*c`r;sh] each c`starts;
        X:{[a;b;len;n;sh;s] (a[`w] s+til len),b[`w] (s+sh+til len) mod n}[c`a;c`b;len;n;sh] each ok;
        sc:$[count X; first each .qs.predict[c`m;X]; `float$()];
        `triggers`seconds!(.qs.gw.peaks[sc;"j"$c[`o;`sep]%c[`o;`step]];(count ok)*c[`o;`step])}[c];
    res:one each shifts;
    `triggers`seconds!(raze res@\:`triggers;sum res@\:`seconds)
 }

/ bulk data (open-search plan, S1): files converted from GWOSC HDF5 by scripts/gwosc_bulk.py into
/ data/gw/bulk/<run>/<det>-<gps>-<duration>.{f32,dq,inj,json}: float32 strain, int32 per-second masks, a description

/ one converted file by its base path (no extension): like .qs.gw.strain (`gps`rate`h) plus the masks
.qs.gw.bulk:{[base]
    info:.j.k raze read0 hsym `$base,".json";
    h:"f"$first (enlist "e";enlist 4) 1: hsym `$base,".f32";
    dq:first (enlist "i";enlist 4) 1: hsym `$base,".dq";
    inj:first (enlist "i";enlist 4) 1: hsym `$base,".inj";
    if[not (count h)="j"$info[`duration]*info`rate; '"strain length doesn't match ",base];
    `detector`gps`rate`h`dq`inj`dqBits`injBits!(`$info`detector;"f"$info`gps;"j"$info`rate;h;dq;inj;`$info`dqBits;`$info`injBits)
 }

/ the converted files of a run: detector, start, duration, missing samples, base path
.qs.gw.bulkIndex:{[run]
    dir:.qs.gw.dataDir[],"/bulk/",string run;
    f:key hsym `$dir; f:f where f like "*.json";
    if[not count f; :([] detector:`symbol$(); gps:`float$(); duration:`long$(); nan:`long$(); base:())];
    info:{[dir;f] .j.k raze read0 hsym `$dir,"/",string f}[dir] each f;
    `detector`gps xasc ([] detector:`$info@\:`detector; gps:"f"$info@\:`gps; duration:"j"$info@\:`duration;
        nan:"j"$info@\:`nan; base:{[dir;f] dir,"/",-5_string f}[dir] each f)
 }

/ whether bit `name` is set in each of a mask's values, given the mask's bit names
.qs.gw.bit:{[mask;bits;name] if[not name in bits; '"no mask bit ",string name]; 1=(mask div "j"$2 xexp bits?name) mod 2}

/ per second of a converted file: usable for a CBC search (data present, CAT1 and CAT2 vetoes passed, no CBC
/ hardware injection)
.qs.gw.usable:{[b]
    (.qs.gw.bit[b`dq;b`dqBits;`DATA])&(.qs.gw.bit[b`dq;b`dqBits;`CBC_CAT1])&(.qs.gw.bit[b`dq;b`dqBits;`CBC_CAT2])&
        .qs.gw.bit[b`inj;b`injBits;`NO_CBC_HW_INJ]
 }

/ contiguous runs of true in a boolean vector of seconds starting at gps: start, end, duration (at least minimum)
.qs.gw.segments:{[ok;gps;minimum]
    if[not any ok; :([] start:`float$(); end:`float$(); duration:`long$())];
    / d[k] = ok[k] - ok[k-1] (with ok[-1] and ok[n] false): starts where it rises, ends (exclusive) where it falls
    e:"j"$ok; d:1_deltas 0,e,0; st:where 1=d; en:where -1=d;
    t:([] start:gps+st; end:gps+en; duration:en-st);
    select from t where duration>=minimum
 }

/ the seconds of a run between two GPS times where both H1 and L1 are usable, as segments of at least minimum s
.qs.gw.analysable:{[run;start;stop;minimum]
    ix:select from .qs.gw.bulkIndex[run] where (gps+duration)>start, gps<stop;
    secs:"f"$start+til "j"$stop-start;
    good:{[ix;secs;det]
        r:select from ix where detector=det;
        ok:(count secs)#0b;
        {[secs;ok;row] b:.qs.gw.bulk row`base; u:.qs.gw.usable b; t:row[`gps]+til count u;
            i:secs?t; w:where i<count secs; @[ok;i w;:;u w]}[secs]/[ok;r]
    }[ix;secs] each `H1`L1;
    .qs.gw.segments[(&). good;start;minimum]
 }

/ the strain of one detector over [start, stop) from a run's converted bulk files (NaN where there are none)
.qs.gw.bulkStretch:{[run;det;start;stop]
    ix:select from .qs.gw.bulkIndex[run] where detector=det, (gps+duration)>start, gps<stop;
    r:4096; n:"j"$r*stop-start; h:n#0n;
    h:{[r;start;n;h;row] b:.qs.gw.bulk row`base; off:"j"$r*b[`gps]-start;
        idx:off+til count b`h; w:where (idx>=0)&idx<n; @[h;idx w;:;b[`h] w]}[r;start;n]/[h;ix];
    `gps`rate`h!(start;r;h)
 }

/ whiten a stretch as the noise library does (64 s chunks at 2048 Hz, own PSD, middle 52 s kept), returning the
/ kept parts joined into one stream starting `margin` seconds in (chunks containing missing samples are zeroed)
.qs.gw.whitenStream:{[strain;o]
    o:.qs.gw.noiseDefaults,o;
    x:strain`h; rate:strain`rate; len:rate*o`chunk; stride:rate*o[`chunk]-2*o`margin;
    starts:stride*til 1+0|(count[x]-len) div stride;
    one:{[x;rate;o;len;s]
        c:x s+til len; r:o`rate; m:r*o`margin; keep:(r*o[`chunk])-2*m;
        if[any null c; :keep#0f];
        c:$[rate=r; c; .qs.sig.resample[c;rate;r]];
        psd:.qs.sig.welch[c;r;4*r;2*r];
        w:.qs.sig.whiten[c*.qs.sig.tukey[count c;0.1];r;`freq`asd!(psd`freq;sqrt psd`psd)];
        w m+til keep}[x;rate;o;len] peach starts;
    `gps`rate`w`ok!(strain[`gps]+o`margin;o`rate;raze one;raze {(count x)#not all 0=x} each one)
 }

/ a noise library (as .qs.gw.buildNoise's: event, det, gps, w, psd) from a run's bulk data between two GPS times:
/ the analysable segments' chunks in both detectors, away from catalogued events (by o`guard seconds). For training
/ on the noise of the run being searched, from days that aren't searched
.qs.gw.bulkNoise:{[run;start;stop;o]
    o:.qs.gw.noiseDefaults,o;
    segs:.qs.gw.analysable[run;start;stop;o`chunk];
    avoid:exec gps from .qs.gw.events[] where gps within (start-60;stop+60);
    t:raze {[run;avoid;o;seg]
        raze {[run;avoid;o;seg;d] c:.qs.gw.noiseChunks[.qs.gw.bulkStretch[run;d;seg`start;seg`end];avoid;o];
            ([] event:(count c)#run; det:(count c)#d),'c}[run;avoid;o;seg] each `H1`L1}[run;avoid;o] each segs;
    $[count t; t; ([] event:`symbol$(); det:`symbol$(); gps:`float$(); w:(); psd:())]
 }

/ both detectors of [start, stop) whitened for the search (S2.1)
.qs.gw.segmentStreams:{[run;start;stop]
    {[run;start;stop;d] .qs.gw.whitenStream[.qs.gw.bulkStretch[run;d;start;stop];()!()]}[run;start;stop] each `H1`L1
 }

/ search whitened streams (H1;L1) with model m: score 1 s windows every `step` seconds (skipping windows touching
/ a zeroed chunk), L1 optionally slid by `shift` seconds against H1 (circularly, for background), and keep
/ triggers: local maxima of the logit at least `sep` s apart. Returns `triggers`seconds: a table of window start
/ (GPS), the merger time implied at `merger` of the window, score, logit and each detector's largest whitened
/ amplitude in the window (ampH1, ampL1: Gaussian noise reaches about 4; a loud BBH about 6, a glitch far more);
/ and the time searched (windows x step)
.qs.gw.searchStreams:{[m;ab;o]
    o:(`step`sep`merger`shift!(0.125;1;0.75;0)),o;
    a:ab 0; b:ab 1; r:a`rate; len:r; n:min count each (a`w;b`w);
    none:`triggers`seconds!(([] gps:`float$(); merger:`float$(); score:`float$(); logit:`float$(); ampH1:`float$(); ampL1:`float$());0f);
    if[n<len; :none];
    sh:("j"$r*o`shift) mod n; st:"j"$o[`step]*r;
    s:st*til 1+(n-len) div st;
    s:s where {[a;b;len;n;sh;s] (all a[`ok] s+til len) and all b[`ok] (s+sh+til len) mod n}[a;b;len;n;sh] each s;
    if[not count s; :none];
    X:{[a;b;len;n;sh;s] (a[`w] s+til len),b[`w] (s+sh+til len) mod n}[a;b;len;n;sh] each s;
    z:first each .qs.predictRaw[m;X];
    w:"j"$o[`sep]%o`step;
    keep:where {[z;w;i] z[i]=max z (0|i-w)+til 1+(2*w)&(count z)-0|i-w}[z;w] each til count z;
    t:a[`gps]+s%r;
    amp:{[Xk;i] max each abs Xk[;i]}[X keep] each (til len;len+til len);
    `triggers`seconds!(([] gps:t keep; merger:t[keep]+o[`merger]*len%r; score:reciprocal 1+exp neg z keep; logit:z keep;
        ampH1:amp 0; ampL1:amp 1);o[`step]*count s)
 }
.qs.gw.searchSegment:{[m;run;start;stop;o] .qs.gw.searchStreams[m;.qs.gw.segmentStreams[run;start;stop];o]}

/ Newtonian stationary-phase time-to-coalescence for an inspiral chirp mass in solar masses.
.qs.gw.newtonianTime:{[frequency;chirpMass]
    if[frequency<=0 or chirpMass<=0; '"frequency and chirp mass must be positive"];
    massSeconds:4.9254909476412675e-6*chirpMass;
    (5%256)*(massSeconds xexp (-5%3))*((.qs.sig.pi*frequency) xexp (-8%3))
 }

.qs.gw.shellQuote:{[path] "'",ssr[path;"'";"'\\''"],"'"}
.qs.gw.words:{[line] words:" " vs line; words where 0<count each words}

/ Load a 4 kHz GWOSC txt.gz file as `gps`rate`h and cache the parsed dictionary beside it.
.qs.gw.strain:{[file]
    path:1 _ string file;
    cachePath:raze (path;".qcache");
    cache:hsym `$raze (":";cachePath);
     if[not ()~key hsym cache; :get hsym cache];
     if[()~key hsym `$path; '"strain file not found: ",path];
    rawPath:raze (path;".";string .z.i;".raw");
    raw:hsym `$raze (":";rawPath);
    shellCommand:raze ("gzip -dc -- ";.qs.gw.shellQuote[path];" > ";.qs.gw.shellQuote[1 _ string raw]);
    system raze ("env sh -c ";.qs.gw.shellQuote[shellCommand]);
     lines:read0 hsym raw;
    system raze ("env rm -f -- ";.qs.gw.shellQuote[1 _ string raw]);
     if[3>=count lines; '"strain file has fewer than three header lines"];
     rateWords:.qs.gw.words lines 1;
     gpsWords:.qs.gw.words lines 2;
     if[not ("4096" in rateWords) or not ("GPS" in gpsWords); '"unrecognized GWOSC strain header"];
     rate:"J"$enlist rateWords[rateWords?"4096"];
     gps:"J"$enlist gpsWords[1+gpsWords?"GPS"];
     samples:"F"$3_lines;
     result:`gps`rate`h!(first gps;first rate;samples);
     cache set result;
     result
 }