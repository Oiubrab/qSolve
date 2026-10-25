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