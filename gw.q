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