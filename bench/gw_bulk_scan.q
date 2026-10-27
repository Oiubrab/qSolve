/ open-search plan S2-S3: search a run's converted bulk data with a trained CNN. Analysable segments (both
/ detectors usable, at least 64 s), zero-lag triggers, a time-slide background over the same segments, a
/ false-alarm rate per trigger, and the catalogue cross-match
/   q bench/gw_bulk_scan.q -s 4 -q -run O4a -start GPS -stop GPS [-model models/gw_cnn_1.qsm] [-slides 20]
/        [-slide 7] [-step 0.25] [-top 20] [-veto 15]
/ -veto: triggers where either detector's whitened amplitude exceeds it are glitches (GW231226, at network SNR 35,
/ peaks near 6), dropped from the background and marked at zero lag
/ needs the run's files: .venv/bin/python scripts/gwosc_bulk.py fetch --run O4a --start GPS --end GPS
/ writes data/gw/search/<run>-<start>-<stop>-<model>.{triggers,background,summary}.csv (summary: searched and
/ background seconds)
.gsb.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
{system "l ",.gsb.root,"/",x} each ("defineEquation.q";"data.q";"signal.q";"gw.q");
.qs.data.dir:.gsb.root,"/data";
system "c 40 200"; system "P 14";
a:.Q.opt .z.x;
g:{[a;k;d;f] $[k in key a; f a k; d]}[a];
run:g[`run;`O4a;{`$first x}]; start:g[`start;0n;{"F"$first x}]; stop:g[`stop;0n;{"F"$first x}];
if[null start+stop; -2 "give -start and -stop (GPS)"; exit 1];
mf:g[`model;"models/gw_cnn_1.qsm";first]; m:.qs.load hsym `$.gsb.root,"/",mf;
slides:g[`slides;20;{"J"$first x}]; slide:g[`slide;7f;{"F"$first x}];
o:`step`sep`merger!(g[`step;0.25;{"F"$first x}];1;0.75);
top:g[`top;20;{"J"$first x}]; veto:g[`veto;15f;{"F"$first x}];

segs:.qs.gw.analysable[run;start;stop;64];
-1 (string count segs)," analysable segments, ",(string sum segs`duration)," s (",(string (sum segs`duration)%3600)," h) of ",
    (string stop-start)," s";
if[not count segs; exit 0];

/ per segment: whiten once, score at zero lag and at each slide whose shift (mod the segment) stays at least 2 s
/ from zero lag
t0:.z.p;
res:{[m;run;o;slides;slide;seg]
    ab:.qs.gw.segmentStreams[run;seg`start;seg`end];
    n:(min count each ab@\:`w)%ab[0;`rate];
    z:.qs.gw.searchStreams[m;ab;o];
    sh:slide*1+til slides; sh:sh where ((sh mod n) within (2;n-2));
    bg:{[m;ab;o;s] r:.qs.gw.searchStreams[m;ab;o,enlist[`shift]!enlist s]; r[`triggers]:update shift:s from r`triggers; r}[m;ab;o] each sh;
    -1 "  segment ",(string seg`start)," ",(string seg`duration)," s: ",(string count z`triggers)," triggers, ",
        (string count sh)," slides";
    `triggers`seconds`background`bgSeconds!(z`triggers;z`seconds;raze bg@\:`triggers;sum bg@\:`seconds)
 }[m;run;o;slides;slide] each segs;
-1 "searched in ",string .z.p-t0;

trig:raze res@\:`triggers; secs:sum res@\:`seconds;
bg:raze res@\:`background; bgs:sum res@\:`bgSeconds;
-1 "zero lag: ",(string count trig)," triggers in ",(string secs%3600)," h; background: ",(string count bg),
    " triggers in ",(string bgs%3600)," h";

trig:update vetoed:veto<ampH1|ampL1 from trig; bg:update vetoed:veto<ampH1|ampL1 from bg;
-1 "vetoed by amplitude over ",(string veto),": ",(string sum trig`vetoed)," zero-lag, ",(string sum bg`vetoed)," background";

/ catalogued events in the searched stretch: the loudest trigger within 1 s of each (on the implied merger time)
ev:select name,gps,mass1,mass2,snr,pAstro from .qs.gw.events[] where gps within (start;stop);

/ false alarms per hour at a trigger's logit, from the background without the catalogued events slid against noise
trig:.qs.gw.far[trig;bg;bgs;ev`gps];
inSeg:{[segs;t] any (t>=segs`start)&t<segs`end}[segs];
ev:update analysable:inSeg each gps from ev;
hit:{[trig;t] r:`logit xdesc select from trig where 1>abs merger-t;
    $[count r; first r; first 0#trig]}[trig] each ev`gps;
if[count ev; ev:ev,'select trigger:merger, logit, far, vetoed from hit];
-1 ""; -1 "catalogued events in the stretch:"; show ev;
trig:update known:{[ev;t] any 1>abs t-ev`gps}[ev] each merger from trig;

-1 ""; -1 "loudest unvetoed zero-lag triggers (far: false alarms per hour; bound: louder than all background):";
show top#`logit xdesc select from trig where not vetoed;
-1 ""; -1 "unvetoed triggers by false-alarm rate:";
show select n:count i by far:{$[x<0.1;`$"<0.1/h";x<1;`$"0.1-1/h";x<10;`$"1-10/h";`$">=10/h"]} each far from trig where not vetoed;

dir:.gsb.root,"/data/gw/search"; system "env mkdir -p ",dir;
name:dir,"/",("-" sv (string run;string "j"$start;string "j"$stop;first "." vs last "/" vs mf));
(hsym `$name,".triggers.csv") 0: csv 0: trig;
(hsym `$name,".background.csv") 0: csv 0: bg;
(hsym `$name,".summary.csv") 0: csv 0: ([] run:enlist run; start:start; stop:stop; model:enlist `$mf; slides:slides;
    slide:slide; step:o`step; veto:veto; seconds:secs; bgSeconds:bgs);
-1 "wrote ",name,".{triggers,background,summary}.csv";
\\
