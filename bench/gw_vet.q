/ open-search plan S4.1: vet the loudest uncatalogued triggers of a searched stretch (bench/gw_bulk_scan.q's
/ results) with the matched-filter follow-up and the data-quality flags
/   q bench/gw_vet.q -s 4 -q -run O4a -start GPS -stop GPS [-model gw_cnn_4] [-top 5] [-far 1]
/ vets up to `top` unvetoed, uncatalogued triggers with false-alarm rates under `far` per hour. A candidate worth a
/ closer look has both detectors' SNRs above their own background in the stretch, an L1-H1 time difference within
/ 12 ms, and none of the flags that matter for a compact-binary search off (DATA, CBC_CAT1-3, NO_CBC_HW_INJ; the
/ continuous-wave hardware injection, NO_CW_HW_INJ off, runs most of the time and is harmless here)
.gvt.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
{system "l ",.gvt.root,"/",x} each ("defineEquation.q";"data.q";"signal.q";"gw.q");
.qs.data.dir:.gvt.root,"/data";
system "c 40 220"; system "P 12";
a:.Q.opt .z.x;
g:{[a;k;d;f] $[k in key a; f a k; d]}[a];
run:g[`run;`O4a;{`$first x}]; start:g[`start;0n;{"J"$first x}]; stop:g[`stop;0n;{"J"$first x}];
mn:g[`model;"gw_cnn_4";first]; top:g[`top;5;{"J"$first x}]; mx:g[`far;1f;{"F"$first x}];
base:.gvt.root,"/data/gw/search/","-" sv (string run;string start;string stop;mn);
t:("FFFFFFBFBB";enlist ",") 0: hsym `$base,".triggers.csv";
c:top#`far`logit xasc select from t where not vetoed, not known, far<mx;
-1 (string count c)," triggers to vet (of ",(string count t)," in ",base,")";
if[not count c; exit 0];
vet:{[run;r]
    f:.qs.gw.followUp[run;r`merger;()!()]; b:first f;
    `gps`far`logit`m1`m2`snrH1`snrL1`network`dt`bgH1`bgL1`flagsH1`flagsL1!(r`merger;r`far;r`logit;b`m1;b`m2;
        b`snrH1;b`snrL1;b`network;b`dt;b`bgH1;b`bgL1;
        " " sv string .qs.gw.flagsAt[run;`H1;r`merger];" " sv string .qs.gw.flagsAt[run;`L1;r`merger])}[run];
v:vet each c;
cbc:`noData`DATA`CBC_CAT1`CBC_CAT2`CBC_CAT3`NO_CBC_HW_INJ;
v:update cbcClean:{[cbc;h;l] not any cbc in `$" " vs h," ",l}[cbc]'[flagsH1;flagsL1] from v;
v:update consistent:(snrH1>bgH1)&(snrL1>bgL1)&(0.012>abs dt)&cbcClean from v;
show v;
(hsym `$base,".vetted.csv") 0: csv 0: v;
-1 "wrote ",base,".vetted.csv";
\\
