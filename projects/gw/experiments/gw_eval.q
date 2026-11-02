/ G6: evaluate a trained detector on simulations in held-out test noise (GW150914's noise library, never used for
/ training or validation):
/   G6.1 a time-slide background (each chunk pair's H1 against other pairs' L1), giving logit thresholds for 1
/        false alarm per hour, per day and per month (when the background is long enough to measure them);
/   G6.2 detection efficiency by network SNR at those thresholds, for injections scored as the search scores them;
/   G6.3 efficiency by total mass and by mass ratio, and the share of windows with a simulated glitch in one
/        detector that score above each threshold.
/   q projects/gw/experiments/gw_eval.q -s 4 -q [-model gw_cnn_4] [-slides 76] [-injections 3000] [-snr 4 30]
/ writes results/eval/<model>.{thresholds,injections}.csv
.gev.root:{$[count x;x;"."]} "/" sv -4_ "/" vs string .z.f;
{system "l ",.gev.root,"/",x} each ("src/defineEquation.q";"src/data.q";"src/signal.q";"projects/gw/src/gw.q");
.qs.data.dir:.gev.root,"/data"; .qs.gw.home:.gev.root,"/projects/gw";
system "c 40 200";
a:.Q.opt .z.x;
g:{[a;k;d;f] $[k in key a; f a k; d]}[a];
mf:g[`model;"gw_cnn_4";first]; m:.qs.load .qs.gw.modelFile mf;
slides:g[`slides;0N;{"J"$first x}]; ninj:g[`injections;3000;{"J"$first x}]; snr:g[`snr;4 30f;"F"$];

test:.qs.gw.loadNoise`GW150914; tp:.qs.gw.pairs test;
-1 "model ",mf,"; test noise: ",(string count tp)," chunk pairs (",(string (count tp)*52%60)," min)";

/ G6.1
t0:.z.p;
bg:.qs.gw.libSlides[m;test;tp;`slides`step`sep!(slides;0.25;1)];
-1 "background: ",(string count bg`logit)," triggers in ",(string bg[`seconds]%3600)," h (",(string .z.p-t0),")";
rates:1f,(1%24),1%24*30;
th:([] rate:`$("1/hour";"1/day";"1/month"); perHour:rates; logit:.qs.gw.thresholds[bg`logit;bg`seconds;rates]);
-1 ""; -1 "thresholds (null: the background is too short to measure that rate):"; show th;

/ G6.2
system "S 7";
params:.qs.gw.drawParams[ninj;(enlist `snr)!enlist snr];
t0:.z.p;
iw:{[lib;pairs;p] .qs.gw.injectWindows[lib;pairs;()!();p]}[test;tp] peach params;
z:first each .qs.predictRaw[m;raze iw@\:`X];
res:params,'([] snrWindow:iw@\:`snrWindow; logit:max each 4 cut z);
-1 ""; -1 (string ninj)," injections scored (",(string .z.p-t0),")";
found:{[th;t] t,'flip (`$"found_",/:string th`rate)!{[t;x] $[null x; (count t)#0n; "f"$t[`logit]>x]}[t] each th`logit}[th];
res:found res;
fc:cols[res] where cols[res] like "found_*";
eff:{[fc;t;by] ?[t;();by;(`n,fc)!(enlist (count;`i)),{(avg;x)} each fc]}[fc];
-1 ""; -1 "efficiency by network SNR (bins of 2):"; show eff[res;(enlist `snr)!enlist (xbar;2;`snr)];

/ G6.3
-1 ""; -1 "efficiency by total mass (SNR 10 and up):";
show eff[select from res where snr>=10;(enlist `mtotal)!enlist (xbar;20;(+;`m1;`m2))];
-1 ""; -1 "efficiency by mass ratio m2/m1 (SNR 10 and up):";
show eff[select from res where snr>=10;(enlist `ratio)!enlist (xbar;0.25;(%;`m2;`m1))];
gb:.qs.gw.batch[test;tp;1000;`signal`glitchy!(0f;1f)];
gz:first each .qs.predictRaw[m;gb`X];
gl:update logit:gz from gb`meta;
-1 ""; -1 "single windows with a simulated glitch in one detector, share above each threshold:";
/ (a threshold the background can't measure gives a null share)
above:{[z;x] $[null x; 0n; avg z>x]};
show select n:count i, above1h:above[logit;th[`logit] 0], above1d:above[logit;th[`logit] 1] by kind from gl;

dir:.qs.gw.resultsDir[],"/eval"; system "env mkdir -p ",dir;
name:dir,"/",first "." vs last "/" vs mf;
(hsym `$name,".thresholds.csv") 0: csv 0: th,'([] seconds:(count th)#bg`seconds);
(hsym `$name,".injections.csv") 0: csv 0: res;
-1 "wrote ",name,".{thresholds,injections}.csv";
\\
