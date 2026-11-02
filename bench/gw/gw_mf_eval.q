/ G6.2's comparison: the matched-filter baseline against a trained detector on the same held-out test noise
/ (GW150914's library) and the same injections. The matched filter uses .qs.gw.bank (41 PhenomB templates, 10-80 M☉,
/ zero spins) and ranks by network SNR (H1's plus L1's loudest within 12 ms, in quadrature); its thresholds come from
/ its own time-slide background, as the model's do. Run bench/gw/gw_eval.q for the model first (its thresholds).
/   q bench/gw/gw_mf_eval.q -s 4 -q [-model models/gw_cnn_4.qsm] [-slides 76] [-injections 3000] [-snr 4 30]
/ writes data/gw/eval/matched_filter.{thresholds,injections}.csv (the injections with both statistics)
.gmf.root:{$[count x;x;"."]} "/" sv -3_ "/" vs string .z.f;
{system "l ",.gmf.root,"/src/",x} each ("defineEquation.q";"data.q";"signal.q";"gw.q");
.qs.data.dir:.gmf.root,"/data";
system "c 40 200";
a:.Q.opt .z.x;
g:{[a;k;d;f] $[k in key a; f a k; d]}[a];
mf:g[`model;"models/gw_cnn_4.qsm";first]; m:.qs.load hsym `$.gmf.root,"/",mf;
slides:g[`slides;0N;{"J"$first x}]; ninj:g[`injections;3000;{"J"$first x}]; snr:g[`snr;4 30f;"F"$];
dir:.gmf.root,"/data/gw/eval"; mname:first "." vs last "/" vs mf;
cth:("SFFF";enlist ",") 0: hsym `$dir,"/",mname,".thresholds.csv";

test:.qs.gw.loadNoise`GW150914; tp:.qs.gw.pairs test;
r:.qs.gw.noiseDefaults`rate;
/ 2^17 samples hold a 52 s chunk and the longest template's wrap-around (about 6 s for 10+10 M☉ from 20 Hz); an
/ injection needs only 16 s around its merger (2^15)
b:.qs.gw.mfBank[131072;r;.qs.gw.bank;20]; bi:.qs.gw.mfBank[32768;r;.qs.gw.bank;20];
-1 "matched filter: ",(string count b`t)," templates; test noise: ",(string count tp)," chunk pairs";

t0:.z.p;
bg:.qs.gw.mfSlides[b;test;tp;(enlist `slides)!enlist slides];
-1 "background: ",(string count bg`snr)," triggers in ",(string bg[`seconds]%3600)," h (",(string .z.p-t0),")";
rates:1f,(1%24),1%24*30;
th:([] rate:`$("1/hour";"1/day";"1/month"); perHour:rates; snr:.qs.gw.thresholds[bg`snr;bg`seconds;rates]);
th:th,'([] logit:cth`logit);
-1 ""; -1 "thresholds: matched-filter network SNR, and the model's logit (from its own background):"; show th;

system "S 7";
params:.qs.gw.drawParams[ninj;(enlist `snr)!enlist snr];
t0:.z.p;
one:{[b;lib;pairs;p] c:.qs.gw.injectChunk[lib;pairs;()!();p];
    w:.qs.gw.chunkWindows[c;()!()]; w,(enlist `mf)!enlist .qs.gw.mfInject[b;lib;c;()!()]}[bi;test;tp];
iw:one peach params;
z:first each .qs.predictRaw[m;raze iw@\:`X];
res:params,'([] snrWindow:iw@\:`snrWindow; logit:max each 4 cut z; mf:iw@\:`mf);
-1 ""; -1 (string ninj)," injections scored by both (",(string .z.p-t0),")";
/ (a threshold the background can't measure gives null, not found: in q a null is below everything)
above:{[x;t] $[null t; (count x)#0n; "f"$x>t]};
res:update cnn1h:above[logit;th[`logit] 0], cnn1d:above[logit;th[`logit] 1], mf1h:above[mf;th[`snr] 0],
    mf1d:above[mf;th[`snr] 1] from res;
-1 ""; -1 "efficiency by network SNR (bins of 2), the model against the matched filter:";
show select n:count i, cnn1h:avg cnn1h, mf1h:avg mf1h, cnn1d:avg cnn1d, mf1d:avg mf1d by snr:2 xbar snr from res;
-1 ""; -1 "by total mass (SNR 10 and up):";
show select n:count i, cnn1h:avg cnn1h, mf1h:avg mf1h, cnn1d:avg cnn1d, mf1d:avg mf1d by mtotal:20 xbar m1+m2
    from res where snr>=10;
-1 ""; -1 "SNR at which each finds half the injections (linear between bins):";
half:{[t;c] e:?[t;();(enlist `s)!enlist (xbar;2;`snr);(enlist `e)!enlist (avg;c)]; s:exec s from e; v:exec e from e;
    i:first where v>=0.5; $[null i; 0n; i=0; first s; s[i-1]+2*(0.5-v i-1)%(v i)-v i-1]+1};
show ([] rate:`$("1/hour";"1/day"); cnn:half[res] each `cnn1h`cnn1d; mf:half[res] each `mf1h`mf1d);

(hsym `$dir,"/matched_filter.thresholds.csv") 0: csv 0: th,'([] seconds:(count th)#bg`seconds);
(hsym `$dir,"/matched_filter.injections.csv") 0: csv 0: res;
-1 "wrote ",dir,"/matched_filter.{thresholds,injections}.csv";
\\
