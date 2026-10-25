/ G7: scan the real data around every in-scope catalogued binary black hole with a trained detector
/   q bench/gw_events.q -s 4 -q [-model models/gw_cnn_1.qsm] [-threshold 0.71]
/ needs the events' 32 s files: scripts/fetch_gw.sh --test-events
/ the false-alarm rate is measured in the events' own (O3) data: L1 time-slid against H1 by 1-19 s in each event's
/ 20 clean seconds, windows touching either detector's event left out; each event's false-alarm rate is how often
/ that background produces a trigger scoring at least as high. Also reported: the earlier local criterion (above the
/ threshold and above everything else in the same 20 s). Events whose surroundings were training or validation
/ noise are flagged
.gwe.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
{system "l ",.gwe.root,"/",x} each ("defineEquation.q";"data.q";"signal.q";"gw.q");
.qs.data.dir:.gwe.root,"/data";
system "c 60 220";
a:.Q.opt .z.x;
mf:$[`model in key a; first a`model; .gwe.root,"/models/gw_cnn_1.qsm"];
thr:$[`threshold in key a; "F"$first a`threshold; 0.71];
m:.qs.load hsym `$mf;
ev:.qs.gw.confidentBBH[.qs.gw.events[];10;80;0.9];
have:{[e] not ()~key hsym `$.qs.gw.dataDir[],"/strain/",string e} each ev`name;
ev:select from ev where have;
-1 "scanning ",(string count ev)," events with ",mf;
r:{[m;e] .qs.gw.scanEvent[m;e`name;e`gps;()!()]}[m] each ev;
r:(`event`status`score`offset`background#/:r);
res:(select name,catalogue,gps,mass1,mass2,distance,snr from ev),'select status,score,offset,background from r;
res:update split:.qs.gw.noiseSplit name from res;
res:update found:(status=`ok)&(score>thr)&score>background from res;
show `snr xdesc select name,snr,mass1,mass2,split,status,score,offset,background,found from res;
-1 "";
-1 "measuring background by time slides (37 shifts per event)...";
bg:{[m;e] .qs.gw.slides[m;e`name;e`gps;1+0.5*til 37;()!()]}[m] each select name,gps from ev where name in exec name from res where status=`ok;
trig:desc raze bg@\:`triggers; hours:(sum bg@\:`seconds)%3600;
-1 "background: ",(string count trig)," triggers in ",(string hours)," hours";
show ([] perHour:1 0.2 0.1; threshold:{[trig;hours;r] trig 0|-1+floor r*hours}[trig;hours] each 1 0.2 0.1);
res:update far:{[trig;hours;sc] $[null sc;0n;(1|sum trig>=sc)%hours]}[trig;hours] each score from res;
res:update farUpper:{[trig;hours;sc] $[null sc;0n;(sum trig>=sc)%hours]}[trig;hours] each score from res;
show `snr xdesc select name,snr,mass1,mass2,split,score,offset,farPerHour:farUpper,found1h:farUpper<1,found5h:farUpper<0.2 from res where status=`ok;
ok:select from res where status=`ok, null split;
-1 "held-out events scanned: ",(string count ok)," (gaps: ",(string count select from res where status=`gap),")";
show select events:count i, localCriterion:sum found, farBelow1h:sum farUpper<1, farBelow5h:sum farUpper<0.2 by snrBin:5 xbar snr from ok;
\\
