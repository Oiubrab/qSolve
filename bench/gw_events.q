/ G7: scan the real data around every in-scope catalogued binary black hole with a trained detector
/   q bench/gw_events.q -s 4 -q [-model models/gw_cnn_1.qsm] [-threshold 0.71]
/ needs the events' 32 s files: scripts/fetch_gw.sh --test-events
/ an event counts as found when its best score near the catalogue time is above the threshold and above everything
/ else in the same 20 s of data; events whose surroundings were training or validation noise are flagged
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
ok:select from res where status=`ok, null split;
-1 "held-out events scanned: ",(string count ok)," (gaps: ",(string count select from res where status=`gap),")";
show select events:count i, found:sum found, share:avg found by snrBin:5 xbar snr from ok;
\\
