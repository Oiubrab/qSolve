/ G5.4: CNN detector (H1 and L1 as two channels of 1 s at 2048 Hz), trained and scored as bench/gw_mlp.q:
/ generated windows each epoch, a fixed validation batch, efficiency by SNR at a 1% false-alarm threshold
/   q bench/gw_cnn.q -s 4 -q [-filters 8 16 32] [-kernel 32 16 8] [-pool 4 4 4] [-hidden 32] [-snr 8 30]
/        [-merger 0.6 0.9] [-glitchy 0.25] [-epochs 20] [-lr 0.001] [-rows 6000] [-batch 128] [-save file]
/        [-mine 6] [-hard models/gw_cnn_1.qsm] [-real 0.5] [-patience 8] [-bulk O4a -bstart GPS -bstop GPS [-mix]]
/        [-single 0.25]
/   -bulk: train on that run's bulk noise between bstart and bstop (.qs.gw.bulkNoise, cached in data/gw/noise),
/   its last 20% of time held out for validation; -mix also keeps the O1-O3 training noise; -single: share of the
/   non-signal windows with a chirp in one detector only
/   -mine: real glitches above that whitened amplitude in the training noise become hard negatives (kind `real);
/   -hard: so do windows that model scores above 0.5; -real: their share of the glitchy windows; -patience: early
/   stopping on validation loss, keeping the best epoch
/ needs the noise library: .qs.gw.buildNoise[key .qs.gw.noiseSplit;()!()] (see plans/inProgress/gw_detection_plan.md)
.gwb.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
{system "l ",.gwb.root,"/",x} each ("defineEquation.q";"data.q";"signal.q";"gw.q");
.qs.data.dir:.gwb.root,"/data";
system "c 40 200";
a:.Q.opt .z.x;
g:{[a;k;d;f] $[k in key a; f a k; d]}[a];
hidden:g[`hidden;enlist 32;"J"$]; act:g[`act;`relu;{`$first x}];
layers:([] filters:g[`filters;8 16 32;"J"$]; kernel:g[`kernel;32 16 8;"J"$]; pool:g[`pool;4 4 4;"J"$]);
layers:update act:`relu from layers;
batch:g[`batch;128;{"J"$first x}];
o:`snr`signal`glitchy`merger`single!(g[`snr;8 30f;"F"$];0.5;g[`glitchy;0.25;{"F"$first x}];g[`merger;0.6 0.9;"F"$];g[`single;0f;{"F"$first x}]);
epochs:g[`epochs;20;{"J"$first x}]; lr:g[`lr;0.001;{"F"$first x}]; n:g[`rows;6000;{"J"$first x}];
train:raze .qs.gw.loadNoise each `GW151226`GW170104`GW190412`GW200322_091133;
val:.qs.gw.loadNoise`GW170814;
if[`bulk in key a;
    run:`$first a`bulk; bs:"F"$first a`bstart; be:"F"$first a`bstop;
    f:hsym `$.qs.gw.dataDir[],"/noise/","-" sv (string run;string "j"$bs;string "j"$be);
    bl:$[()~key f; [t:.qs.gw.bulkNoise[run;bs;be;()!()]; f set t; t]; get f];
    vcut:{x floor 0.8*count x} asc distinct bl`gps;
    -1 "bulk noise: ",(string (count bl) div 2)," chunk pairs, validation from GPS ",string vcut;
    train:$[`mix in key a; train; 0#train],select from bl where gps<vcut;
    val:select from bl where gps>=vcut];
tp:.qs.gw.pairs train; vp:.qs.gw.pairs val;
/ hard negatives: real glitches by amplitude and by an earlier model's score, in each library
mined:{[a;g;lib;pairs]
    t:$[`mine in key a; .qs.gw.mine[lib;"F"$first a`mine;0.5]; ()];
    t,:$[`hard in key a; .qs.gw.mineScores[.qs.load hsym `$first a`hard;lib;pairs;()!()]; ()];
    t}[a;g];
if[count tm:mined[train;tp]; o[`mined]:tm; o[`real]:g[`real;0.5;{"F"$first x}];
    -1 "hard negatives in the training noise: ",string count tm];
ov:o; if[count vm:mined[val;vp]; ov[`mined]:vm; -1 "hard negatives in the validation noise: ",string count vm];
system "S 1";
vb:.qs.gw.batch[val;vp;2000;ov];
gen:{[train;tp;o;n;e] .qs.gw.batch[train;tp;n;o]}[train;tp;o;n];
m:.qs.initConv[2;2048;layers;hidden,1;((count hidden)#act),`sigmoid;1];
show .qs.summary m;
patience:g[`patience;0N;{"J"$first x}];
r:.qs.train[m;`gen`Xval`Yval!(gen;vb`X;vb`Y);`epochs`batch`lr`loss`optimiser`l2`patience`keepBest`verbose!(epochs;batch;lr;`bce;`adam;0.0001;patience;not null patience;1)];
if[`save in key a; .qs.save[`$first a`save;r`model]];
p:first each .qs.predict[r`model;vb`X];
info:update score:p from vb`meta;
thr:{x (floor 0.99*count x)} asc exec score from info where kind<>`signal;
-1 "threshold for 1% false alarms: ",string thr;
show select n:count i, found:avg score>thr by 5 xbar snr from info where kind=`signal;
show select n:count i, falseAlarm:avg score>thr by kind from info where kind<>`signal;
-1 "validation loss ",string .qs.evaluate[r`model;vb`X;vb`Y;`bce]`loss;
\\
