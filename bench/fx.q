/ FX forecasting experiment (PLAN.md Phase 9): can the model forecast daily exchange-rate returns better
/ than "no change"?   q bench/fx.q [-s 4]        needs scripts/fetch_data.sh fx
/ every score is on later data than the model trained on; skill = 1 - MSE(model)/MSE(no change)

.fxb.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
{system "l ",.fxb.root,"/",x} each ("defineEquation.q";"data.q";"fx.q");
system "c 60 220";

.fxb.file:`$":",.fxb.root,"/data/fx/eurofxref-hist.csv";
if[()~key .fxb.file; -2 "no ECB data: run scripts/fetch_data.sh fx"; exit 1];
.fxb.t:.qs.fx.ecb .fxb.file;
.fxb.majors:`EURUSD`GBPUSD`USDJPY`USDCHF`AUDUSD`USDCAD;
.fxb.in:.qs.fx.inputs[.fxb.t;.fxb.majors];
-1 "ECB rates: ",(string count .fxb.t)," days, ",(string first .fxb.t`date)," to ",string last .fxb.t`date;

/ model configurations: a linear model (no hidden layer: an autoregression) and a small network
.fxb.base:`out`optimiser`lr`epochs`batch`seed`verbose!(`linear;`adam;0.001;30;64;42;0);
.fxb.linear:.fxb.base,`hidden`loss!(`long$();`mse);
.fxb.mlp:.fxb.base,`hidden`act`loss`dropout!(enlist 16;`tanh;`mse;0.1);

/ one experiment: inputs, target, lags, horizon, model; train on the first 80% of days, score on the rest
/ (.fxb.runT also applies tf to the windowed data before training, for the shuffled control)
.fxb.run:{[name;inputs;target;lags;horizon;spec] .fxb.runT[name;inputs;target;lags;horizon;spec;::]};
.fxb.runT:{[name;inputs;target;lags;horizon;spec;tf]
    system "S 42";
    d:tf .qs.data.lagged[inputs;target;lags;horizon;0.8];
    s:.qs.fx.fit[d;spec];
    `experiment`target`inputs`lags`horizon`model`trainDays`testDays`testFrom`skill`skillMean`direction`upShare!(
        name;target;count inputs;lags;horizon;$[count spec`hidden;`mlp;`linear];count d`Y;count d`Yval;
        .fxb.t[`date] first d`tVal;s`skill;s`skillMean;s`direction;s`upShare)
 };

.fxb.own:{[p] enlist[p]!enlist .fxb.in p};
.fxb.results:(
    .fxb.run["own history";.fxb.own `EURUSD;`EURUSD;10;1;.fxb.linear];
    .fxb.run["own history";.fxb.own `EURUSD;`EURUSD;10;1;.fxb.mlp];
    .fxb.run["6 majors";.fxb.in;`EURUSD;5;1;.fxb.linear];
    .fxb.run["6 majors";.fxb.in;`EURUSD;5;1;.fxb.mlp];
    .fxb.run["6 majors, mae loss";.fxb.in;`EURUSD;5;1;@[.fxb.mlp;`loss;:;`mae]];
    .fxb.run["6 majors, 5 days";.fxb.in;`EURUSD;5;5;.fxb.mlp];
    .fxb.run["own history";.fxb.own `GBPUSD;`GBPUSD;10;1;.fxb.mlp];
    .fxb.run["6 majors";.fxb.in;`GBPUSD;5;1;.fxb.mlp];
    / controls: shuffled training targets (no signal) and the answer leaked as an input (all signal)
    .fxb.runT["control: shuffled";.fxb.in;`EURUSD;5;1;.fxb.mlp;.qs.fx.shuffled];
    .fxb.run["control: answer leaked";.qs.fx.leaky[.fxb.in;`EURUSD];`EURUSD;5;1;.fxb.mlp]);

-1 "";
-1 "Next-day forecasts, scored on the last 20% of days (skill > 0 beats 'no change'; direction is the share of";
-1 "days where the forecast's sign was right, to compare with upShare, the share of up days):";
show select experiment,target,inputs,lags,horizon,model,testFrom,testDays,skill,skillMean,direction,upShare from .fxb.results;

/ up/down classification: does the model call the direction better than always guessing the more common one?
.fxb.updown:{[inputs;target;lags]
    d:.qs.data.lagged[inputs;target;lags;1;0.8];
    d[`Y]:.qs.data.oneHot[2;"j"$0<d`yraw]; d[`Yval]:.qs.data.oneHot[2;"j"$0<d`yrawVal];
    m:.qs.init[count first d`X;16 2;`tanh`softmax;42];
    r:.qs.train[m;`X`Y`Xval`Yval#d;`loss`optimiser`lr`epochs`batch`verbose!(`xent;`adam;0.001;30;64;0)];
    acc:.qs.evaluate[r`model;d`Xval;d`Yval;`xent]`acc;
    up:avg 0<d`yrawVal;
    `target`accuracy`alwaysMoreCommon!(target;acc;up|1-up)
 };
-1 "";
-1 "Up/down classification from the 6 majors' last 5 days, on the same test days:";
show (.fxb.updown[.fxb.in;`EURUSD;5];.fxb.updown[.fxb.in;`GBPUSD;5]);

/ walk-forward: retrain each year on every earlier year and test on that year alone
.fxb.years:2010+til 1+(`year$last .fxb.t`date)-2010;
.fxb.walkFor:{[spec] .qs.fx.walk[.fxb.in;`EURUSD;5;.fxb.t`date;.fxb.years;0N;spec]};
.fxb.wl:.fxb.walkFor .fxb.linear;
.fxb.wm:.fxb.walkFor .fxb.mlp;
.fxb.walk:([] year:.fxb.wl`year; testDays:.fxb.wl`n; skillLinear:.fxb.wl`skill; skillMLP:.fxb.wm`skill; directionMLP:.fxb.wm`direction; upShare:.fxb.wm`upShare);
-1 "";
-1 "Walk-forward, EUR/USD next day from the 6 majors: retrained each year on all earlier years:";
show .fxb.walk;
-1 "";
-1 "years with positive skill: linear ",(string sum 0<.fxb.walk`skillLinear),", MLP ",(string sum 0<.fxb.walk`skillMLP)," of ",string count .fxb.walk;
-1 "mean skill: linear ",(string avg .fxb.walk`skillLinear),", MLP ",(string avg .fxb.walk`skillMLP);
-1 "mean direction (MLP) ",(string avg .fxb.walk`directionMLP)," against mean upShare ",string avg .fxb.walk`upShare;
exit 0
