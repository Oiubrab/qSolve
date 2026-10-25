/ FX slow-trend research (plans/inProgress/fx_trends_plan.md): does months-to-years structure predict the next
/ weeks to months?   q bench/fx_trends.q [-s 4]       needs scripts/fetch_data.sh fx
/ every forecast is out of sample (walk-forward by year, purged by the horizon); skill = 1 - MSE(model)/MSE(zero change)
/ The experiment list is fixed here before it is run; K counts every configuration scored (F-D8).

.fxt.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
{system "l ",.fxt.root,"/",x} each ("defineEquation.q";"data.q";"fx.q");
system "c 400 250";

.fxt.file:`$":",.fxt.root,"/data/fx/eurofxref-hist.csv";
if[()~key .fxt.file; -2 "no ECB data: run scripts/fetch_data.sh fx"; exit 1];
.fxt.t:.qs.fx.ecb .fxt.file; .fxt.dates:.fxt.t`date;
.fxt.years:2010+til 1+(`year$last .fxt.dates)-2010;
.fxt.pairs:`EURUSD`GBPUSD`USDJPY`USDCHF`AUDUSD`USDCAD;
.fxt.horizons:5 20 60 120;
.fxt.rate:.fxt.pairs!.qs.fx.rate[.fxt.t] each .fxt.pairs;
.fxt.feat:{[d;r] .qs.fx.features[r;d]}[.fxt.dates] each .fxt.rate;
-1 "ECB rates: ",(string count .fxt.t)," days, ",(string first .fxt.dates)," to ",(string last .fxt.dates),"; test years ",(string first .fxt.years),"-",string last .fxt.years;

.fxt.base:`optimiser`lr`epochs`batch`seed`verbose!(`adam;0.001;30;64;42;0);
.fxt.linear:.fxt.base,`out`hidden`loss`l2!(`linear;`long$();`mse;0.1);
.fxt.mlp:.fxt.base,`out`hidden`act`loss`dropout!(`linear;enlist 16;`tanh;`mse;0.1);

.fxt.rowsOf:{[pair;h] .qs.fx.rows[.fxt.feat pair;.qs.fx.returns .fxt.rate pair;h]};

/ results: one dict per configuration; walks keeps the forecasts of each for the later analyses
.fxt.res:(); .fxt.walks:(`symbol$())!(); .fxt.K:0;
.fxt.record:{[model;pair;h;counted;w]
    s:.qs.fx.summary[w`p;w`y;20|h;500];
    by:.qs.fx.byYear[w`p;w`y;w`t;.fxt.dates];
    / pooled skill without the single best year, to see whether one year carries it
    yr:`year$.fxt.dates w`t; k:where yr<>(by`year) first idesc by`skill; e:w[`p][k]-w[`y] k;
    row:(`model`pair`h`counted!(model;pair;h;counted)),s,`yearsPos`years`skillExBest!(sum 0<by`skill;count by;1-(avg e*e)%avg w[`y][k]*w[`y] k);
    .fxt.res,:enlist row;
    .fxt.walks[`$"/" sv (string model;string pair;string h)]:w;
    if[counted; .fxt.K+:1];
 };
.fxt.show:{[t] show select model,pair,h,n,skill,skillLo,skillHi,pNoSkill,ic,direction,upShare,yearsPos,years,skillExBest from t};

/ F2.3 positive control: next-20-day realised variance (log) from the same features; skill is against the training mean
-1 "";
-1 "== F2.3 volatility control: forecast log of the next 20 days' summed squared returns, skill vs the training mean ==";
.fxt.vol:{[pair]
    r:.qs.fx.returns .fxt.rate pair; rows:.qs.fx.rows[.fxt.feat pair;r*r;20]; rows[`Y]:log rows`Y;
    system "S 42";
    w:.qs.fx.walkRows[rows;.fxt.dates;.fxt.years;20;0N;::;.fxt.linear];
    `pair`meanSkillVsMean`yearsPositive`years!(pair;avg w[`years]`skillVsMean;sum 0<w[`years]`skillVsMean;count w`years)
 };
show .fxt.vol each .fxt.pairs;

/ F3.1-F3.3: fixed rules, linear model, small MLP; every pair and horizon
-1 "";
-1 "== F3.1-F3.3 rules, linear and MLP: 6 pairs x horizons ",(" " sv string .fxt.horizons)," ==";
.fxt.main:{[pair;h]
    rows:.fxt.rowsOf[pair;h];
    .fxt.record[`trend;pair;h;1b;.qs.fx.walkRule[rows;.fxt.dates;.fxt.years;h;`m250;1]];
    .fxt.record[`reversion;pair;h;1b;.qs.fx.walkRule[rows;.fxt.dates;.fxt.years;h;`d750;-1]];
    system "S 42"; .fxt.record[`linear;pair;h;1b;.qs.fx.walkRows[rows;.fxt.dates;.fxt.years;h;0N;::;.fxt.linear]];
    system "S 42"; .fxt.record[`mlp;pair;h;1b;.qs.fx.walkRows[rows;.fxt.dates;.fxt.years;h;0N;::;.fxt.mlp]];
 };
.fxt.main .' .fxt.pairs cross .fxt.horizons;
.fxt.show select from .fxt.res where model in `trend`reversion`linear`mlp;

/ F3.4 ablation: linear model with each feature group left out
-1 "";
-1 "== F3.4 ablation: linear model without one feature group (horizons 20 and 60) ==";
.fxt.ablate:{[pair;h;g]
    rows:.fxt.rowsOf[pair;h]; rows:.qs.fx.keep[rows;.qs.fx.without[.fxt.feat pair;enlist g]];
    system "S 42";
    .fxt.record[`$"linear_no_",string g;pair;h;1b;.qs.fx.walkRows[rows;.fxt.dates;.fxt.years;h;0N;::;.fxt.linear]];
 };
{[p;h] .fxt.ablate[p;h] each `mom`rev`vol`season} .' .fxt.pairs cross 20 60;
.fxt.abl:select skill:avg skill,ic:avg ic,yearsPos:avg yearsPos by h,model from .fxt.res where model in `linear`linear_no_mom`linear_no_rev`linear_no_vol`linear_no_season,h in 20 60;
show .fxt.abl;

/ F3.5 pooled model: one model for all six pairs, scored per pair
-1 "";
-1 "== F3.5 pooled models (one model for 6 pairs), scored per pair ==";
.fxt.pooled:{[h;name;spec]
    system "S 42";
    w:.qs.fx.walkPooled[.fxt.pairs!.fxt.rowsOf[;h] each .fxt.pairs;.fxt.dates;.fxt.years;h;::;spec];
    {[name;h;w;pair] k:where w[`pair]=pair; .fxt.record[name;pair;h;1b;`p`y`t!(w[`p] k;w[`y] k;w[`t] k)]}[name;h;w] each .fxt.pairs;
 };
{.fxt.pooled[x;`pooledLinear;.fxt.linear]; .fxt.pooled[x;`pooledMlp;.fxt.mlp]} each .fxt.horizons;
.fxt.show select from .fxt.res where model in `pooledLinear`pooledMlp;

/ F3.6 direction: up/down classification of the sign of the h-day return, against the training-majority guess
-1 "";
-1 "== F3.6 up/down classification (MLP, softmax) vs always guessing the training majority ==";
.fxt.classify:{[pair;h]
    rows:.fxt.rowsOf[pair;h]; yr:`year$.fxt.dates rows`t;
    c:`rows`yr`t`purge`window!(rows;yr;rows`t;h;0N);
    one:{[c;h;y]
        sp:.qs.fx.split[c;y]; d:.qs.data.laggedSplit[c`rows;sp 0;sp 1];
        d[`Y]:.qs.data.oneHot[2;"j"$0<d`yraw]; d[`Yval]:.qs.data.oneHot[2;"j"$0<d`yrawVal];
        system "S 42";
        m:.qs.init[count first d`X;16 2;`tanh`softmax;42];
        r:.qs.train[m;`X`Y`Xval`Yval#d;`loss`optimiser`lr`epochs`batch`verbose`dropout!(`xent;`adam;0.001;30;64;0;0.1)];
        a:.qs.predict[r`model;d`Xval]; up:a[;1]>a[;0]; tru:0<d`yrawVal; maj:0<avg 0<d`yraw;
        (avg up=tru;avg maj=tru;count tru)
    }[c;h] each .fxt.years;
    n:sum one[;2]; acc:(sum one[;0]*one[;2])%n; base:(sum one[;1]*one[;2])%n;
    / the z-score counts each h days as one independent observation
    z:(acc-base)%sqrt (base*1-base)%n%h;
    `pair`h`accuracy`majority`z`yearsBetter`years!(pair;h;acc;base;z;sum one[;0]>one[;1];count one)
 };
.fxt.cls:.fxt.classify .' .fxt.pairs cross 20 60;
show .fxt.cls;
.fxt.K+:count .fxt.cls;

/ F4.3 controls on the real features: shuffled training targets (no skill) and the answer leaked (skill near 1)
-1 "";
-1 "== F4.3 controls on real features (not counted in K) ==";
.fxt.controls:{[pair;h]
    rows:.fxt.rowsOf[pair;h];
    system "S 42"; .fxt.record[`ctlShuffled;pair;h;0b;.qs.fx.walkRows[rows;.fxt.dates;.fxt.years;h;0N;.qs.fx.shuffled;.fxt.mlp]];
    system "S 42"; .fxt.record[`ctlLeak;pair;h;0b;.qs.fx.walkRows[.qs.fx.leakRows rows;.fxt.dates;.fxt.years;h;0N;::;.fxt.linear]];
 };
.fxt.controls .' `EURUSD`USDJPY cross 20 60;
.fxt.show select from .fxt.res where model in `ctlShuffled`ctlLeak;

/ F4.1-F4.2: significance across everything counted, and stability
-1 "";
-1 "== summary by model and horizon (6 pairs each; positive = pairs with skill > 0; sig = skill > 0 and one-sided p <= 0.05) ==";
show select configs:count i,skill:avg skill,ic:avg ic,positive:sum skill>0,sig:sum (skill>0) and pNoSkill<=0.05,yearsPos:avg yearsPos by model,h from .fxt.res where counted;
-1 "";
-1 "== F4.1 multiple comparisons ==";
.fxt.c:select from .fxt.res where counted;
.fxt.pos:select from .fxt.c where skill>0,pNoSkill<=0.05;
-1 "configurations scored (K, including ",(string count .fxt.cls)," classifiers): ",string .fxt.K;
-1 "forecast configurations: ",(string count .fxt.c),"; with skill > 0: ",(string sum 0<.fxt.c`skill),"; with skill > 0 and one-sided p <= 0.05: ",(string count .fxt.pos),"; expected by chance at 5% if none were real: about ",string .fxt.K*0.05;
-1 "smallest p: ",(string min .fxt.c`pNoSkill),"; Bonferroni-adjusted: ",string 1&.fxt.K*min .fxt.c`pNoSkill;
-1 "best 8 by skill:";
.fxt.show 8#`skill xdesc .fxt.c;
-1 "";
-1 "== F4.2 stability: skill by year for the 5 best configurations (and the share of configurations positive per year) ==";
.fxt.top:5#`skill xdesc .fxt.c;
.fxt.cfgKey:{[r] `$"/" sv (string r`model;string r`pair;string r`h)};
.fxt.tbl:{[r] k:.fxt.cfgKey r; w:.fxt.walks k; by:.qs.fx.byYear[w`p;w`y;w`t;.fxt.dates]; (`model`pair`h!(r`model;r`pair;r`h)),(`$string each by`year)!by`skill} each .fxt.top;
show .fxt.tbl;
.fxt.allBy:raze {[r] w:.fxt.walks .fxt.cfgKey r; by:.qs.fx.byYear[w`p;w`y;w`t;.fxt.dates]; flip `year`skill!(by`year;by`skill)} each .fxt.c;
show select configs:count i,positive:sum skill>0,meanSkill:avg skill by year from .fxt.allBy;

/ F4.4 cost-aware check: hold the sign of the forecast, re-decided every h days (no overlap), net of 2 basis points per
/ switch; annualised return in log terms, and the same for buying and holding
-1 "";
-1 "== F4.4 trading check, h = 20: sign of the forecast held for 20 days; 2bp per switch; annualised log return ==";
.fxt.trade:{[k]
    w:.fxt.walks k; n:count w`y; i:where 0=(til n) mod 20; pos:signum w[`p] i; r:w[`y] i;
    sw:0,1_abs deltas pos; net:(pos*r)-0.0002*sw;
    per:252%20;
    `config`gross`net`hold`positionLong!(k;per*avg pos*r;per*avg net;per*avg r;avg pos>0)
 };
.fxt.tk:.fxt.cfgKey each select from .fxt.res where counted,h=20,model in `trend`reversion`linear`mlp`pooledLinear`pooledMlp;
.fxt.tr:.fxt.trade each .fxt.tk;
.fxt.trs:`config xasc .fxt.tr;
show .fxt.trs;
-1 "mean gross ",(string avg .fxt.tr`gross),", net ",(string avg .fxt.tr`net),", buy-and-hold ",(string avg .fxt.tr`hold);

-1 "";
-1 "done";
exit 0
