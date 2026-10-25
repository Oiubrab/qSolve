/ FX data, multi-series windows, scoring and controls

.t.tmpFile:{[name] t:getenv`TMPDIR; hsym `$$[""~t;"/tmp";t],"/",name}

.t.add[`fx_ecb_loader;{
    f:.t.tmpFile "qsolve_ecb.csv";
    f 0: ("Date,USD,GBP,CYP,";"1999-01-06,1.2,0.8,N/A,";"1999-01-05,N/A,0.7,0.58,";"1999-01-04,1.0,0.5,0.6,");
    t:.qs.fx.ecb f;
    .t.eq[cols t;`date`USD`GBP`CYP];
    .t.eq[t`date;1999.01.04 1999.01.05 1999.01.06];
    .t.eq[t`USD;1 1 1.2];
    / a currency that stopped is not carried forward
    .t.eq[t`CYP;0.6 0.58 0n];
    .t.near[.qs.fx.rate[t;`GBPUSD];(1%0.5;1%0.7;1.2%0.8);1e-12];
    .t.near[.qs.fx.rate[t;`EURGBP];0.5 0.7 0.8;1e-12];
    .t.near[.qs.fx.rate[t;`USDEUR];1%1 1 1.2;1e-12];
    .t.throws[.qs.fx.rate[t];`GBPXXX];
    r:.qs.fx.returns 1 2 4f;
    .t.ok[null first r;"first return should be null"];
    .t.near[1_r;2#log 2;1e-12];
    hdel f;
 }];

.t.add[`fx_ecb_real;{
    f:.t.path "data/fx/eurofxref-hist.csv";
    if[()~key f; .t.skip "run scripts/fetch_data.sh fx"];
    t:.qs.fx.ecb f;
    .t.ok[5000<count t;"too few days"];
    .t.eq[first t`date;1999.01.04];
    .t.ok[all `USD`GBP`JPY`CHF in cols t;"major currencies missing"];
    .t.ok[not any null t`USD;"gaps in USD"];
 }];

/ row t holds inputs up to and including t; the target starts at t+1
.t.add[`fx_lagged_alignment;{
    a:"f"$til 20; b:100+"f"$til 20;
    r:.qs.data.laggedRows[`a`b!(a;b);`a;3;2];
    .t.eq[r`t;2+til 16];
    .t.eq[first r`X;0 1 2 100 101 102f];
    .t.eq[first r`Y;3+4f];
    / nulls drop their rows
    r:.qs.data.laggedRows[`a`b!(@[a;5;:;0n];b);`a;3;1];
    .t.ok[not any (r`t) in 4 5 6 7;"rows touching the null kept"];
    .t.ok[all 3 8 in r`t;"rows clear of the null dropped"];
    / scaling uses training rows only
    d:.qs.data.lagged[`a`b!(a;b);`a;3;1;0.5];
    .t.near[avg d`yraw;d`mu;1e-12];
    .t.ok[(avg d`yrawVal)>d`mu;"validation targets used for scaling"];
    .t.ok[all (d`t)<min d`tVal;"validation isn't after training"];
 }];

/ synthetic returns with a real signal: tomorrow's return is half of today's plus noise
.t.fxSignal:{[n] system "S 7"; e:0.01*.qs.data.normal n; {[a;x;y] (a*x)+y}[0.5]\[e]}

.t.add[`fx_score_finds_signal;{
    ys:.t.fxSignal 3000;
    d:.qs.data.lagged[enlist[`y]!enlist ys;`y;5;1;0.7];
    s:.qs.fx.fit[d;`hidden`out`loss`optimiser`lr`epochs`verbose!(`long$();`linear;`mse;`adam;0.01;20;0)];
    / an AR(1) with coefficient 0.5 explains a quarter of the variance
    .t.ok[s[`skill] within 0.15 0.35;"skill ",string s`skill];
    .t.ok[s[`direction]>0.6;"direction ",string s`direction];
 }];

.t.add[`fx_controls;{
    system "S 3"; ys:0.01*.qs.data.normal 3000;
    spec:`hidden`act`out`loss`optimiser`lr`epochs`verbose!(enlist 8;`tanh;`linear;`mse;`adam;0.01;15;0);
    / pure noise: no skill
    n:.qs.fx.fit[.qs.data.lagged[enlist[`y]!enlist ys;`y;5;1;0.7];spec];
    .t.ok[n[`skill]<0.02;"skill on noise ",string n`skill];
    / shuffled targets on the signal data: no skill either
    s:.qs.fx.fit[.qs.fx.shuffled .qs.data.lagged[enlist[`y]!enlist .t.fxSignal 3000;`y;5;1;0.7];spec];
    .t.ok[s[`skill]<0.02;"skill with shuffled targets ",string s`skill];
    / the answer leaked as an input: near-perfect skill
    l:.qs.fx.fit[.qs.data.lagged[.qs.fx.leaky[enlist[`y]!enlist ys;`y];`y;5;1;0.7];spec];
    .t.ok[l[`skill]>0.9;"skill with the answer leaked ",string l`skill];
 }];

.t.add[`fx_walk_forward;{
    ys:.t.fxSignal 1500;
    dates:2000.01.01+til 1500;
    w:.qs.fx.walk[enlist[`y]!enlist ys;`y;5;dates;2002 2003;0N;`hidden`out`loss`optimiser`lr`epochs`verbose!(`long$();`linear;`mse;`adam;0.01;10;0)];
    .t.eq[w`year;2002 2003];
    .t.ok[all 0.1<w`skill;"walk-forward skill ",-3!w`skill];
    .t.ok[(w[1;`train])>w[0;`train];"training window didn't grow"];
 }];

/ ---- slow-trend research (plans/inProgress/fx_trends_plan.md) ----

.t.fxLin:`out`optimiser`lr`epochs`batch`seed`verbose`hidden`loss`l2!(`linear;`adam;0.001;30;64;42;0;`long$();`mse;0.1)
.t.fxMlp:`out`optimiser`lr`epochs`batch`seed`verbose`hidden`act`loss`dropout!(`linear;`adam;0.001;30;64;42;0;enlist 16;`tanh;`mse;0.1)

/ synthetic daily returns with a slow drift that flips sign about every 120 days, buried in noise
.t.fxDrift:{[n;mu] fl:(n?1f)<1%120; sg:1-2*(sums fl) mod 2; (mu*sg)+0.005*.qs.data.normal n}

/ rows for a synthetic return series: horizon h, dates a day apart from 2000
.t.fxRowsOf:{[r;h] rate:exp sums r; dt:2000.01.01+til count r;
    `rows`dates!(.qs.fx.rows[.qs.fx.features[rate;dt];.qs.fx.returns rate;h];dt)}

.t.add[`fx_trend_features;{
    dt:2000.01.01+til 1000;
    f:.qs.fx.features[1000#1.5;dt];
    .t.eq[key f;`m5`m20`m60`m120`m250`d250`d750`v20`v60`sinM`cosM];
    / warm-up: the first values lack history
    .t.eq["j"$sum each null f;`m5`m20`m60`m120`m250`d250`d750`v20`v60`sinM`cosM!5 20 60 120 250 249 749 20 60 0 0];
    / a flat rate has no trend, no distance from its mean and no volatility
    .t.near[f[`m250] 250+til 750;0f;1e-12]; .t.near[f[`d750] 749+til 251;0f;1e-9]; .t.near[f[`v60] 60+til 940;0f;1e-6];
    / a constant growth rate: trailing returns are k steps of it, the distance from a mean is half the window
    g:.qs.fx.features[exp 0.001*"f"$til 1000;dt];
    .t.near[g[`m60] 60+til 940;0.06;1e-9];
    .t.near[g[`d250] 249+til 751;0.001*249%2;1e-9];
    .t.near[g[`v20] 20+til 980;0f;1e-6];
    / the month as a circle
    .t.near[f[`sinM] 0;sin 0f;1e-12]; .t.near[f[`cosM] 0;1f;1e-12];
 }];

/ features at time t must not change when later rates do
.t.add[`fx_trend_features_no_lookahead;{
    system "S 5"; n:1000; r:exp sums 0.01*.qs.data.normal n; dt:2000.01.01+til n;
    r2:@[r;601+til n-601;:;3f];
    a:.qs.fx.features[r;dt]; b:.qs.fx.features[r2;dt];
    .t.ok[all {[a;b;k] (601#a k)~601#b k}[a;b] each key a;"a feature at time t used a later rate"];
    .t.ok[not all {[a;b;k] a[k]~b k}[a;b] each key a;"perturbation changed nothing at all"];
 }];

/ the target of a row at t is the log return from t to t+h; a missing return drops every row whose target spans it
.t.add[`fx_trend_rows_alignment;{
    system "S 6"; rate:exp sums 0.01*.qs.data.normal 200; ret:.qs.fx.returns rate;
    f:enlist[`a]!enlist "f"$til 200;
    r:.qs.fx.rows[f;ret;5];
    .t.eq[r`names;enlist `a];
    .t.near[r`Y;(log rate[r[`t]+5])-log rate r`t;1e-12];
    .t.eq[raze r`X;"f"$r`t];
    ret2:@[ret;50;:;0n]; r2:.qs.fx.rows[f;ret2;5];
    .t.ok[not any (r2`t) within 45 49;"rows whose target spans the gap kept"];
    / (row 50 goes too: its own return, a column of the raw rows, is missing)
    .t.ok[all 44 51 in r2`t;"rows clear of the gap dropped"];
    / dropping and keeping feature groups
    rate:exp sums 0.01*.qs.data.normal 1200; ret:.qs.fx.returns rate;
    g:.qs.fx.features[rate;2000.01.01+til 1200];
    .t.eq[.qs.fx.without[g;`mom`season];`d250`d750`v20`v60];
    w:.qs.fx.rows[g;ret;5]; k:.qs.fx.keep[w;`m5`m20];
    .t.eq[k`names;`m5`m20]; .t.eq[count first k`X;2];
    .t.eq[.qs.fx.col[w;`m20];.qs.fx.col[k;`m20]];
 }];

/ a training row's target must end before the test year begins
.t.add[`fx_purged_split;{
    dt:2000.01.01+til 3000; t:til 3000;
    c:`rows`yr`t`purge`window!(0;`year$dt;t;20;0N);
    s:.qs.fx.split[c;2005]; i:s 0; j:s 1;
    .t.ok[all 2005=(c`yr) j;"test rows aren't all in the test year"];
    .t.ok[all ((c`yr) i)<2005;"training rows from the test year or later"];
    .t.ok[(max t i)+20<=min t j;"a training target overlaps the test period"];
    .t.eq[(min t j)-max t i;20];
    / no purge keeps the rows right up to the test year
    u:.qs.fx.split[@[c;`purge;:;0];2005]; .t.eq[(min t u 1)-max t u 0;1];
    / a window keeps only the last years
    w:.qs.fx.split[@[c;`window;:;2];2005]; .t.ok[all ((c`yr) w 0)>=2003;"window ignored"];
 }];

/ a slow drift hidden in noise is found; the same noise without the drift gives nothing
.t.add[`fx_trend_synthetic_truth;{
    system "S 11";
    a:.t.fxRowsOf[.t.fxDrift[7000;0.001];20];
    w:.qs.fx.walkRows[a`rows;a`dates;2010+til 9;20;0N;::;.t.fxLin];
    s:.qs.fx.summary[w`p;w`y;20;100];
    .t.ok[s[`ic]>0.15;"information coefficient with a drift ",string s`ic];
    .t.ok[s[`skill]>0.02;"skill with a drift ",string s`skill];
    .t.ok[s[`direction]>s`upShare;"direction no better than always-up"];
    b:.t.fxRowsOf[.t.fxDrift[7000;0f];20];
    v:.qs.fx.walkRows[b`rows;b`dates;2010+til 9;20;0N;::;.t.fxLin];
    n:.qs.fx.summary[v`p;v`y;20;100];
    .t.ok[n[`skill]<0.03;"skill without a drift ",string n`skill];
    .t.ok[n[`skillLo]<0;"interval says there is skill with no drift"];
    / the rows' timing: out-of-sample points are in the test years only
    .t.ok[all (`year$a[`dates] w`t) within 2010 2018;"forecasts outside the test years"];
    / skill by year agrees with the pool
    .t.eq[count .qs.fx.byYear[w`p;w`y;w`t;a`dates];9];
 }];

/ a random split with long overlapping targets leaks and shows fake skill on a random walk; the purged walk-forward doesn't
.t.add[`fx_trend_overlap_trap;{
    system "S 11";
    a:.t.fxRowsOf[.t.fxDrift[7000;0f];60];
    leak:.qs.fx.interleaved[a`rows;5;`out`optimiser`lr`epochs`batch`seed`verbose`hidden`act`loss!(`linear;`adam;0.001;60;64;42;0;enlist 32;`tanh;`mse)];
    .t.ok[leak[`skill]>0.2;"expected fake skill from the leak, got ",string leak`skill];
    w:.qs.fx.walkRows[a`rows;a`dates;2010+til 9;60;0N;::;.t.fxMlp];
    s:.qs.fx.summary[w`p;w`y;60;50];
    .t.ok[s[`skill]<0.05;"purged skill on a random walk ",string s`skill];
 }];

/ the fixed rules: a trend rule finds the drift, and on pure noise earns no more than buying and holding
.t.add[`fx_trend_rules;{
    system "S 11";
    a:.t.fxRowsOf[.t.fxDrift[7000;0.001];20];
    r:.qs.fx.walkRule[a`rows;a`dates;2010+til 9;20;`m60;1];
    s:.qs.fx.summary[r`p;r`y;20;50];
    .t.ok[s[`ic]>0.1;"trend rule's ic ",string s`ic];
    yrs:r`years;
    .t.ok[(avg yrs`ruleReturn)>(avg yrs`holdReturn)+0.002;"rule return no better than holding"];
    .t.eq[count r`sign;count r`p];
    .t.ok[all (abs r`sign)<=1;"sign rule outputs"];
    / betting against the feature reverses the rule
    m:.qs.fx.walkRule[a`rows;a`dates;2010+til 9;20;`m60;-1];
    .t.ok[(avg m[`years]`ruleReturn)<0;"reversed rule should lose"];
 }];

.t.add[`fx_trend_pooled;{
    system "S 11";
    a:.t.fxRowsOf[.t.fxDrift[5000;0.001];20]; b:.t.fxRowsOf[.t.fxDrift[5000;0.001];20];
    w:.qs.fx.walkPooled[`A`B!(a`rows;b`rows);a`dates;2008+til 5;20;::;.t.fxLin];
    .t.eq[distinct w`pair;`A`B];
    .t.eq[count each (w`pair;w`p;w`y;w`t);4#count w`p];
    i:where w[`pair]=`A;
    .t.ok[cor[w[`p] i;w[`y] i]>0.1;"pooled model found no signal"];
 }];

.t.add[`fx_trend_bootstrap;{
    system "S 2"; yy:0.01*.qs.data.normal 3000;
    / a perfect forecast
    s:.qs.fx.summary[yy;yy;20;50]; .t.near[s`skill;1f;1e-12]; .t.ok[s[`skillLo]>0.99;"interval of a perfect forecast"];
    / a tiny unrelated forecast has about zero skill, and an interval that covers it
    n:.qs.fx.summary[0.0003*.qs.data.normal 3000;yy;20;200];
    .t.ok[(n[`skillLo]<0.005) and n[`skillHi]>-0.005;"interval misses zero"];
    .t.ok[n[`pNoSkill]>0.05;"claimed skill from noise: ",string n`pNoSkill];
    / blocks keep serial dependence: overlapping targets make a wider interval than independent ones
    ov:sum each (0.01*.qs.data.normal 3000) (til 2981)+\:til 20; p:sum each (0.01*.qs.data.normal 3000) (til 2981)+\:til 20;
    wide:.qs.fx.summary[p;ov;20;200]; narrow:.qs.fx.summary[p;ov;1;200];
    .t.ok[(wide[`icHi]-wide`icLo)>(narrow[`icHi]-narrow`icLo);"block bootstrap interval not wider with overlap"];
 }];
