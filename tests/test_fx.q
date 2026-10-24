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
