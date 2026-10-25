/ qSolve FX: exchange rates from the European Central Bank, currency pairs, and honest forecast scoring
/ needs defineEquation.q and data.q loaded first

/ fill gaps from the previous value, but not after a series' last value (currencies that stopped trading)
.qs.fx.fillGaps:{[x] ?[null reverse fills reverse x;0n;fills x]}

/ ECB euro reference rates (eurofxref-hist.csv): date and one column per currency, in units per euro,
/ oldest first; N/A becomes null and interior gaps are filled from the previous day
.qs.fx.ecb:{[file]
    l:read0 hsym file; l:l where 0<count each l;
    h:"," vs first l; n:count h where 0<count each h;
    t:(("D",(n-1)#"F"),(count[h]-n)#" ";enlist ",") 0: l;
    t:`Date xasc t;
    t:`date xcol t;
    @[t;1_cols t;.qs.fx.fillGaps]
 }

/ the rate of a pair from an ECB table, as quote currency per base currency: `GBPUSD is dollars per pound
.qs.fx.rate:{[t;pair]
    p:string pair; if[not 6=count p; '"a pair is 6 letters, e.g. GBPUSD"];
    c:{[t;c] $[c=`EUR; (count t)#1f; c in cols t; t c; '"no currency ",string c]}[t];
    c[`$3_p]%c[`$3#p]
 }

/ daily log returns, the first one null
.qs.fx.returns:{[x] 0n,1_deltas log "f"$x}

/ log returns of several pairs, as inputs for .qs.data.lagged
.qs.fx.inputs:{[t;pairs] pairs!.qs.fx.returns each .qs.fx.rate[t] each pairs}

/ how a model trained on d (from .qs.data.lagged) forecasts its validation returns, against naive forecasts:
/ zero change (the random-walk forecast) and the training mean; skill is 1 - MSE(model)/MSE(zero change), so
/ 0 is no better than "no change" and below 0 is worse; direction is how often the forecast's sign is right
/ (moves of exactly zero left out), against upShare, how often the market simply went up; skillMean is the
/ training mean's own skill against zero change, skillVsMean the model's skill against the training mean
/ (the one to use for targets that aren't centred on zero, like log variance), ic the correlation of forecast and outcome
.qs.fx.score:{[m;d]
    p:.qs.fx.forecast[m;d];
    y:d`yrawVal; tm:avg d`yraw;
    mse:{avg x*x};
    w:where y<>0;
    `n`mseModel`mseZero`mseMean`skill`skillMean`skillVsMean`ic`direction`upShare!(count y;mse p-y;mse y;mse y-tm;
        1-(mse p-y)%mse y;1-(mse y-tm)%mse y;1-(mse p-y)%mse y-tm;cor[p;y];avg (signum p w)=signum y w;avg 0<y w)
 }

/ train a model to configuration spec (hidden, act, out, loss and training options, as for .qs.spec) and score it
.qs.fx.fit:{[d;spec]
    c:.qs.spec[`X`Y#d;spec];
    m:.qs.init[count first d`X;c[`hidden],1;((count c`hidden)#c`act),c`out;(.qs.defaults,c)`seed];
    r:.qs.train[m;`X`Y`Xval`Yval#d;`hidden`act`out _ c];
    .qs.fx.score[r`model;d],enlist[`model]!enlist r`model
 }

/ controls: the same data with the training targets shuffled (no signal: skill should be about 0 or below)
.qs.fx.shuffled:{[d] k:count d`Y; i:neg[k]?k; d[`Y]:d[`Y] i; d[`yraw]:d[`yraw] i; d}

/ and with the answer leaked: the target's own next return added as an input (skill should be near 1)
.qs.fx.leaky:{[inputs;target] inputs,enlist[`leak]!enlist 1_(inputs target),0n}

/ walk-forward: for each test year, train on the years before it (all of them, or the last `window`) and
/ score on that year; returns one row per year
.qs.fx.walk:{[inputs;target;lags;dates;years;window;spec]
    r:.qs.data.laggedRows[inputs;target;lags;1];
    yr:`year$dates r`t;
    {[r;yr;window;spec;y]
        i:where (yr<y) and $[null window; 1b; yr>=y-window]; j:where yr=y;
        s:.qs.fx.fit[.qs.data.laggedSplit[r;i;j];spec];
        (enlist[`year]!enlist y),`train`n`skill`direction`upShare#(enlist[`train]!enlist count i),s
    }[r;yr;window;spec] each years
 }

/ ---- slow-trend research (plans/inProgress/fx_trends_plan.md) ----

/ feature groups: trailing returns over 5..250 days, distance from the 250 and 750 day mean (log), trailing
/ volatility of daily returns (20, 60 days), and month of the year as a circle
.qs.fx.groups:`mom`rev`vol`season!(`m5`m20`m60`m120`m250;`d250`d750;`v20`v60;`sinM`cosM)

/ null the first k values, whose windows are incomplete
.qs.fx.warm:{[k;x] @[x;til k&count x;:;0n]}

/ the features of a rate series as a dictionary name -> series (null until enough history); the value at time t
/ uses the rate up to and including t only
.qs.fx.features:{[rate;dates]
    lr:log "f"$rate; r:.qs.fx.returns rate; mo:-1+`mm$dates;
    mom:{[lr;k] lr-k xprev lr}[lr] each 5 20 60 120 250;
    rev:{[lr;w] .qs.fx.warm[w-1;lr-w mavg lr]}[lr] each 250 750;
    vol:{[r;w] m:w mavg r; .qs.fx.warm[w;sqrt 0f|(w mavg r*r)-m*m]}[0f^r] each 20 60;
    (raze value .qs.fx.groups)!mom,rev,vol,(sin 2*.qs.pi*mo%12;cos 2*.qs.pi*mo%12)
 }

/ rows for the model: the features at t (`X, ordered as the dictionary) and the sum of series over t+1..t+h
/ (`Y); series is daily log returns for return targets or their squares for variance; `t indexes the series
.qs.fx.rows:{[feats;series;h]
    r:.qs.data.laggedRows[(enlist[`target]!enlist series),feats;`target;1;h];
    r[`X]:1_/:r`X;
    r,enlist[`names]!enlist key feats
 }

/ the same rows with some feature groups left out
.qs.fx.without:{[feats;groups] (key feats) except raze .qs.fx.groups groups}
.qs.fx.keep:{[rows;names] i:where (rows`names) in names; @[rows;`X`names;:;(rows[`X][;i];(rows`names) i)]}

/ the feature column of rows as a list
.qs.fx.col:{[rows;name] rows[`X][;(rows`names)?name]}

/ rows with the target's own future sum added as an input (skill should be near 1)
.qs.fx.leakRows:{[rows] @[rows;`X`names;:;(rows[`X],'enlist each rows`Y;(rows`names),`leak)]}

/ forecasts of d's validation targets in the target's own units
.qs.fx.forecast:{[m;d] d[`mu]+d[`sd]*first each .qs.predict[m;d`Xval]}

/ which rows train and which test for year y: training rows are from earlier years (the last `window` of them if
/ not null) and, when purging, their targets must end before the first test row, since a target h days
/ long overlaps the h-1 rows before it
.qs.fx.split:{[c;y]
    j:where c[`yr]=y; t0:min c[`t] j;
    i:where (c[`yr]<y) and ((c[`t]+c`purge)<=t0) and $[null c`window; 1b; c[`yr]>=y-c`window];
    (i;j)
 }

/ walk-forward over test years for any rows from .qs.fx.rows: for each year train a model to spec on rows before
/ it (all of them or the last window years; purge is how many days to hold back before the test year, h to
/ avoid overlap, 0 to see the leak) and score it on the year; tf is applied to the training data (e.g. shuffled).
/ Returns `years (one row per year) and the pooled out-of-sample forecasts `p, outcomes `y and times `t
.qs.fx.walkRows:{[rows;dates;years;purge;window;tf;spec]
    c:`rows`yr`t`purge`window`tf`spec!(rows;`year$dates rows`t;rows`t;purge;window;tf;spec);
    r:.qs.fx.walkYear[c] each years;
    `years`p`y`t!(r[;0];raze r[;1];raze r[;2];raze r[;3])
 }
.qs.fx.walkYear:{[c;y]
    sp:.qs.fx.split[c;y]; i:sp 0; j:sp 1;
    d:c[`tf] .qs.data.laggedSplit[c`rows;i;j];
    s:.qs.fx.fit[d;c`spec];
    p:.qs.fx.forecast[s`model;d];
    ((enlist[`year]!enlist y),(enlist[`train]!enlist count i),`n`skill`skillMean`skillVsMean`ic`direction`upShare#s;p;d`yrawVal;d`tVal)
 }

/ one-feature rules scored like a model, with no hidden layer and nothing but a slope: forecast = b * the
/ feature (b fitted on the training rows by least squares through the origin) and the fixed rule that bets
/ on the feature's sign (no fitting). sign is 1 to follow the feature, -1 to bet against it
.qs.fx.walkRule:{[rows;dates;years;purge;name;sign]
    c:`rows`yr`t`purge`window!(rows;`year$dates rows`t;rows`t;purge;0N);
    r:.qs.fx.ruleYear[c;sign*.qs.fx.col[rows;name];rows`Y] each years;
    yrs:.qs.fx.ruleScore each r;
    `years`p`y`t`sign!(yrs;raze r[;1];raze r[;2];raze r[;4];raze r[;3])
 }
.qs.fx.ruleYear:{[c;s;y;yr]
    sp:.qs.fx.split[c;yr]; i:sp 0; j:sp 1;
    b:(sum s[i]*y i)%sum s[i]*s i;
    (yr;b*s j;y j;signum s j;c[`t] j)
 }
/ per year: the slope rule's skill and information coefficient, and the fixed rule's mean return against holding
.qs.fx.ruleScore:{[r]
    p:r 1; o:r 2; e:p-o;
    `year`n`skill`ic`direction`ruleReturn`holdReturn!(r 0;count o;1-(avg e*e)%avg o*o;cor[p;o];avg (signum p)=signum o;avg (r 3)*o;avg o)
 }

/ one model for several pairs: rowsBy is pair -> rows (as .qs.fx.rows, all on the same dates); each year trains
/ on every pair's earlier rows. Returns flat, aligned lists: `pair `p `y `t of the out-of-sample forecasts
.qs.fx.walkPooled:{[rowsBy;dates;years;purge;tf;spec]
    R:value rowsBy; ns:count each R[;`Y];
    rows:`X`Y`t!(raze R[;`X];raze R[;`Y];raze R[;`t]);
    pid:raze ns#'til count R;
    c:`rows`yr`t`purge`window`tf`spec!(rows;`year$dates rows`t;rows`t;purge;0N;tf;spec);
    r:.qs.fx.poolYear[c;pid] each years;
    `pair`p`y`t!((key rowsBy) raze r[;0];raze r[;1];raze r[;2];raze r[;3])
 }
.qs.fx.poolYear:{[c;pid;y]
    sp:.qs.fx.split[c;y]; i:sp 0; j:sp 1;
    d:c[`tf] .qs.data.laggedSplit[c`rows;i;j];
    s:.qs.fx.fit[d;c`spec];
    (pid j;.qs.fx.forecast[s`model;d];d`yrawVal;d`tVal)
 }

/ skill and information coefficient by calendar year, from forecasts p, outcomes y and their times t
.qs.fx.byYear:{[p;y;t;dates]
    yr:`year$dates t;
    {[p;y;yr;k] i:where yr=k; e:p[i]-y i; `year`n`skill`ic!(k;count i;1-(avg e*e)%avg y[i]*y i;cor[p i;y i])}[p;y;yr] each distinct yr
 }

/ the leak that purging prevents: train on all rows except every k-th and score on those. Neighbouring rows share
/ most of a long target, so a flexible model "forecasts" them from their neighbours; on a random walk
/ any skill here is fake
.qs.fx.interleaved:{[rows;every;spec]
    k:count rows`Y; j:where 0=(til k) mod every;
    .qs.fx.fit[.qs.data.laggedSplit[rows;(til k) except j;j];spec]
 }

/ pooled out-of-sample scores of forecasts p against outcomes y, with a circular block bootstrap (blocks of
/ block days, at least the horizon, because targets overlap) for the skill and information coefficient:
/ 95% intervals and the share of resamples with skill <= 0 (a one-sided p-value for "no skill")
.qs.fx.boot1:{[p;y;L;b]
    n:count y; i:n#((raze (ceiling[n%L]?n)+\:til L) mod n); e:p[i]-y i; z:y i;
    `skill`ic!(1-(avg e*e)%avg z*z;cor[p i;z])
 }
.qs.fx.summary:{[p;y;block;B]
    n:count y; e:p-y;
    bs:.qs.fx.boot1[p;y;block] each til B;
    sk:asc bs`skill; ic:asc bs`ic; q:{[v;f] v floor f*count v};
    `n`skill`skillLo`skillHi`pNoSkill`ic`icLo`icHi`direction`upShare!(n;1-(avg e*e)%avg y*y;q[sk;0.025];q[sk;0.975];
        avg sk<=0;cor[p;y];q[ic;0.025];q[ic;0.975];avg (signum p)=signum y;avg 0<y)
 }

/ the dashboard's and train.q's FX datasets: tomorrow's rate of a pair from its last 20 daily rates
/ (windows relative to the latest rate, the last 20% of days validating), when the ECB file is present
.qs.fx.file:{[] `$":",.qs.data.dir,"/fx/eurofxref-hist.csv"}
.qs.fx.dataset:{[pair;n;noise]
    if[()~key .qs.fx.file[]; '"no ECB data: run scripts/fetch_data.sh fx"];
    r:.qs.fx.rate[.qs.fx.ecb .qs.fx.file[];pair];
    .qs.data.windows[r where not null r;20;1;0.8]
 }
.qs.data.extra[`eurusd]:.qs.fx.dataset[`EURUSD];
.qs.data.extra[`gbpusd]:.qs.fx.dataset[`GBPUSD];
.qs.data.names:distinct .qs.data.names,`eurusd`gbpusd;
