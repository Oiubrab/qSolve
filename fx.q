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
/ (moves of exactly zero left out), against upShare, how often the market simply went up
.qs.fx.score:{[m;d]
    p:d[`mu]+d[`sd]*first each .qs.predict[m;d`Xval];
    y:d`yrawVal; tm:avg d`yraw;
    mse:{avg x*x};
    w:where y<>0;
    `n`mseModel`mseZero`mseMean`skill`skillMean`direction`upShare!(count y;mse p-y;mse y;mse y-tm;
        1-(mse p-y)%mse y;1-(mse y-tm)%mse y;avg (signum p w)=signum y w;avg 0<y w)
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
