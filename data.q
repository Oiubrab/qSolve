/ qSolve datasets: generators and loaders, each returning `X`Y (and `Xval`Yval where there is a split)
/ X is a matrix with one row per example, Y one-hot (or a single column for binary and regression targets)
/ generators use q's random state: set it with system "S <seed>" for reproducible data

.qs.data.pi:acos -1

/ n draws from the standard normal distribution (Box-Muller)
.qs.data.normal:{[n] sqrt[-2*log 1-n?1f]*cos 2*.qs.data.pi*n?1f}

/ one-hot rows for integer labels in 0..k-1
.qs.data.oneHot:{[k;labels] "f"$labels=\:til k}

/ the four rows of a two-input logic gate f, e.g. gate[`xor]
.qs.data.gates:`and`or`xor`nand!(&;|;<>;{not x&y})
.qs.data.gate:{[f]
    X:(0 0f;0 1f;1 0f;1 1f);
    `X`Y!(X;enlist each "f"$.qs.data.gates[f] .' "b"$X)
 }

/ the original qSolve task: 4 random bits, a fifth of rows forced to 1100 or 0011
/ classes are 1100, 0011, and neither (most rows), one-hot over three columns
.qs.data.fourBit:{[n]
    X:"f"$(n;4)#(4*n)?0 1;
    i:5*til ceiling n%5;
    X[i]:(count i)?(1 1 0 0f;0 0 1 1f);
    c:{$[x~1 1 0 0f;0;x~0 0 1 1f;1;2]} each X;
    `X`Y!(X;.qs.data.oneHot[3;c])
 }

/ regression: y = sin x for x in [0,2pi], with X scaled to [-1,1] and Y to [0,1]
.qs.data.sine:{[n]
    x:n?2*.qs.data.pi;
    `X`Y!(enlist each -1+x%.qs.data.pi;enlist each 0.5+0.5*sin x)
 }

/ binary 2-D datasets: n points, half per class, gaussian noise of sd noise added to each coordinate
.qs.data.noisy2D:{[pts;labels;noise] `X`Y!(pts+noise*(count pts;2)#.qs.data.normal 2*count pts;.qs.data.oneHot[2;labels])}

/ two concentric circles of radius 1 (class 0) and 0.5 (class 1)
.qs.data.circles:{[n;noise]
    k:n div 2; t:n?2*.qs.data.pi; r:?[n#(k#1b),(n-k)#0b;1f;0.5];
    .qs.data.noisy2D[flip (r*cos t;r*sin t);"j"$r=0.5;noise]
 }

/ two interleaving half circles
.qs.data.moons:{[n;noise]
    k:n div 2; t:n?.qs.data.pi; c:n#(k#0),(n-k)#1;
    x:?[c=0;cos t;1-cos t]; y:?[c=0;sin t;0.5-sin t];
    .qs.data.noisy2D[flip (x;y);c;noise]
 }

/ two interleaved spirals, 1.5 turns each, scaled to about [-1,1]
.qs.data.spirals:{[n;noise]
    k:n div 2; t:3*.qs.data.pi*sqrt n?1f; c:n#(k#0),(n-k)#1;
    s:?[c=0;1f;-1f]*t%3*.qs.data.pi;
    .qs.data.noisy2D[flip (s*cos t;s*sin t);c;noise]
 }

/ shuffle and split data into training and validation parts, frac of rows going to training
.qs.data.split:{[d;frac]
    i:neg[n]?n:count d`X; k:floor frac*n;
    `X`Y`Xval`Yval!(d[`X] k#i;d[`Y] k#i;d[`X] k _i;d[`Y] k _i)
 }

/ column means and standard deviations, and scaling by them (sd 0 columns are left unscaled)
.qs.data.stats:{[X] sd:dev X; `mu`sd!(avg X;?[sd=0;1f;sd])}
.qs.data.scale:{[st;X] (X-\:st`mu)%\:st`sd}

/ standardise X and Xval using the training columns' statistics
.qs.data.standardise:{[d]
    st:.qs.data.stats d`X;
    d[`X]:.qs.data.scale[st;d`X];
    if[`Xval in key d; d[`Xval]:.qs.data.scale[st;d`Xval]];
    d
 }

/ read an IDX file of unsigned bytes (the MNIST format): a vector for 1-D files, otherwise one row per item
.qs.data.idx:{[file]
    b:read1 hsym file;
    if[not (0x0000~2#b) and 0x08=b 2; '"not an unsigned-byte IDX file: ",string file];
    dims:{"j"$0x0 sv x} each 4 cut (4*"j"$b 3)#4_b;
    v:(4+4*count dims)_b;
    if[not (prd dims)=count v; '"IDX file is truncated: ",string file];
    $[1=count dims; "j"$v; (first dims;prd 1_dims)#v]
 }

/ MNIST-format directory (MNIST or Fashion-MNIST): pixels scaled to [0,1], labels one-hot, test set as validation
.qs.data.mnist:{[dir]
    f:{[dir;x] .qs.data.idx `$(string dir),"/",x}[dir];
    `X`Y`Xval`Yval!(
        (1%255)*"f"$f "train-images-idx3-ubyte";
        .qs.data.oneHot[10;f "train-labels-idx1-ubyte"];
        (1%255)*"f"$f "t10k-images-idx3-ubyte";
        .qs.data.oneHot[10;f "t10k-labels-idx1-ubyte"])
 }

/ the first n training and nval validation rows of a dataset
.qs.data.take:{[d;n;nval] d:@[d;`X`Y;n sublist]; $[`Xval in key d; @[d;`Xval`Yval;nval sublist]; d]}

/ tables: encode the columns of any kdb+ table as model inputs and targets
/ an encoding is a keyed table, one row per column: kind is `num (standardised, nulls become the mean)
/ or `cat (one-hot over levels seen in training; nulls and unseen levels encode as all zeros)

/ symbols for string columns, otherwise the column as it is
.qs.data.asSym:{[x] $[0h=type x; `$x; 10h=type x; `$enlist each x; x]}

/ encoding of one column; lo and hi are the training range of a numeric column
.qs.data.fitCol:{[x]
    x:.qs.data.asSym x;
    $[11h=type x; `kind`mu`sd`levels`lo`hi!(`cat;0n;1f;asc distinct x where not null x;0n;0n);
        [f:"f"$x; sd:dev f; `kind`mu`sd`levels`lo`hi!(`num;avg f;$[(null sd) or sd=0;1f;sd];`symbol$();min f;max f)]]
 }

/ encode one column with its encoding, one row per value
.qs.data.encodeCol:{[e;x]
    x:.qs.data.asSym x;
    $[`cat=e`kind; "f"$x=\:e`levels; enlist each 0f^(("f"$x)-e`mu)%e`sd]
 }

/ encoding of the features and target columns of table t; empty features means every other column
.qs.data.fit:{[t;features;target]
    features:$[count features; (),features; cols[t] except target];
    if[not all (features,target) in cols t; '"no such column: ",-3!(features,target) except cols t];
    enc:([] col:features,target; role:((count features)#`feature),`target) ,' .qs.data.fitCol each t features,target;
    `col xkey enc
 }

/ inputs and targets of table t under encoding enc
.qs.data.encodeX:{[enc;t] f:exec col from enc where role=`feature; (,'/) {[enc;t;c] .qs.data.encodeCol[enc c;t c]}[enc;t] each f}
.qs.data.encodeY:{[enc;t] c:first exec col from enc where role=`target; .qs.data.encodeCol[enc c;t c]}

/ model outputs back to target values: class symbols, or numbers in the target's units
.qs.data.decodeY:{[enc;A]
    e:enc first exec col from enc where role=`target;
    $[`cat=e`kind; e[`levels] .qs.classes A; e[`mu]+e[`sd]*first each A]
 }

/ names of the encoded input columns, e.g. city_paris for a categorical column
.qs.data.inputNames:{[enc]
    e:select from 0!enc where role=`feature;
    raze {[c;e] $[`cat=e`kind; `$(string c),/:"_",/:string e`levels; enlist c]}'[e`col;e]
 }

/ shuffle table t, split it frac/1-frac, fit the encoding on the training rows only, and encode both parts
/ returns `X`Y`Xval`Yval`Tval`enc (no Xval, Yval and Tval when frac is 1), Tval being the validation rows as
/ they were (the used columns only); generated rows are reproducible after system "S n"
.qs.data.fromTable:{[t;features;target;frac]
    t:t neg[n]?n:count t; k:floor frac*n;
    enc:.qs.data.fit[k#t;features;target];
    d:`X`Y!(.qs.data.encodeX[enc;k#t];.qs.data.encodeY[enc;k#t]);
    if[k<n; d,:`Xval`Yval`Tval!(.qs.data.encodeX[enc;k _t];.qs.data.encodeY[enc;k _t];(exec col from enc)#k _t)];
    d,enlist[`enc]!enlist enc
 }

/ read a CSV with a header row; column names are made q-safe, and a column becomes float if every
/ non-empty value parses as a number, otherwise symbol (empty fields are nulls)
.qs.data.readCsv:{[file] .qs.data.parseCsv read0 hsym file}

/ the same from CSV text already split into lines (blank lines are ignored)
.qs.data.parseCsv:{[lines]
    lines:lines where 0<count each lines;
    if[2>count lines; '"CSV needs a header row and at least one data row"];
    t:.Q.id (count["," vs first lines]#"*";enlist ",") 0: lines;
    {[x] v:x where 0<count each x; $[(count v) and not any null "F"$v; "F"$x; `$x]} each flip t
 }

/ train/validation data from a CSV file: target column name, features (empty for all others), training fraction
.qs.data.csv:{[file;features;target;frac] .qs.data.fromTable[flip .qs.data.readCsv file;features;target;frac]}

/ Iris CSV (UCI iris.data, no header): 4 standardised features, 3 one-hot classes, shuffled 80/20 split
.qs.data.iris:{[file]
    t:flip `sl`sw`pl`pw`species!("FFFFS";",") 0: hsym file;
    .qs.data.fromTable[select from t where not null species;`sl`sw`pl`pw;`species;0.8]
 }

/ a synthetic house-price table for regression, n rows: numeric, categorical and missing values
/ price depends nonlinearly on area, rooms, age and city, plus noise
.qs.data.houses:{[n]
    area:40+n?160f; rooms:1+n?6; age:n?80f; city:n?`north`south`east`west;
    age[(n div 20)?n]:0n;
    base:(`north`south`east`west!300 220 260 180) city;
    price:base+(2.5*area)+(15*rooms)-(0.02*a*a:0f^age)+(0.004*area*area)+10*.qs.data.normal n;
    ([] area;rooms;age;city;price)
 }

/ time series: lagged windows of a numeric series, split by time rather than shuffled

/ times t whose window (lags values up to t) and target (horizon steps after t) fit in n points, and
/ how many of them train: the first frac by time
.qs.data.windowTimes:{[n;lags;horizon;frac]
    if[n<lags+horizon+1; '"series too short for ",(string lags)," lags and horizon ",string horizon];
    t:(lags-1)+til n-lags+horizon-1;
    (t;floor frac*count t)
 }

/ training rows from series s: inputs are the lags-1 earlier values of each window and the target is the
/ value horizon steps later, all relative to the window's last value, so a trend or level shift doesn't
/ carry validation outside the range seen in training; inputs are standardised per column and the target by
/ its own mean and sd, all from training rows only. Returns `X`Y`Xval`Yval`mu`sd`level`levelVal, levels being
/ each row's last value, so .qs.data.unwindow can put predictions back in the series' units, plus what
/ .qs.data.windowRow needs to encode new values (xstats, lags, horizon) and the raw series and window times
.qs.data.windows:{[s;lags;horizon;frac]
    s:"f"$s; tk:.qs.data.windowTimes[count s;lags;horizon;frac]; t:tk 0; k:tk 1;
    L:s t; X:(s t+\:neg reverse 1_til lags)-L; y:s[t+horizon]-L;
    st:.qs.data.stats k#X;
    d:`X`Y`Xval`Yval!(.qs.data.scale[st;k#X];();.qs.data.scale[st;k _X];());
    mu:avg k#y; sd:dev k#y; sd:$[sd=0;1f;sd];
    d[`Y]:enlist each (k#y-mu)%sd; d[`Yval]:enlist each (k _y-mu)%sd;
    d,`mu`sd`level`levelVal`xstats`lags`horizon`series`times`k!(mu;sd;k#L;k _L;st;lags;horizon;s;t;k)
 }

/ the model input for the most recent values of a series (oldest first, at least lags of them), as for
/ .qs.data.windows d: `X (one row) and `level, so .qs.data.unwindow[d;predict X;level] is the forecast
.qs.data.windowRow:{[d;recent]
    if[d[`lags]>count recent; '"need the last ",(string d`lags)," values"];
    r:"f"$(neg d`lags)#recent; L:last r;
    `X`level!(.qs.data.scale[d`xstats;enlist (-1_r)-L];enlist L)
 }

/ windowed predictions A back in the series' units, given the rows' levels (d`level or d`levelVal)
.qs.data.unwindow:{[d;A;levels] levels+d[`mu]+d[`sd]*first each A}

/ mean absolute error, in the series' units, of naive forecasts for the validation targets of .qs.data.windows:
/ the last value seen, and the value one period before the target (seasonal naive, null if not in the window)
.qs.data.naive:{[s;lags;horizon;frac;period]
    s:"f"$s; tk:.qs.data.windowTimes[count s;lags;horizon;frac]; t:tk[1]_tk 0;
    y:s t+horizon;
    `last`seasonal!(avg abs y-s t;$[(period>=horizon) and period<=lags-1+horizon; avg abs y-s t+horizon-period; 0n])
 }

/ windows over several aligned series (inputs: name -> series, e.g. daily log returns of several pairs):
/ row t holds the last lags values of every input up to and including t (input by input, oldest first),
/ and the target is the sum of target's next horizon values (for log returns, the horizon-step return).
/ Rows with a null anywhere are dropped. Returns `X`Y (unscaled) and `t, the time of each row
.qs.data.laggedRows:{[inputs;target;lags;horizon]
    S:"f"$value inputs; y:"f"$inputs target; n:count y;
    if[n<lags+horizon+1; '"series too short for ",(string lags)," lags and horizon ",string horizon];
    t:(lags-1)+til n-lags+horizon-1;
    X:raze each flip {[t;lags;s] s t+\:neg reverse til lags}[t;lags] each S;
    / (sum skips nulls, so a target spanning a missing value is null rather than a smaller sum)
    Y:{$[any null x;0n;sum x]} each y t+\:1+til horizon;
    ok:where not (any each null X) or null Y;
    `X`Y`t!(X ok;Y ok;t ok)
 }

/ model data from laggedRows output: rows i train and rows j validate, scaled with the training rows'
/ statistics (inputs per column, the target by its mean and sd); yraw and t keep the unscaled targets and times
.qs.data.laggedSplit:{[rows;i;j]
    st:.qs.data.stats rows[`X] i; y:rows`Y;
    mu:avg y i; sd:dev y i; sd:$[sd=0;1f;sd];
    `X`Y`Xval`Yval`mu`sd`xstats`yraw`yrawVal`t`tVal!(.qs.data.scale[st;rows[`X] i];enlist each (y[i]-mu)%sd;
        .qs.data.scale[st;rows[`X] j];enlist each (y[j]-mu)%sd;mu;sd;st;y i;y j;rows[`t] i;rows[`t] j)
 }

/ the same split by time: the first frac of rows train, the rest validate
.qs.data.lagged:{[inputs;target;lags;horizon;frac]
    r:.qs.data.laggedRows[inputs;target;lags;horizon]; k:floor frac*count r`Y;
    .qs.data.laggedSplit[r;til k;k+til count[r`Y]-k],`lags`horizon`names!(lags;horizon;key inputs)
 }

/ a generated seasonal series of n points: slow trend, a daily-style cycle of length period with a
/ harmonic, and autocorrelated noise of sd about noise
.qs.data.seasonal:{[n;period;noise]
    t:"f"$til n; w:2*.qs.data.pi*t%period;
    e:{[a;x;y] (a*x)+y}[0.6]\[noise*0.8*.qs.data.normal n];
    (0.001*t)+(sin w)+(0.4*sin 2*w)+e
 }

/ directory holding downloaded data (see scripts/fetch_data.sh)
.qs.data.dir:"data"

/ datasets by name, split into training and validation: n rows and noise are used by the generated ones
/ the logic gates validate on their own four rows
.qs.data.names:`and`or`xor`nand`fourBit`sine`circles`moons`spirals`houses`seasonal`iris`mnist`fashion
/ datasets other modules add by name: name -> function of (n;noise), e.g. fx.q's currency pairs
.qs.data.extra:(`symbol$())!();
.qs.data.load:{[name;n;noise]
    $[name in key .qs.data.gates; {x,`Xval`Yval!x`X`Y} .qs.data.gate name;
        name in `fourBit`sine; .qs.data.split[.qs.data[name] n;0.8];
        name in `circles`moons`spirals; .qs.data.split[.qs.data[name][n;noise];0.8];
        name=`iris; .qs.data.iris `$":",.qs.data.dir,"/iris/iris.data";
        name=`houses; .qs.data.fromTable[.qs.data.houses n;();`price;0.8];
        name=`seasonal; .qs.data.windows[.qs.data.seasonal[n;24;noise];48;1;0.8];
        name in `mnist`fashion; .qs.data.mnist `$":",.qs.data.dir,"/",string name;
        name in key .qs.data.extra; .qs.data.extra[name][n;noise];
        '"unknown dataset: ",(string name),", expected one of ",", " sv string .qs.data.names]
 }
