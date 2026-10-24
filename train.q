/ train a model from the command line, e.g.
/   q train.q -data moons -hidden 16 -act tanh -epochs 200 -lr 0.1
/   q train.q -data mnist -hidden 128 -act relu -optimiser adam -lr 0.001 -epochs 5 -batch 64 -save model.qsm
/   q train.q -data csv -file sales.csv -target price -features area rooms city -save price.qsm
/   q train.q -data csv -file load.csv -series demand -lags 48 -horizon 1 -hidden none
/ options (defaults in brackets):
/   -data      and or xor nand fourBit sine circles moons spirals houses seasonal iris mnist fashion, or csv [xor]
/   -file      csv: the file (with a header row);  -target  csv: the column to predict
/   -features  csv: input columns [every other column];  -frac  csv: training fraction [0.8]
/   -series    csv: forecast this column from its own past instead (rows in time order), with
/   -lags      values per window [24] and -horizon steps ahead [1]; validation is the last 1-frac of time
/   -rows      rows to generate for generated datasets [1000];  -noise  2-D noise sd [0.1]
/   -limit     use only the first N training rows (and N/6 validation rows) [all]
/   -hidden    hidden layer sizes, or none [16];  -act  hidden activation [tanh]
/   -out       output activation [softmax for several outputs, sigmoid for one 0/1 output, else linear]
/   -loss      [xent for softmax, bce for sigmoid, else mse]
/   -epochs -batch -lr -optimiser -momentum -l2 -dropout -seed -verbose      as in .qs.defaults
/   -schedule none|step|cosine, -decay -stepEvery -minLr                     learning-rate schedule
/   -patience N   stop after N epochs without improvement;  -keepBest 1      return the best epoch's model
/   -load      start from a saved model instead of a new one
/   -save      save the trained model here (with the column encoding, for table and csv data)
/   -plot      write the decision boundary as an SVG file (2-input datasets only)
/   -search    try configurations first and train the best, e.g. -search lr=0.01,0.001 hidden=8,32x32 act=tanh,relu
/              (any option above; layers in a hidden value are joined by x);  -trials N  sample N of them [all]
/   -cv K      cross-validate the configuration over K folds before the final training run

.cli.root:{$[count x;x;"."]} "/" sv -1_ "/" vs string .z.f;
system "l ",.cli.root,"/defineEquation.q";
system "l ",.cli.root,"/data.q";
system "l ",.cli.root,"/plot.q";
.qs.data.dir:.cli.root,"/data";

.cli.o:.Q.opt .z.x;
system "c 50 1000";
/ option k parsed with f, or default d
.cli.get:{[k;f;d] $[k in key .cli.o; f .cli.o k; d]};

.cli.name:.cli.get[`data;{`$first x};`xor];
.cli.seed:.cli.get[`seed;{"J"$first x};42];
system "S ",string .cli.seed;
.cli.data:$[.cli.name=`csv;
    [if[not `file in key .cli.o; '"csv needs -file"];
        .cli.frac:.cli.get[`frac;{"F"$first x};0.8];
        $[`series in key .cli.o;
            .qs.data.windows[(.qs.data.readCsv `$first .cli.o`file) `$first .cli.o`series;.cli.get[`lags;{"J"$first x};24];.cli.get[`horizon;{"J"$first x};1];.cli.frac];
          `target in key .cli.o;
            .qs.data.csv[`$first .cli.o`file;.cli.get[`features;`$;()];`$first .cli.o`target;.cli.frac];
          '"csv needs -target or -series"]];
    .qs.data.load[.cli.name;.cli.get[`rows;{"J"$first x};1000];.cli.get[`noise;{"F"$first x};0.1]]];
if[`limit in key .cli.o; .cli.data:.qs.data.take[.cli.data;l;1|(l:"J"$first .cli.o`limit) div 6]];

.cli.nIn:count first .cli.data`X;
.cli.nOut:count first .cli.data`Y;
.cli.hidden:.cli.get[`hidden;{$[x~enlist "none";`long$();"J"$x]};enlist 16];
.cli.act:.cli.get[`act;{`$first x};`tanh];
.cli.class:.qs.isClass .cli.data`Y;
.cli.out:.cli.get[`out;{`$first x};$[not .cli.class;`linear;.cli.nOut>1;`softmax;`sigmoid]];
.cli.loss:.cli.get[`loss;{`$first x};$[.cli.out=`softmax;`xent;.cli.out=`sigmoid;`bce;`mse]];

.cli.opts:`loss`seed!(.cli.loss;.cli.seed);
.cli.num:`epochs`batch`lr`momentum`l2`dropout`decay`stepEvery`minLr`patience`keepBest`verbose!"JJFFFFFJFJBJ";
.cli.opts,:{x!{y$first .cli.o x}'[x;.cli.num x]} key[.cli.num] inter key .cli.o;
.cli.opts,:{x!`$first each .cli.o x} `optimiser`schedule inter key .cli.o;

/ parse "key=v1,v2" search terms, typed like the options they set
.cli.term:{[t]
    kv:"=" vs t; k:`$kv 0; v:"," vs kv 1;
    (k;$[k=`hidden; {$[x~"none";`long$();"J"$"x" vs x]} each v;
        k in key .cli.num; .cli.num[k]$v;
        `$v])
 };
if[`search in key .cli.o;
    .cli.space:(!) . flip .cli.term each .cli.o`search;
    .cli.base:.cli.opts,`hidden`act`out!(.cli.hidden;.cli.act;.cli.out);
    -1 "searching ",-3!.cli.space;
    .cli.found:.qs.search[.cli.data;.cli.space;.cli.base;.cli.get[`trials;{"J"$first x};0N]];
    show .cli.found;
    .cli.best:(key .cli.space)#first .cli.found;
    -1 "best: ",-3!.cli.best;
    if[`hidden in key .cli.best; .cli.hidden:.cli.best`hidden];
    if[`act in key .cli.best; .cli.act:.cli.best`act];
    if[`out in key .cli.best; .cli.out:.cli.best`out];
    .cli.opts,:`hidden`act`out _ .cli.best;
    .cli.loss:.cli.opts`loss;
    -1 ""];

.cli.model:$[`load in key .cli.o;
    .qs.load `$first .cli.o`load;
    .qs.init[.cli.nIn;.cli.hidden,.cli.nOut;((count .cli.hidden)#.cli.act),.cli.out;.cli.seed]];

-1 "data ",(string .cli.name),": ",(string count .cli.data`X)," training rows, ",(string count .cli.data`Xval)," validation rows";
-1 "model ",("-" sv string .cli.nIn,count each .cli.model`weight)," ",(" " sv string .cli.model`act),", loss ",(string .cli.loss),", ",(string sum .qs.summary[.cli.model]`params)," parameters";
-1 "options ",-3!.cli.opts;

if[`cv in key .cli.o;
    .cli.k:"J"$first .cli.o`cv;
    -1 (string .cli.k),"-fold cross-validation:";
    .cli.cv:.qs.crossValidate[.cli.data;.cli.k;.cli.opts,`hidden`act`out!(.cli.hidden;.cli.act;.cli.out)];
    show .cli.cv`folds; show .cli.cv`summary; -1 ""];

.cli.r:.qs.train[.cli.model;.cli.data;.cli.opts];
.cli.tr:.qs.evaluate[.cli.r`model;.cli.data`X;.cli.data`Y;.cli.loss];
.cli.va:.qs.evaluate[.cli.r`model;.cli.data`Xval;.cli.data`Yval;.cli.loss];
-1 "";
if[not null .cli.r`bestEpoch; -1 "best epoch ",(string .cli.r`bestEpoch),$[.cli.get[`keepBest;{"B"$first x};0b];" (model returned)";""]];
/ regression errors are reported in the target's own units when there is an encoding
.cli.sd:$[`enc in key .cli.data; (.cli.data[`enc] first exec col from .cli.data[`enc] where role=`target)`sd;
    `level in key .cli.data; .cli.data`sd; 1f];
.cli.score:{$[.cli.class; "acc ",.Q.fmt[6;4;x`acc]; "R2 ",(trim .Q.fmt[9;4;x`r2])," MAE ",trim .Q.fmt[20;4;.cli.sd*x`mae]]};
-1 "final model: train ",(.cli.score .cli.tr),", val ",(.cli.score .cli.va),", ",(trim .Q.fmt[12;2;sum .cli.r[`history]`seconds]),"s total";
/ for time series, the naive forecast (no change from the last value) to beat
if[`level in key .cli.data; -1 "naive last-value forecast: val MAE ",trim .Q.fmt[20;4;avg abs .cli.data[`mu]+.cli.sd*first each .cli.data`Yval]];
if[.cli.class and (.cli.nOut>1) and .cli.nOut<=10; -1 "validation confusion matrix (rows true, columns predicted):"; show .qs.confusion[.cli.r`model;.cli.data`Xval;.cli.data`Yval]];
if[`plot in key .cli.o;
    .qs.plot2D[.cli.r`model;.cli.data;`$first .cli.o`plot;(string .cli.name),"  ",("-" sv string .cli.nIn,count each .cli.model`weight),"  val acc ",.Q.fmt[6;4;.cli.va`acc]];
    -1 "plotted ",first .cli.o`plot];
if[`save in key .cli.o;
    .cli.m:.cli.r`model; if[`enc in key .cli.data; .cli.m[`enc]:.cli.data`enc];
    .qs.save[`$first .cli.o`save;.cli.m]; -1 "saved ",first .cli.o`save];
exit 0
