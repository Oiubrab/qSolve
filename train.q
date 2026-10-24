/ train a model from the command line, e.g.
/   q train.q -data moons -hidden 16 -act tanh -epochs 200 -lr 0.1
/   q train.q -data mnist -hidden 128 -act relu -optimiser adam -lr 0.001 -epochs 5 -batch 64 -save model.qsm
/ options (defaults in brackets):
/   -data      and or xor nand fourBit sine circles moons spirals iris mnist fashion  [xor]
/   -rows      rows to generate for generated datasets [1000];  -noise  2-D noise sd [0.1]
/   -limit     use only the first N training rows (and N/6 validation rows) [all]
/   -hidden    hidden layer sizes, or none [16];  -act  hidden activation [tanh]
/   -out       output activation [softmax for several outputs, linear for sine, else sigmoid]
/   -loss      [xent for softmax, bce for sigmoid, else mse]
/   -epochs -batch -lr -optimiser -momentum -l2 -dropout -seed -verbose      as in .qs.defaults
/   -schedule none|step|cosine, -decay -stepEvery -minLr                     learning-rate schedule
/   -patience N   stop after N epochs without improvement;  -keepBest 1      return the best epoch's model
/   -load      start from a saved model instead of a new one;  -save  save the trained model here
/   -plot      write the decision boundary as an SVG file (2-input datasets only)

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
.cli.data:.qs.data.load[.cli.name;.cli.get[`rows;{"J"$first x};1000];.cli.get[`noise;{"F"$first x};0.1]];
if[`limit in key .cli.o; .cli.data:.qs.data.take[.cli.data;l;1|(l:"J"$first .cli.o`limit) div 6]];

.cli.nIn:count first .cli.data`X;
.cli.nOut:count first .cli.data`Y;
.cli.hidden:.cli.get[`hidden;{$[x~enlist "none";`long$();"J"$x]};enlist 16];
.cli.act:.cli.get[`act;{`$first x};`tanh];
.cli.out:.cli.get[`out;{`$first x};$[.cli.nOut>1;`softmax;.cli.name=`sine;`linear;`sigmoid]];
.cli.loss:.cli.get[`loss;{`$first x};$[.cli.out=`softmax;`xent;.cli.out=`sigmoid;`bce;`mse]];

.cli.opts:`loss`seed!(.cli.loss;.cli.seed);
.cli.num:`epochs`batch`lr`momentum`l2`dropout`decay`stepEvery`minLr`patience`keepBest`verbose!"JJFFFFFJFJBJ";
.cli.opts,:{x!{y$first .cli.o x}'[x;.cli.num x]} key[.cli.num] inter key .cli.o;
.cli.opts,:{x!`$first each .cli.o x} `optimiser`schedule inter key .cli.o;

.cli.model:$[`load in key .cli.o;
    .qs.load `$first .cli.o`load;
    .qs.init[.cli.nIn;.cli.hidden,.cli.nOut;((count .cli.hidden)#.cli.act),.cli.out;.cli.seed]];

-1 "data ",(string .cli.name),": ",(string count .cli.data`X)," training rows, ",(string count .cli.data`Xval)," validation rows";
-1 "model ",("-" sv string .cli.nIn,count each .cli.model`weight)," ",(" " sv string .cli.model`act),", loss ",string .cli.loss;
-1 "options ",-3!.cli.opts;

.cli.r:.qs.train[.cli.model;.cli.data;.cli.opts];
.cli.tr:.qs.evaluate[.cli.r`model;.cli.data`X;.cli.data`Y;.cli.loss];
.cli.va:.qs.evaluate[.cli.r`model;.cli.data`Xval;.cli.data`Yval;.cli.loss];
-1 "";
if[not null .cli.r`bestEpoch; -1 "best epoch ",(string .cli.r`bestEpoch),$[.cli.get[`keepBest;{"B"$first x};0b];" (model returned)";""]];
-1 "final model: train acc ",(.Q.fmt[6;4;.cli.tr`acc]),", val acc ",(.Q.fmt[6;4;.cli.va`acc]),", ",(trim .Q.fmt[12;2;sum .cli.r[`history]`seconds]),"s total";
if[(.cli.nOut>1) and .cli.nOut<=10; -1 "validation confusion matrix (rows true, columns predicted):"; show .qs.confusion[.cli.r`model;.cli.data`Xval;.cli.data`Yval]];
if[`plot in key .cli.o;
    .qs.plot2D[.cli.r`model;.cli.data;`$first .cli.o`plot;(string .cli.name),"  ",("-" sv string .cli.nIn,count each .cli.model`weight),"  val acc ",.Q.fmt[6;4;.cli.va`acc]];
    -1 "plotted ",first .cli.o`plot];
if[`save in key .cli.o; .qs.save[`$first .cli.o`save;.cli.r`model]; -1 "saved ",first .cli.o`save];
exit 0
