/ qSolve dashboard: a web UI for training, comparing, searching and using models
/   q dashboard.q [-port 8080] [-host 127.0.0.1] [-s 4]      then open http://127.0.0.1:8080
/ serves ui/ over HTTP and a JSON API over a WebSocket on the same port; the same port answers q IPC
/ queries (e.g. from KX Dashboards) about .dash.runs and .dash.history[]
/ API: send {"cmd":name,...}; replies are JSON objects with a "type" (and the request's "id", if it had one)

/ when loaded from another script (the tests), .dash.root can be set first; otherwise it is this file's directory
.dash.main:(string .z.f) like "*dashboard.q";
if[not `root in key `.dash; .dash.root:{$[count x;x;"."]} "/" sv -1_ "/" vs string .z.f];
system "l ",.dash.root,"/defineEquation.q";
system "l ",.dash.root,"/data.q";
system "l ",.dash.root,"/plot.q";
.qs.data.dir:.dash.root,"/data";
.dash.modelDir:.dash.root,"/models";

/ session state: one row per run, with its model, history and dataset details alongside
.dash.runs:([id:`long$()] time:`timestamp$(); data:`symbol$(); arch:(); task:`symbol$(); loss:`symbol$();
    epochs:`long$(); valLoss:`float$(); valAcc:`float$(); valR2:`float$(); seconds:`float$(); source:`symbol$());
.dash.model:(`long$())!();
.dash.hist:(`long$())!();
.dash.meta:(`long$())!();
.dash.uploads:(`symbol$())!();

/ every run's history in one table, for KX Dashboards and other q clients
.dash.history:{[] $[count .dash.hist; raze {[i;h] ([] run:(count h)#i),'h}'[key .dash.hist;value .dash.hist]; ()]}

/ JSON can't carry infinities: replace them with nulls, recursively
.dash.clean:{[x]
    t:type x;
    $[t=-9h; $[0w=abs x;0n;x]; t=9h; ?[0w=abs x;0n;x]; t=0h; .z.s each x;
        t=99h; key[x]!.z.s value x; t=98h; flip .z.s flip x; x]
 }

/ send a message to the client that made the current request; tests replace this
.dash.send:{[msg] neg[.z.w] .j.j .dash.clean .dash.reqId,msg; neg[.z.w][]}
.dash.reqId:()!();
.dash.debug:0b;

/ request field k, or d when it is missing, null or empty
/ true for JSON null, empty strings and lists, and null atoms
.dash.empty:{[v] $[(::)~v; 1b; 0<=type v; 0=count v; null v]}
.dash.f:{[r;k;d] $[not k in key r; d; .dash.empty v:r k; d; v]}
.dash.sym:{[r;k;d] $[10h=type v:.dash.f[r;k;d]; `$v; -11h=type v; v; `$string v]}
.dash.num:{[r;k;d] "f"$.dash.f[r;k;d]}
.dash.int:{[r;k;d] "j"$.dash.f[r;k;d]}
/ a JSON list of names as symbols; .j.k turns ["a","b"] (one-letter names) into the string "ab"
.dash.syms:{[x] $[10h=type x; `$enlist each x; -10h=type x; enlist `$enlist x; `$x]}

/ datasets the UI can offer: built-in ones (with whether their files are present) and uploads
.dash.files:`iris`mnist`fashion!("iris/iris.data";"mnist/train-images-idx3-ubyte";"fashion/train-images-idx3-ubyte");
.dash.kinds:`and`or`xor`nand`fourBit`sine`circles`moons`spirals`houses`seasonal`iris`mnist`fashion!
    `logic`logic`logic`logic`classes`regression`2d`2d`2d`table`series`table`images`images;
.dash.about:`and`or`xor`nand`fourBit`sine`circles`moons`spirals`houses`seasonal`iris`mnist`fashion!(
    "AND gate, 4 rows";"OR gate, 4 rows";"XOR gate, 4 rows: needs a hidden layer";"NAND gate, 4 rows";
    "4 random bits: 1100, 0011 or neither";"y = sin x, regression";"two concentric circles";
    "two interleaving half circles";"two spirals, the hard one";"house prices: numbers, categories, gaps";
    "a seasonal series with trend, 48 lags";"150 flowers, 3 species";"70,000 handwritten digits";
    "70,000 clothing images, 10 kinds");
.dash.datasets:{[]
    b:{[n] `name`kind`about`available`generated!(n;.dash.kinds n;.dash.about n;
        $[n in key .dash.files; not ()~key hsym `$.qs.data.dir,"/",.dash.files n; 1b];
        n in `fourBit`sine`circles`moons`spirals`houses`seasonal)} each .qs.data.names;
    u:{[n] t:.dash.uploads n; `name`kind`about`available`generated!(n;`upload;(string count t)," rows, ",(string count cols t)," columns";1b;0b)} each key .dash.uploads;
    b,u
 }

/ saved models in models/
.dash.saved:{[]
    f:$[()~key hsym `$.dash.modelDir; `symbol$(); f where (f:key hsym `$.dash.modelDir) like "*.qsm"];
    {[f] m:.qs.load `$":",.dash.modelDir,"/",string f;
        `name`arch`params`target!(`$-4_string f;"-" sv string (count first first m`weight),count each m`weight;
            sum .qs.summary[m]`params;$[`enc in key m; first exec col from m[`enc] where role=`target; `])} each f
 }

.dash.runList:{[] 0!.dash.runs}

/ the input columns of a table encoding, for building a prediction form: name, kind, levels (categories), mean
.dash.features:{[enc] $[count enc; select col,kind,levels,mu from 0!enc where role=`feature; ()]}

/ the data a request asks for: a built-in dataset, or an upload used as a table or as a time series
.dash.loadData:{[r]
    name:.dash.sym[r;`data;`xor];
    frac:.dash.num[r;`frac;0.8];
    d:$[name in key .dash.uploads;
        [t:.dash.uploads name;
            $[not null s:.dash.sym[r;`series;`];
                .qs.data.windows[t s;.dash.int[r;`lags;24];.dash.int[r;`horizon;1];frac];
                .qs.data.fromTable[t;.dash.syms .dash.f[r;`features;()];.dash.sym[r;`target;`];frac]]];
        name in .qs.data.names; .qs.data.load[name;.dash.int[r;`rows;1000];.dash.num[r;`noise;0.1]];
        '"unknown dataset: ",string name];
    if[not null l:.dash.int[r;`limit;0N]; d:.qs.data.take[d;l;1|l div 6]];
    d
 }

/ model shape and training options from a request; "auto" output activation and loss follow the targets
.dash.config:{[r;d]
    o:`epochs`batch`lr`optimiser`momentum`l2`dropout`schedule`patience`keepBest`seed!(
        .dash.int[r;`epochs;50];.dash.int[r;`batch;32];.dash.num[r;`lr;0.01];.dash.sym[r;`optimiser;`adam];
        .dash.num[r;`momentum;0.9];.dash.num[r;`l2;0f];.dash.num[r;`dropout;0f];.dash.sym[r;`schedule;`none];
        .dash.int[r;`patience;0N];"b"$.dash.f[r;`keepBest;0b];.dash.int[r;`seed;42]);
    c:`hidden`act!("j"$.dash.f[r;`hidden;`long$()];.dash.sym[r;`act;`tanh]);
    if[not `auto=out:.dash.sym[r;`out;`auto]; c[`out]:out];
    if[not `auto=loss:.dash.sym[r;`loss;`auto]; c[`loss]:loss];
    .qs.spec[d;c,o]
 }

/ description of a configuration for the runs table, e.g. 2-16-2 tanh/softmax
.dash.arch:{[nIn;nOut;c] ("-" sv string nIn,c[`hidden],nOut),$[count c`hidden; " ",string c`act; ""]," ",string c`out}

/ everything the UI shows about a finished run
.dash.report:{[id]
    m:.dash.model id; h:.dash.hist id; mt:.dash.meta id; d:mt`data;
    cls:(mt`task) in `classes`binary;
    va:$[`Xval in key d; .qs.evaluate[m;d`Xval;d`Yval;mt`loss]; ()!()];
    tr:.qs.evaluate[m;d`X;d`Y;mt`loss];
    k:count first d`Y;
    conf:$[cls and (`Xval in key d) and k<=12; .qs.confusion[m;d`Xval;d`Yval]; ()];
    labels:$[not cls; (); `enc in key d; string (d[`enc] first exec col from d[`enc] where role=`target)`levels; string til max 2,k];
    / decision boundary for 2-D classes: the prediction grid and the points, drawn by the browser
    plot:$[(2=count first d`X) and cls;
        .qs.plot.grid[m;d;60],`train`val!({[X;Y] X,'.qs.classes Y}[d`X;d`Y];$[`Xval in key d;{[X;Y] X,'.qs.classes Y}[d`Xval;d`Yval];()]);
        ()!()];
    / regression errors in the target's units when the data carries a scale
    sd:$[`enc in key d; (d[`enc] first exec col from d[`enc] where role=`target)`sd; `sd in key d; d`sd; 1f];
    naive:$[`level in key d; avg abs d[`mu]+sd*first each d`Yval; 0n];
    / regression: validation predictions against the truth, in the target's units (the last 200 steps of a
    / series, in order; up to 400 rows of a table)
    fit:$[(mt[`task]=`regression) and `Xval in key d;
        [ser:`level in key d; n:count d`Yval;
            i:$[ser; (neg 200&n)#til n; (400&n)#til n];
            dec:{[d;ser;i;A] $[ser; .qs.data.unwindow[d;A;d[`levelVal] i]; `enc in key d; .qs.data.decodeY[d`enc;A]; first each A]}[d;ser;i];
            `series`actual`predicted!(ser;dec d[`Yval] i;dec .qs.predict[m;d[`Xval] i])];
        ()!()];
    `type`run`row`history`train`val`confusion`labels`plot`layers`unitScale`naiveMae`fit`inputs`features`bestEpoch`config!(
        `done;id;.dash.runs id;h;tr;va;conf;labels;plot;.qs.summary m;sd;naive;fit;
        $[`enc in key d; .qs.data.inputNames d`enc; `$()];.dash.features $[`enc in key d;d`enc;()];mt`bestEpoch;
        `hidden`act`out`loss`epochs`lr`optimiser#mt`config)
 }

/ commands: every function in .dash.api can be called by name

.dash.api.hello:{[r] `type`datasets`models`runs`defaults!(`hello;.dash.datasets[];.dash.saved[];.dash.runList[];
    `epochs`batch`lr`optimiser`hidden`act!(50;32;0.01;`adam;enlist 16;`tanh))}

.dash.api.datasets:{[r] `type`datasets!(`datasets;.dash.datasets[])}

/ CSV text -> a named in-memory table; returns its columns with their kinds and a preview
.dash.api.upload:{[r]
    n:.dash.sym[r;`name;`upload];
    n:`$ssr[;"[^A-Za-z0-9_]";"_"] string n;
    if[n in .qs.data.names; n:`$"my_",string n];
    t:flip .qs.data.parseCsv "\n" vs ssr[.dash.f[r;`csv;""];"\r";""];
    .dash.uploads[n]:t;
    `type`name`rows`columns`preview`datasets!(`uploaded;n;count t;
        ([] name:cols t; kind:{$[11h=type x;`cat;`num]} each value flip t; levels:{count distinct x} each value flip t);
        5 sublist t;.dash.datasets[])
 }

/ train a model, streaming an "epoch" message per epoch, then "done" with the full report
.dash.api.train:{[r]
    system "S ",string .dash.int[r;`seed;42];
    d:.dash.loadData r;
    c:.dash.config[r;d];
    nIn:count first d`X; nOut:count first d`Y;
    id:1+max 0,key[.dash.runs]`id;
    task:$[not .qs.isClass d`Y; `regression; nOut=1; `binary; `classes];
    .dash.send `type`run`data`arch`rows`valRows`epochs`task!(`start;id;.dash.sym[r;`data;`];.dash.arch[nIn;nOut;c];
        count d`X;count d`Xval;c`epochs;task);
    if[1<k:.dash.int[r;`cv;0];
        cv:.qs.crossValidate[d;k;c];
        .dash.send `type`run`folds`summary!(`cv;id;cv`folds;cv`summary)];
    m:.qs.init[nIn;c[`hidden],nOut;((count c`hidden)#c`act),c`out;c`seed];
    t0:.z.p;
    res:.qs.train[m;d;(`hidden`act`out _ c),`verbose`onEpoch!(0;{[id;row] .dash.send `type`run`row!(`epoch;id;row)}[id])];
    va:$[`Xval in key d; .qs.evaluate[res`model;d`Xval;d`Yval;c`loss]; `loss`acc`r2!3#0n];
    `.dash.runs upsert (id;.z.p;.dash.sym[r;`data;`];.dash.arch[nIn;nOut;c];task;c`loss;count res`history;va`loss;va`acc;$[task=`regression;va`r2;0n];1e-9*.z.p-t0;`trained);
    .dash.model[id]:res`model; .dash.hist[id]:res`history;
    .dash.meta[id]:`data`loss`task`config`bestEpoch!(d;c`loss;task;c;res`bestEpoch);
    .dash.report id
 }

/ search: one "trial" message per configuration as it finishes, then "searched" with the ranked table
/ space: {"lr":[0.01,0.001],"hidden":[[8],[16,16]],"act":["tanh","relu"]}; trials: sample size or null
.dash.api.search:{[r]
    system "S ",string .dash.int[r;`seed;42];
    d:.dash.loadData r;
    base:.dash.config[r;d];
    sp:.dash.f[r;`space;()!()];
    if[not count sp; '"search needs a space: option -> list of values"];
    sp:key[sp]!{[k;v] $[k=`hidden; {"j"$x} each v; k in `act`optimiser`schedule`loss`out; `$v;
        k in `epochs`batch`patience`seed`stepEvery; "j"$v; "f"$v]}'[key sp;value sp];
    g:.qs.grid[sp;.dash.int[r;`trials;0N]];
    .dash.send `type`count`keys!(`searching;count g;key sp);
    res:{[d;base;i;c] row:c,.qs.fitScore[d;base,c]; .dash.send `type`i`row!(`trial;i;row); row}[d;base]'[til count g;g];
    `type`results!(`searched;`valLoss xasc res)
 }

/ predictions: rows (objects keyed by column) for table models, or inputs (numbers) for any model
.dash.api.predict:{[r]
    id:.dash.int[r;`run;0N];
    if[not id in key .dash.model; '"no such run: ",string id];
    m:.dash.model id; d:.dash.meta[id]`data;
    $[`rows in key r;
        [if[not `enc in key d; '"this model wasn't trained on a table: send inputs"];
            t:r`rows; if[99h=type t; t:enlist t]; if[not 98h=type t; '"rows must be objects with the same columns"];
            A:.qs.predict[m;.qs.data.encodeX[d`enc;t]];
            out:t,'([] prediction:.qs.data.decodeY[d`enc;A]);
            if[1<count first A; out:out,'([] confidence:max each A)];
            `type`run`rows!(`predicted;id;out)];
        [X:"f"$r`inputs; if[0h<type X; X:enlist X];
            if[not (count first X)=count first first m`weight; '"the model takes ",(string count first first m`weight)," inputs"];
            `type`run`outputs!(`predicted;id;.qs.predict[m;X])]]
 }

/ save a run's model (with its encoding, if any) to models/name.qsm
.dash.api.save:{[r]
    id:.dash.int[r;`run;0N];
    if[not id in key .dash.model; '"no such run: ",string id];
    n:ssr[string .dash.sym[r;`name;`$"run",string id];"[^A-Za-z0-9_-]";"_"];
    m:.dash.model id; d:.dash.meta[id]`data;
    if[`enc in key d; m[`enc]:d`enc];
    .qs.save[`$":",.dash.modelDir,"/",n,".qsm";m];
    `type`name`models!(`saved;`$n;.dash.saved[])
 }

/ load a saved model as a run (no history), for predictions
.dash.api.load:{[r]
    n:ssr[string .dash.sym[r;`name;`];"[^A-Za-z0-9_-]";"_"];
    f:`$":",.dash.modelDir,"/",n,".qsm";
    if[()~key f; '"no saved model ",n];
    m:.qs.load f;
    id:1+max 0,key[.dash.runs]`id;
    nIn:count first first m`weight;
    d:`X`Y!(enlist nIn#0f;enlist (count last m`bias)#0f);
    if[`enc in key m; d[`enc]:m`enc];
    a:("-" sv string nIn,count each m`weight),$[1<count m`act; " ",string first m`act; ""]," ",string last m`act;
    `.dash.runs upsert (id;.z.p;`$n;a;$[`enc in key m; $[`cat=(m[`enc] first exec col from m[`enc] where role=`target)`kind;`classes;`regression]; `saved];`;0;0n;0n;0n;0n;`loaded);
    .dash.model[id]:`weight`bias`act#m; .dash.hist[id]:();
    .dash.meta[id]:`data`loss`task`config`bestEpoch!(d;`;`saved;()!();0N);
    `type`run`runs`inputs`features`nIn`target!(`loaded;id;.dash.runList[];$[`enc in key m; .qs.data.inputNames m`enc; `$()];
        .dash.features $[`enc in key m;m`enc;()];nIn;$[`enc in key m; first exec col from m[`enc] where role=`target; `])
 }

.dash.api.report:{[r] id:.dash.int[r;`run;0N]; if[not id in key .dash.hist; '"no such run"]; .dash.report id}

.dash.api.forget:{[r]
    i:.dash.int[r;`run;0N];
    .dash.runs:delete from .dash.runs where id=i;
    .dash.model:enlist[i]_.dash.model; .dash.hist:enlist[i]_.dash.hist; .dash.meta:enlist[i]_.dash.meta;
    `type`runs!(`runs;.dash.runList[])
 }

.dash.api.runs:{[r] `type`runs!(`runs;.dash.runList[])}

/ one WebSocket message: parse, dispatch, send the reply; errors go back to the client as messages
.dash.handle:{[x]
    r:@[.j.k;x;{(::)}];
    if[not 99h=type r; :.dash.send `type`message!(`error;"send a JSON object")];
    .dash.reqId:$[`id in key r; enlist[`id]!enlist r`id; ()!()];
    c:.dash.sym[r;`cmd;`];
    if[not c in 1_key .dash.api; :.dash.send `type`message!(`error;"unknown command: ",string c)];
    res:.Q.trp[.dash.api c;r;{[e;b] (`dashErr;e;b)}];
    if[(`dashErr~first res) and .dash.debug; -2 res[1],"\n",.Q.sbt res 2];
    $[`dashErr~first res; .dash.send `type`cmd`message!(`error;c;res 1); .dash.send res];
 }
.z.ws:{[x] .dash.handle x}

/ static files: only names directly inside ui/
.dash.types:`html`js`css`svg`png`ico!("text/html; charset=utf-8";"text/javascript; charset=utf-8";"text/css; charset=utf-8";"image/svg+xml";"image/png";"image/x-icon");
.dash.http:{[code;ctype;body]
    "HTTP/1.1 ",code,"\r\nContent-Type: ",ctype,"\r\nContent-Length: ",(string count body),"\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",body
 }
.dash.file:{[path]
    p:first "?" vs path; if[p~""; p:"index.html"];
    if[not p like "[A-Za-z0-9_]*"; :.dash.http["404 Not Found";"text/plain";"not found"]];
    if[any not p in .Q.an,"."; :.dash.http["404 Not Found";"text/plain";"not found"]];
    f:`$":",.dash.root,"/ui/",p;
    if[()~key f; :.dash.http["404 Not Found";"text/plain";"not found"]];
    e:`$last "." vs p;
    .dash.http["200 OK";$[e in key .dash.types;.dash.types e;"application/octet-stream"];"c"$read1 f]
 }
.z.ph:{[x] .dash.file first x}

.dash.o:.Q.opt .z.x;
.dash.port:$[`port in key .dash.o; first .dash.o`port; "8080"];
.dash.host:$[`host in key .dash.o; first .dash.o`host; "127.0.0.1"];
if[.dash.main;
    @[system;"p ",.dash.host,":",.dash.port;{-2 "can't listen on port ",.dash.port,": ",x; exit 1}];
    -1 "qSolve dashboard: http://",.dash.host,":",.dash.port,"/"];
