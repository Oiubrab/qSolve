/ serve a saved model over kdb+ IPC
/   q serve.q -model price.qsm -p 5050 [-s 4]
/ clients call (`function;argument), e.g. h (`predict;X) or h (`predictTable;newRows); anything else,
/ including strings of q code, is refused
/   predict       matrix of inputs (one row per example) or one example vector -> model outputs
/   predictTable  table with the training feature columns -> the table plus prediction (and confidence for
/                 classes); needs a model saved by train.q from table or csv data (it carries the encoding)
/   info          any argument -> `layers`inputs`target`port`pid

.srv.root:{$[count x;x;"."]} "/" sv -1_ "/" vs string .z.f;
system "l ",.srv.root,"/defineEquation.q";
system "l ",.srv.root,"/data.q";

.srv.o:.Q.opt .z.x;
if[not `model in key .srv.o; -2 "usage: q serve.q -model file.qsm -p port"; exit 1];
.srv.m:.qs.load hsym `$first .srv.o`model;
.srv.enc:$[`enc in key .srv.m; .srv.m`enc; ()];
.srv.nIn:count first first .srv.m`weight;

/ the functions clients may call
.srv.api.predict:{[X]
    if[not (0h<type X) or 0h=type X; '"predict takes a matrix or a vector"];
    if[not .srv.nIn=$[0h<type X; count X; count first X]; '"the model takes ",(string .srv.nIn)," inputs"];
    .qs.predict[.srv.m;X]
 };
.srv.api.predictTable:{[t]
    if[not count .srv.enc; '"this model has no encoding: use predict, or save it from table or csv data with train.q"];
    if[not 98h=type t; '"predictTable takes a table"];
    f:exec col from .srv.enc where role=`feature;
    if[count c:f except cols t; '"missing columns: ",", " sv string c];
    A:.qs.predict[.srv.m;.qs.data.encodeX[.srv.enc;t]];
    r:t,'([] prediction:.qs.data.decodeY[.srv.enc;A]);
    $[1<count first A; r,'([] confidence:max each A); r]
 };
.srv.api.info:{[x]
    `layers`inputs`target`port`pid!(.qs.summary .srv.m;$[count .srv.enc;.qs.data.inputNames .srv.enc;`$()];
        $[count .srv.enc;first exec col from .srv.enc where role=`target;`];system "p";.z.i)
 };

/ only (`function;argument) calls to .srv.api
.z.pg:{[x]
    if[10h=abs type x; '"send (`function;argument), not q code"];
    if[not (2=count x) and -11h=type first x; '"send (`function;argument)"];
    if[not (f:first x) in 1_key .srv.api; '"unknown function ",(string f),", expected one of ",", " sv string 1_key .srv.api];
    .srv.api[f] x 1
 };
.z.ps:{[x] '"async calls are not supported"};

if[0i=system "p"; -2 "no port: start with -p port"; exit 1];
-1 "serving ",(first .srv.o`model)," on port ",string system "p";
show .qs.summary .srv.m;
