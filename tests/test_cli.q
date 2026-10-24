/ train.q end to end, run as a separate q process
/ the command goes through env: system treats a leading "q" as its own \q command

.t.add[`cli_xor;{
    f:getenv[`TMPDIR],"/qsolve_cli.qsm"; f:$[f~"/qsolve_cli.qsm";"/tmp/qsolve_cli.qsm";f];
    out:system "env ",(first .z.X)," ",.t.root,"/train.q -q -data xor -hidden 4 -epochs 300 -batch 4 -lr 0.5 -verbose 0 -save ",f," </dev/null";
    .t.ok[any out like "final model: train acc 1.0000*";"xor not solved: ",-3!out];
    m:.qs.load `$f;
    .t.eq[.qs.classes .qs.predict[m;.qs.data.gate[`xor]`X];0 1 1 0];
    hdel hsym `$f;
 }];

/ a CSV regression through train.q; the saved model carries its column encoding
.t.add[`cli_csv;{
    d:getenv[`TMPDIR]; d:$[""~d;"/tmp";d];
    system "S 2"; (hsym `$d,"/qsolve_cli.csv") 0: csv 0: .qs.data.houses 400;
    out:system "env ",(first .z.X)," ",.t.root,"/train.q -q -data csv -file ",d,"/qsolve_cli.csv -target price -features area rooms city -epochs 30 -optimiser adam -lr 0.01 -verbose 0 -save ",d,"/qsolve_cli_csv.qsm </dev/null";
    .t.ok[any out like "final model: train R2 *";"no regression summary: ",-3!out];
    m:.qs.load hsym `$d,"/qsolve_cli_csv.qsm";
    .t.eq[exec col from m`enc;`area`rooms`city`price];
    .t.eq[count first first m`weight;6];
    hdel each hsym `$d,/:("/qsolve_cli.csv";"/qsolve_cli_csv.qsm");
 }];

.t.add[`cli_search;{
    out:system "env ",(first .z.X)," ",.t.root,"/train.q -q -data moons -rows 200 -optimiser adam -epochs 5 -verbose 0 -search lr=0.01,0.001 hidden=4,8x8 </dev/null";
    .t.ok[any out like "best: *";"no best configuration: ",-3!out];
    .t.ok[any out like "final model: *";"best configuration not trained"];
 }];
