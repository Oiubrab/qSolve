/ train.q end to end, run as a separate q process
/ the command goes through env: system treats a leading "q" as its own \q command

.t.add[`cli_xor;{
    f:getenv[`TMPDIR],"/qsolve_cli.qsm"; f:$[f~"/qsolve_cli.qsm";"/tmp/qsolve_cli.qsm";f];
    out:system "env ",(first .z.X)," ",.t.root,"/train.q -q -data xor -hidden 4 -epochs 300 -batch 4 -lr 0.5 -verbose 0 -save ",f," </dev/null";
    .t.ok[any out like "final: train acc 1.0000*";"xor not solved: ",-3!out];
    m:.qs.load `$f;
    .t.eq[.qs.classes .qs.predict[m;.qs.data.gate[`xor]`X];0 1 1 0];
    hdel hsym `$f;
 }];
