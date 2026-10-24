/ encoding tables and CSVs, regression metrics

.t.tbl:([] x:1 2 3 4f; n:10 0N 30 20; c:`a`b`a`c; s:("p";"q";"p";"p"); y:`no`yes`no`yes)

.t.add[`table_fit_columns;{
    e:.qs.data.fit[.t.tbl;();`y];
    .t.eq[exec col from e;`x`n`c`s`y];
    .t.eq[exec role from e;`feature`feature`feature`feature`target];
    .t.eq[exec kind from e;`num`num`cat`cat`cat];
    .t.near[e[`x]`mu`sd;(2.5;dev 1 2 3 4f);1e-12];
    .t.near[e[`n]`mu;20f;1e-12];
    .t.eq[e[`c]`levels;`a`b`c];
    .t.eq[e[`s]`levels;`p`q];
    .t.eq[exec col from .qs.data.fit[.t.tbl;`x`c;`y];`x`c`y];
    .t.throws[{.qs.data.fit[.t.tbl;`x`nope;`y]};::];
 }];

.t.add[`table_encode;{
    e:.qs.data.fit[.t.tbl;`x`n`c;`y];
    X:.qs.data.encodeX[e;.t.tbl];
    .t.eq[count each X;4#5];
    .t.near[X[;0];(1 2 3 4f-2.5)%dev 1 2 3 4f;1e-12];
    .t.eq[X[1;1];0f];
    .t.eq[X[;2 3 4];(1 0 0f;0 1 0f;1 0 0f;0 0 1f)];
    .t.eq[.qs.data.encodeY[e;.t.tbl];(1 0f;0 1f;1 0f;0 1f)];
    / unseen and missing levels encode as zeros
    .t.eq[.qs.data.encodeX[e;([] x:enlist 2.5; n:enlist 20; c:enlist `zzz)];enlist 0 0 0 0 0f];
    .t.eq[.qs.data.encodeX[e;([] x:enlist 2.5; n:enlist 20; c:enlist `)];enlist 0 0 0 0 0f];
    .t.eq[.qs.data.inputNames e;`x`n`c_a`c_b`c_c];
 }];

.t.add[`table_decode;{
    e:.qs.data.fit[.t.tbl;`x`c;`y];
    .t.eq[.qs.data.decodeY[e;(0.9 0.1;0.2 0.8)];`no`yes];
    e:.qs.data.fit[.t.tbl;`c`y;`x];
    .t.near[.qs.data.decodeY[e;.qs.data.encodeY[e;.t.tbl]];1 2 3 4f;1e-12];
 }];

/ the encoding is fitted on training rows only
.t.add[`table_from_table;{
    system "S 1"; t:([] x:til 100; y:100?`a`b);
    d:.qs.data.fromTable[t;();`y;0.8];
    .t.eq[count each d`X`Y`Xval`Yval;80 80 20 20];
    e:d`enc;
    .t.ok[not (e[`x]`mu)=avg til 100;"encoding used validation rows"];
    .t.near[avg raze d`X;0f;1e-12];
    .t.eq[key .qs.data.fromTable[t;();`y;1f];`X`Y`enc];
 }];

.t.add[`table_csv;{
    f:`$":",getenv[`TMPDIR],"/qsolve_test.csv"; f:$[""~getenv`TMPDIR; `:/tmp/qsolve_test.csv; f];
    f 0: ("sepal length,kind,count";"1.5,a,3";",b,";"2.5,a,7";"3.5,c,x");
    t:.qs.data.readCsv f;
    .t.eq[key t;`sepallength`kind`count1];
    .t.eq[t`sepallength;1.5 0n 2.5 3.5];
    .t.eq[t`kind;`a`b`a`c];
    .t.eq[t`count1;`3``7`x];
    d:.qs.data.csv[f;`sepallength;`kind;1f];
    .t.eq[count first d`Y;3];
    hdel f;
 }];

.t.add[`metrics_regression;{
    r:.qs.regression[(enlist 1f;enlist 2f;enlist 4f);(enlist 1f;enlist 3f;enlist 3f)];
    .t.near[r`mae;2%3;1e-12];
    .t.near[r`rmse;sqrt 2%3;1e-12];
    .t.near[r`r2;1-2%sum {x*x} 1 3 3f-7%3;1e-12];
    .t.ok[.qs.isClass (1 0f;0 1f);"one-hot not classes"];
    .t.ok[.qs.isClass (enlist 1f;enlist 0f);"0/1 column not classes"];
    .t.ok[not .qs.isClass (enlist 1.5;enlist 0f);"numbers taken as classes"];
    m:`weight`bias`act!(enlist 1 1#2f;enlist enlist 0f;enlist `linear);
    e:.qs.evaluate[m;(enlist 1f;enlist 2f);(enlist 2f;enlist 4.5);`mse];
    .t.ok[null e`acc;"accuracy for a regression target"];
    .t.near[e`mae;0.25;1e-12];
 }];

/ regression through a table: house prices with numeric, categorical and missing values
.t.add[`table_houses_regression;{
    system "S 1"; d:.qs.data.fromTable[.qs.data.houses 600;();`price;0.8];
    .t.eq[count first d`X;7];
    r:.qs.train[.qs.init[7;16 1;`tanh`linear;1];d;`loss`optimiser`lr`epochs`verbose!(`mse;`adam;0.01;60;0)];
    .t.ok[0.9<.qs.evaluate[r`model;d`Xval;d`Yval;`mse]`r2;"houses R squared below 0.9"];
 }];

.t.add[`table_raw_validation_rows;{
    system "S 1"; d:.qs.data.fromTable[.qs.data.houses 100;`area`city;`price;0.8];
    .t.eq[cols d`Tval;`area`city`price];
    .t.eq[count d`Tval;20];
    .t.near[.qs.data.encodeX[d`enc;d`Tval];d`Xval;1e-12];
    e:d`enc;
    .t.ok[(e[`area]`lo)<e[`area]`hi;"no training range"];
    .t.ok[null e[`city]`lo;"no range for categories"];
 }];
