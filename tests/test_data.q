/ dataset generators and loaders

.t.add[`data_gates;{
    .t.eq[.qs.data.gate[`xor]`Y;enlist each 0 1 1 0f];
    .t.eq[.qs.data.gate[`and]`Y;enlist each 0 0 0 1f];
    .t.eq[.qs.data.gate[`or]`Y;enlist each 0 1 1 1f];
    .t.eq[.qs.data.gate[`nand]`Y;enlist each 1 1 1 0f];
 }];

.t.add[`data_one_hot;{.t.eq[.qs.data.oneHot[3;2 0 1];(0 0 1f;1 0 0f;0 1 0f)]}];

.t.add[`data_four_bit;{
    system "S 1"; d:.qs.data.fourBit 500;
    .t.eq[count each d`X`Y;500 500];
    .t.ok[all raze d[`X] in 0 1f;"inputs not bits"];
    .t.eq[d[`Y] where d[`X]~\:1 1 0 0f;(sum d[`X]~\:1 1 0 0f)#enlist 1 0 0f];
    .t.eq[d[`Y] where d[`X]~\:0 0 1 1f;(sum d[`X]~\:0 0 1 1f)#enlist 0 1 0f];
    .t.ok[0.1<avg d[`Y][;0];"too few 1100 rows"];
 }];

.t.add[`data_sine;{
    system "S 1"; d:.qs.data.sine 200;
    .t.ok[all raze (d`X) within -1 1;"X out of range"];
    .t.ok[all raze (d`Y) within 0 1;"Y out of range"];
    .t.near[raze d`Y;0.5+0.5*sin .qs.data.pi*1+raze d`X;1e-12];
 }];

.t.add[`data_2d;{
    system "S 1";
    {d:x[400;0f];
        .t.eq[count each d`X`Y;400 400];
        .t.eq[count first d`X;2];
        .t.eq[sum d`Y;200 200f];
    } each (.qs.data.circles;.qs.data.moons;.qs.data.spirals);
    r:{sqrt sum x*x} each .qs.data.circles[400;0f]`X;
    .t.near[asc distinct 1e-9 xbar r;0.5 1;1e-8];
 }];

.t.add[`data_split_standardise;{
    system "S 1"; d:.qs.data.split[.qs.data.moons[100;0.1];0.8];
    .t.eq[count each d`X`Y`Xval`Yval;80 80 20 20];
    s:.qs.data.standardise d;
    .t.near[avg s`X;0 0f;1e-12];
    .t.near[dev s`X;1 1f;1e-12];
    .t.eq[.qs.data.stats[(1 5f;1 6f)]`sd;1 0.5];
 }];

/ write a tiny IDX file and read it back
.t.add[`data_idx;{
    f:`$":",getenv[`TMPDIR],"/qsolve_test.idx";
    f:$[""~getenv`TMPDIR; `:/tmp/qsolve_test.idx; f];
    f 1: 0x00000803,(raze 0x0 vs/: 2 2 3i),"x"$til 12;
    .t.eq[.qs.data.idx f;(2;6)#"x"$til 12];
    f 1: 0x00000801,(0x0 vs 3i),0x070809;
    .t.eq[.qs.data.idx f;7 8 9];
    f 1: 0x00000801,(0x0 vs 5i),0x070809;
    .t.throws[.qs.data.idx;f];
    f 1: 0x00000d01,(0x0 vs 3i),0x070809;
    .t.throws[.qs.data.idx;f];
    hdel f;
 }];

.t.add[`data_mnist;{
    if[()~key .t.path "data/mnist/train-images-idx3-ubyte"; .t.skip "run scripts/fetch_data.sh mnist"];
    d:.qs.data.mnist .t.path "data/mnist";
    .t.eq[count each d`X`Y`Xval`Yval;60000 60000 10000 10000];
    .t.eq[count first d`X;784];
    .t.ok[all raze (100#d`X) within 0 1;"pixels not scaled"];
    .t.eq[.qs.classes 10#d`Y;5 0 4 1 9 2 1 3 1 4];
 }];

.t.add[`data_iris;{
    if[()~key .t.path "data/iris/iris.data"; .t.skip "run scripts/fetch_data.sh iris"];
    system "S 1"; d:.qs.data.iris .t.path "data/iris/iris.data";
    .t.eq[count each d`X`Y`Xval`Yval;120 120 30 30];
    .t.eq[sum (d`Y),d`Yval;50 50 50f];
 }];
