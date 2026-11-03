/ init, model checks, forward pass, save and load

.t.add[`init_shapes;{
    m:.qs.init[5;4 3 2;`tanh`relu`softmax;1];
    .t.eq[.qs.kinds m;`dense`act`dense`act`dense`act];
    .t.eq[.qs.sizes m;4 3 2];
    .t.eq[.qs.acts m;`tanh`relu`softmax];
    .t.eq[(.qs.inputSize m;.qs.outputSize m);5 2];
    P:m[`params] 0 2 4;
    .t.eq[count each P[;0];4 3 2];
    .t.eq[count each first each P[;0];5 4 3];
    .t.eq[P[;1];(4#0f;3#0f;2#0f)];
    .t.eq[.qs.acts .qs.init[5;4 2;`sigmoid;1];`sigmoid`sigmoid];
    / every layer records the shapes it takes and gives
    .t.eq[.qs.shapes m;(enlist 4;enlist 4;enlist 3;enlist 3;enlist 2;enlist 2)];
 }];

.t.add[`init_seeded;{
    .t.eq[.qs.init[3;4 2;`tanh;7];.qs.init[3;4 2;`tanh;7]];
    .t.ok[not .qs.init[3;4 2;`tanh;7]~.qs.init[3;4 2;`tanh;8];"different seeds gave the same model"];
 }];

/ symmetric and within the Xavier limit
.t.add[`init_distribution;{
    w:raze first first .qs.init[100;100;`tanh;3]`params;
    .t.ok[0.01>abs avg w;"weights not centred on zero"];
    .t.ok[(sqrt 6%200)>=max abs w;"weights outside Xavier limit"];
    .t.ok[(sqrt 6%200)>max w;"no negative weights"];
    / relu layers use the He limit
    h:raze first first .qs.init[100;100;`relu;3]`params;
    .t.ok[(sqrt 6%100)>=max abs h;"weights outside He limit"];
    .t.ok[(sqrt 6%200)<max abs h;"relu weights not wider than Xavier's"];
 }];

.t.add[`check_rejects_bad_models;{
    m:.qs.init[3;4 2;`tanh;1];
    .t.eq[.qs.check m;m];
    / layers: 0 dense, 1 act, 2 dense, 3 act
    .t.throws[.qs.check;.[m;(`layers;1;1;`f);:;`nope]];
    .t.throws[.qs.check;.[m;(`layers;1;1;`f);:;`softmax]];
    .t.throws[.qs.check;.[m;(`layers;1;0);:;`nope]];
    .t.throws[.qs.check;.[m;(`params;0;1);:;3#0f]];
    .t.throws[.qs.check;.[m;(`params;0;0);:;4 5#0f]];
    .t.throws[.qs.check;.[m;(`layers;2;1;`inShape);:;enlist 5]];
    .t.throws[.qs.check;.[m;(`layers;0;1;`outShape);:;enlist 5]];
    .t.throws[.qs.check;@[m;`params;1_]];
    .t.throws[.qs.check;`foo`bar!(1;2)];
    .t.throws[.qs.check;enlist[`layers]!enlist m`layers];
 }];

/ T2: 2-2-1 sigmoid net with hand-picked weights, expected value computed by hand
.t.add[`forward_hand_computed;{
    m:.qs.fromWeights[(2 2#0.1 0.2 0.3 0.4;enlist 0.7 0.8);(0.5 0.6;enlist 0.9);`sigmoid`sigmoid];
    .t.near[.qs.predict[m;1 2f];enlist 0.8897479186937528;1e-12];
    .t.near[.qs.predict[m;(1 2f;1 2f)];(enlist 0.8897479186937528;enlist 0.8897479186937528);1e-12];
    .t.near[.qs.predict[m;1 2];enlist 0.8897479186937528;1e-12];
 }];

.t.add[`save_load_roundtrip;{
    m:.qs.init[3;4 2;`relu`softmax;1];
    f:`$":",getenv[`TMPDIR],"/qsolve_test_model";
    f:$[""~getenv`TMPDIR; `:/tmp/qsolve_test_model; f];
    .qs.save[f;m];
    .t.eq[.qs.load f;m];
    hdel f;
 }];

.t.add[`model_summary;{
    s:.qs.summary .qs.init[4;8 3;`tanh`softmax;1];
    .t.eq[cols s;`layer`inputs`outputs`act`params];
    .t.eq[s`inputs;4 8];
    .t.eq[s`outputs;8 3];
    .t.eq[s`params;40 27];
 }];
