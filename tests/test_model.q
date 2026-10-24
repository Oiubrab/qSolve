/ init, model checks, forward pass, save and load

.t.add[`init_shapes;{
    m:.qs.init[5;4 3 2;`tanh`relu`softmax;1];
    .t.eq[count each m`weight;4 3 2];
    .t.eq[count each first each m`weight;5 4 3];
    .t.eq[m`bias;(4#0f;3#0f;2#0f)];
    .t.eq[m`act;`tanh`relu`softmax];
    .t.eq[.qs.init[5;4 2;`sigmoid;1]`act;`sigmoid`sigmoid];
 }];

.t.add[`init_seeded;{
    .t.eq[.qs.init[3;4 2;`tanh;7];.qs.init[3;4 2;`tanh;7]];
    .t.ok[not .qs.init[3;4 2;`tanh;7]~.qs.init[3;4 2;`tanh;8];"different seeds gave the same model"];
 }];

/ symmetric and within the Xavier limit
.t.add[`init_distribution;{
    w:raze first .qs.init[100;100;`tanh;3]`weight;
    .t.ok[0.01>abs avg w;"weights not centred on zero"];
    .t.ok[(sqrt 6%200)>=max abs w;"weights outside Xavier limit"];
    .t.ok[(sqrt 6%200)>max w;"no negative weights"];
 }];

.t.add[`check_rejects_bad_models;{
    m:.qs.init[3;4 2;`tanh;1];
    .t.throws[.qs.check;@[m;`act;:;`tanh`nope]];
    .t.throws[.qs.check;@[m;`act;:;`softmax`tanh]];
    .t.throws[.qs.check;@[m;`bias;:;(3#0f;2#0f)]];
    .t.throws[.qs.check;@[m;`weight;:;(4 3#0f;2 5#0f)]];
    .t.throws[.qs.check;`weight`bias!(m`weight;m`bias)];
 }];

/ T2: 2-2-1 sigmoid net with hand-picked weights, expected value computed by hand
.t.add[`forward_hand_computed;{
    m:`weight`bias`act!((2 2#0.1 0.2 0.3 0.4;enlist 0.7 0.8);(0.5 0.6;enlist 0.9);`sigmoid`sigmoid);
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
