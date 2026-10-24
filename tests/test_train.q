/ training loop, optimisers, metrics

.t.bits:(0 0f;0 1f;1 0f;1 1f)
.t.gate:{[f] `X`Y!(.t.bits;enlist each "f"$f .' .t.bits)}

/ T3: one sigmoid node learns AND and OR
.t.add[`train_and_or;{
    o:`epochs`batch`lr`loss`verbose!(300;4;1f;`bce;0);
    {[o;f] r:.qs.train[.qs.init[2;1;`sigmoid;1];.t.gate f;o]; .t.eq[last[r`history]`trainAcc;1f]}[o] each (&;|);
 }];

/ T4: XOR needs a hidden layer; every optimiser should get it
.t.add[`train_xor;{
    d:.t.gate {x<>y};
    m:.qs.init[2;4 1;`tanh`sigmoid;1];
    o:`epochs`batch`loss`verbose!(300;4;`bce;0);
    {[m;d;o;opt;lr] r:.qs.train[m;d;o,`optimiser`lr!(opt;lr)]; .t.ok[1f=last[r`history]`trainAcc;"xor not solved by ",string opt]}[m;d;o]'[`sgd`momentum`adam;0.5 0.2 0.05];
 }];

/ a single layer cannot solve XOR: guards against the test above passing for the wrong reason
.t.add[`train_xor_single_layer_fails;{
    r:.qs.train[.qs.init[2;1;`sigmoid;1];.t.gate {x<>y};`epochs`batch`lr`loss`verbose!(300;4;1f;`bce;0)];
    .t.ok[1f>last[r`history]`trainAcc;"single layer solved xor"];
 }];

.t.add[`train_history;{
    d:.t.gate {x<>y}; d[`Xval]:d`X; d[`Yval]:d`Y;
    r:.qs.train[.qs.init[2;4 1;`tanh`sigmoid;1];d;`epochs`verbose!(7;0)];
    h:r`history;
    .t.eq[cols h;`epoch`trainLoss`trainAcc`valLoss`valAcc`seconds];
    .t.eq[h`epoch;1+til 7];
    .t.ok[not any null h`valLoss;"missing validation loss"];
 }];

.t.add[`train_seeded;{
    d:.t.gate {x<>y}; m:.qs.init[2;4 1;`tanh`sigmoid;1]; o:`epochs`batch`verbose!(5;2;0);
    .t.eq[.qs.train[m;d;o]`model;.qs.train[m;d;o]`model];
 }];

/ batch 3 over 4 rows is two steps per epoch: the remainder row is not dropped
.t.add[`train_remainder_batch;{
    d:.t.gate {x<>y}; m:.qs.init[2;4 1;`tanh`sigmoid;1];
    o:.qs.defaults,`batch`optimiser`shuffle!(3;`adam;0b);
    s:.qs.epoch[o;d`X;d`Y;`model`state!(m;.qs.opt.init[`adam] `weight`bias#m)];
    .t.eq[s[`state]`t;2];
 }];

.t.add[`train_l2_shrinks_weights;{
    d:.t.gate {x<>y}; m:.qs.init[2;4 1;`tanh`sigmoid;1]; o:`epochs`batch`verbose!(50;4;0);
    w:{sum (raze/) {x*x} x[`model]`weight};
    .t.ok[w[.qs.train[m;d;o,enlist[`l2]!enlist 0.1]]<w .qs.train[m;d;o];"l2 did not shrink weights"];
 }];

.t.add[`train_rejects_bad_input;{
    m:.qs.init[2;4 1;`tanh`sigmoid;1]; d:.t.gate {x<>y};
    .t.throws[{.qs.train[x;y;()!()]}[m];@[d;`X;:;3#'d`X]];
    .t.throws[{.qs.train[x;y;()!()]}[m];@[d;`Y;:;2#'d`Y]];
    .t.throws[{.qs.train[x;y;()!()]}[m];@[d;`Y;:;-1_d`Y]];
    .t.throws[{.qs.train[x;y;enlist[`optimiser]!enlist `nope]}[m];d];
    .t.throws[{.qs.train[x;y;enlist[`loss]!enlist `nope]}[m];d];
 }];

.t.add[`metrics;{
    .t.eq[.qs.classes (0.1 0.9;0.8 0.2);1 0];
    .t.eq[.qs.classes (enlist 0.2;enlist 0.7);0 1];
    .t.near[.qs.accuracy[(0.1 0.9;0.8 0.2;0.6 0.4);(0 1f;0 1f;1 0f)];2%3;1e-12];
    m:`weight`bias`act!(enlist 2 2#1 0 0 1f;enlist 0 0f;enlist `linear);
    .t.eq[.qs.confusion[m;(1 0f;0 1f;1 0f);(1 0f;0 1f;0 1f)];(1 0;1 1)];
    .t.eq[.qs.evaluate[m;(1 0f;0 1f);(1 0f;0 1f);`mse];`loss`acc!0 1f];
 }];
