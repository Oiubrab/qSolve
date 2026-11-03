/ 1-D convolution layers (G5.2), as the layers seq, conv1d, act, maxpool1d and flatten

.t.convCase:{[layers;sizes;acts;loss;seed]
    system "S ",string seed;
    m:.t.randomBiases .qs.initConv[2;24;layers;sizes;acts;0N];
    X:{x?1f} each 3#48;
    k:last sizes;
    Y:"f"$$[loss=`xent; {x=til y}[;k] each 3?k; {x?1f} each 3#k];
    .t.gradDiff[m;X;Y;loss;.qs.grad[m;X;Y;loss]]
 }

.t.add[`conv_gradients;{
    / smooth activations, pooling, two conv layers, then dense
    e1:.t.convCase[([] filters:3 2; kernel:5 3; pool:2 2; act:`tanh`sigmoid);4 1;`tanh`sigmoid;`bce;1];
    .t.ok[e1<1e-6;"tanh/sigmoid conv gradient error ",string e1];
    / one conv layer straight into a softmax output
    e2:.t.convCase[([] filters:enlist 4; kernel:enlist 3; pool:enlist 3; act:enlist `tanh);enlist 3;`softmax;`xent;2];
    .t.ok[e2<1e-6;"conv into softmax gradient error ",string e2];
    / relu (kinks are measure-zero for random data) with pool 1
    e3:.t.convCase[([] filters:2 3; kernel:4 2; pool:1 2; act:`relu`relu);2 1;`relu`linear;`mse;3];
    .t.ok[e3<1e-6;"relu conv gradient error ",string e3];
 }];

/ a layer with no pool (pool 1) has no pooling layer at all
.t.add[`conv_pool_one_has_no_pool_layer;{
    m:.qs.initConv[2;24;([] filters:2 3; kernel:4 2; pool:1 2; act:`relu`relu);2 1;`relu`sigmoid;1];
    .t.eq[.qs.kinds m;`seq`conv1d`act`conv1d`act`maxpool1d`flatten`dense`act`dense`act];
 }];

.t.add[`conv_shapes_and_checks;{
    m:.qs.initConv[2;64;([] filters:3 2; kernel:5 3; pool:2 2; act:`relu`tanh);4 1;`tanh`sigmoid;1];
    K:.qs.kinds m;
    .t.eq[K;`seq`conv1d`act`maxpool1d`conv1d`act`maxpool1d`flatten`dense`act`dense`act];
    .t.eq[.qs.inputSize m;128];
    / the shape after each layer: (channels;length) through the conv stack, 28 features into the dense layers
    sh:.qs.shapes m;
    .t.eq[sh 0 1 3 4 6;(2 64;3 60;3 30;2 28;2 14)];
    .t.eq[sh 7;enlist 28];
    / kernels: one row per filter, kernel*channels columns
    P:m[`params] where K=`conv1d;
    .t.eq[count each P[;0];3 2];
    .t.eq[count each first each P[;0];10 9];
    .t.eq[count first first first m[`params] where K=`dense;28];
    s:.qs.summary m;
    .t.eq[s`kind;`conv`conv`dense`dense];
    .t.eq[s`params;33 20 116 5];
    ic:where K=`conv1d; id:where K=`dense;
    / kernels swapped between the layers, a dense weight of the wrong width
    .t.throws[.qs.check;.[m;(`params;first ic;0);:;m[`params;last ic;0]]];
    .t.throws[.qs.check;.[m;(`params;first id;0);:;4 10#0f]];
    .t.throws[{.qs.initConv[1;8;([] filters:enlist 2; kernel:enlist 9; pool:enlist 1; act:enlist `relu);1;`sigmoid;1]};::];
    / a single example vector works like a one-row batch
    x:128?1f;
    .t.near[.qs.predict[m;x];first .qs.predict[m;enlist x];1e-12];
 }];

/ sharding over threads gives the same conv gradient as one pass
.t.add[`conv_sharded;{
    m:.qs.initConv[2;24;([] filters:3 2; kernel:5 3; pool:2 2; act:`tanh`relu);4 1;`tanh`sigmoid;1];
    system "S 5"; X:{x?1f} each 100#48; Y:enlist each "f"$100?0 1;
    r:.t.withThreads[4;{[m;X;Y;u] .qs.grad[m;X;Y;`bce]}[m;X;Y]];
    s:.qs.grad1[m;X;Y;`bce;(count m`layers)#1b];
    .t.near[r;s;1e-12];
 }];

.t.add[`conv_save_load_train;{
    m:.qs.initConv[1;16;([] filters:enlist 2; kernel:enlist 3; pool:enlist 2; act:enlist `relu);enlist 1;`sigmoid;1];
    f:.t.tmpFile "qsolve_conv_model"; .qs.save[f;m]; .t.eq[.qs.load f;m]; hdel f;
    / adam and l2 update the conv parameters too
    system "S 2"; X:{x?1f} each 32#16; Y:enlist each "f"$32?0 1;
    r:.qs.train[m;`X`Y!(X;Y);`epochs`optimiser`lr`l2`verbose!(3;`adam;0.01;0.01;0)];
    ci:first where `conv1d=.qs.kinds m;
    .t.ok[not r[`model;`params;ci;0]~m[`params;ci;0];"conv kernels didn't move"];
    .t.ok[not r[`model;`params;ci;1]~m[`params;ci;1];"conv biases didn't move"];
 }];

/ a pulse at a random place in noise: a small CNN finds it, an MLP of similar size struggles (the plan's toy check)
.t.add[`conv_finds_pulse_anywhere;{
    system "S 13"; n:600; len:64;
    shape:0.9 -1.8 1.8 -0.9;
    make:{[shape;len;n] y:n?0 1; x:{[shape;len;y] v:0.5*.qs.data.normal len; $[y; @[v;(first 1?len-4)+til 4;+;shape]; v]}[shape;len] each y;
        `X`Y!(x;enlist each "f"$y)}[shape;len];
    tr:make n; va:make 400;
    o:`epochs`batch`lr`loss`optimiser`verbose!(40;32;0.01;`bce;`adam;0);
    cnn:.qs.initConv[1;len;([] filters:enlist 4; kernel:enlist 4; pool:enlist 61; act:enlist `relu);enlist 1;`sigmoid;1];
    mlp:.qs.init[len;4 1;`relu`sigmoid;1];
    ac:.qs.evaluate[.qs.train[cnn;tr;o]`model;va`X;va`Y;`bce]`acc;
    am:.qs.evaluate[.qs.train[mlp;tr;o]`model;va`X;va`Y;`bce]`acc;
    .t.ok[ac>0.9;"CNN accuracy ",string ac];
    .t.ok[ac>am+0.1;"CNN ",(string ac)," should clearly beat the MLP ",string am];
    .t.ok[(sum .qs.summary[cnn]`params)<sum .qs.summary[mlp]`params;"the CNN should be the smaller model"];
 }];

/ predicting in blocks gives the same outputs as one pass
.t.add[`conv_predict_blocks;{
    m:.qs.initConv[2;24;([] filters:3 2; kernel:5 3; pool:2 2; act:`tanh`relu);4 1;`tanh`sigmoid;1];
    system "S 6"; X:{x?1f} each 50#48;
    b:.qs.predictBlock; .qs.predictBlock:7; p:.qs.predict[m;X]; .qs.predictBlock:b;
    .t.near[p;.qs.infer[m;X;count m`layers];1e-12];
 }];

/ the raw output is the logit of the sigmoid output, and it keeps order where probabilities round to 1
.t.add[`predict_raw_logit;{
    m:.qs.initConv[2;24;([] filters:3 2; kernel:5 3; pool:2 2; act:`tanh`relu);4 1;`tanh`sigmoid;1];
    system "S 6"; X:{x?1f} each 20#48;
    lz:first each .qs.predictRaw[m;X]; p:first each .qs.predict[m;X];
    .t.near[reciprocal 1+exp neg lz;p;1e-12];
    d:.qs.init[3;4 1;`tanh`sigmoid;1];
    .t.near[first each .qs.predictRaw[d;(1 2 3f;3 2 1f)];{log x%1-x} first each .qs.predict[d;(1 2 3f;3 2 1f)];1e-9];
 }];
