/ 1-D convolution layers (G5.2)

/ numeric gradient of every parameter (dense and conv) by central differences
.t.fdAll:{[m;X;Y;loss;h]
    L:{[X;Y;loss;m] .qs.loss.v[loss][.qs.predict[m;X];Y]}[X;Y;loss];
    fd:{[L;m;h;k;i] (L[@[m;k;.[;i;+;h]]]-L[@[m;k;.[;i;-;h]]])%2*h}[L;m;h];
    ixM:{[ms] raze {[l;w] raze l,/:'til[count w],/:\:til count first w}'[til count ms;ms]};
    ixV:{[vecs] raze {[l;b] l,/:til count b}'[til count vecs;vecs]};
    ks:.qs.params m;
    ixs:{[ixM;ixV;m;k] $[k in `weight`ckernel; ixM m k; ixV m k]}[ixM;ixV;m] each ks;
    ks!{[m;fd;k;ix] {.[x;y 0;:;y 1]}/[0f*m k;flip (ix;fd[k] each ix)]}[m;fd]'[ks;ixs]
 }

.t.convCase:{[layers;sizes;acts;loss;seed]
    system "S ",string seed;
    m:.qs.initConv[2;24;layers;sizes;acts;0N];
    m[`cbias]:{-0.3+(count x)?0.6} each m`cbias;
    X:{x?1f} each 3#48;
    k:last sizes;
    Y:"f"$$[loss=`xent; {x=til y}[;k] each 3?k; {x?1f} each 3#k];
    g:.qs.grad[m;X;Y;loss]; n:.t.fdAll[m;X;Y;loss;1e-6];
    {[g;n;k] max abs (raze/) g[k]-n k}[g;n] each .qs.params m
 }

.t.add[`conv_gradients;{
    / smooth activations, pooling, two conv layers, then dense
    e1:.t.convCase[([] filters:3 2; kernel:5 3; pool:2 2; act:`tanh`sigmoid);4 1;`tanh`sigmoid;`bce;1];
    .t.ok[all e1<1e-6;"tanh/sigmoid conv gradient error ",-3!e1];
    / one conv layer straight into a softmax output
    e2:.t.convCase[([] filters:enlist 4; kernel:enlist 3; pool:enlist 3; act:enlist `tanh);enlist 3;`softmax;`xent;2];
    .t.ok[all e2<1e-6;"conv into softmax gradient error ",-3!e2];
    / relu (kinks are measure-zero for random data) with pool 1
    e3:.t.convCase[([] filters:2 3; kernel:4 2; pool:1 2; act:`relu`relu);2 1;`relu`linear;`mse;3];
    .t.ok[all e3<1e-6;"relu conv gradient error ",-3!e3];
 }];

.t.add[`conv_shapes_and_checks;{
    m:.qs.initConv[2;64;([] filters:3 2; kernel:5 3; pool:2 2; act:`relu`tanh);4 1;`tanh`sigmoid;1];
    .t.eq[.qs.conv.lengths m`conv;64 30 14];
    .t.eq[.qs.inputSize m;128];
    .t.eq[count each m`ckernel;3 2];
    .t.eq[count each first each m`ckernel;10 9];
    .t.eq[count first first m`weight;28];
    s:.qs.summary m;
    .t.eq[s`kind;`conv`conv`dense`dense];
    .t.eq[s`params;33 20 116 5];
    .t.throws[.qs.check;@[m;`ckernel;:;reverse m`ckernel]];
    .t.throws[.qs.check;@[m;`weight;:;(4 10#0f;1 4#0f)]];
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
    s:.qs.grad1[m;X;Y;`bce;(count m`weight)#1f];
    {[r;s;k] .t.near[r k;s k;1e-12]}[r;s] each .qs.params m;
 }];

.t.add[`conv_save_load_train;{
    m:.qs.initConv[1;16;([] filters:enlist 2; kernel:enlist 3; pool:enlist 2; act:enlist `relu);enlist 1;`sigmoid;1];
    f:.t.tmpFile "qsolve_conv_model"; .qs.save[f;m]; .t.eq[.qs.load f;m]; hdel f;
    / adam and l2 update the conv parameters too
    system "S 2"; X:{x?1f} each 32#16; Y:enlist each "f"$32?0 1;
    r:.qs.train[m;`X`Y!(X;Y);`epochs`optimiser`lr`l2`verbose!(3;`adam;0.01;0.01;0)];
    .t.ok[not r[`model;`ckernel]~m`ckernel;"conv kernels didn't move"];
    .t.ok[not r[`model;`cbias]~m`cbias;"conv biases didn't move"];
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
