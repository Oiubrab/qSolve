/ T1: analytic gradient vs central finite differences, for every activation and loss

/ largest difference between a gradient g (one list of arrays per layer, as .qs.grad gives) and central finite
/ differences of the function L of the model, over every parameter of m
.t.gradDiffL:{[m;L;g]
    h:1e-6;
    e:{[L;h;m;g;ix] pp:(`params),ix; ((L[.[m;pp;+;h]]-L[.[m;pp;-;h]])%2*h)-g . ix}[L;h;m;g] each .qs.paths m`params;
    max abs e
 }

/ the same for the loss of the model's predictions on (X;Y)
.t.gradDiff:{[m;X;Y;loss;g]
    .t.gradDiffL[m;{[X;Y;loss;m] .qs.loss.v[loss][.qs.predict[m;X];Y]}[X;Y;loss];g]
 }

/ random biases (every parameter named b), so the checks don't sit at a symmetric point
.t.randomBiases:{[m]
    K:.qs.kinds m;
    ix:raze {[K;i] i,/:where `b=.qs.layer.names[K i]}[K] each til count K;
    {[m;pos] .[m;(`params),pos;:;-0.5+(count m . (`params),pos)?1f]}/[m;ix]
 }

/ max abs difference between .qs.grad and finite differences
.t.gradErr:{[nIn;sizes;acts;loss]
    m:.t.randomBiases .qs.init[nIn;sizes;acts;0N];
    X:{x?1f} each 4#nIn;
    k:last sizes;
    Y:$[loss=`xent; {x=til y}[;k] each 4?k; {x?1f} each 4#k];
    Y:"f"$Y;
    .t.gradDiff[m;X;Y;loss;.qs.grad[m;X;Y;loss]]
 }

/ every hidden activation with every compatible output activation and loss, at three depths
.t.gradOut:((`linear;`mae);(`sigmoid;`mse);(`sigmoid;`bce);(`linear;`mse);(`tanh;`norm);(`softmax;`xent);(`sigmoid;`xent);(`relu;`mse));
.t.gradCases:raze raze (4 2;4 3 2;5 4 3 2) {[s;h] {[s;h;o] (s;h;o)}[s;h] each .t.gradOut}/:\: `sigmoid`tanh`relu`linear;

.t.add[`grad_all_combinations;{
    system "S 11";
    bad:{[c]
        sizes:c 0; h:c 1; o:c 2;
        acts:((-1+count sizes)#h),o 0;
        e:.t.gradErr[3;sizes;acts;o 1];
        $[1e-6>e; (); enlist (sizes;acts;o 1;e)]
    } each .t.gradCases;
    bad:raze bad;
    .t.ok[0=count bad;"mismatches: ",-3!bad];
 }];

/ a gradient is one list of arrays per layer, shaped like the parameters
.t.add[`grad_shapes;{
    m:.qs.init[3;4 2;`tanh`sigmoid;1];
    g:.qs.grad[m;(1 2 3f;4 5 6f);(0 1f;1 0f);`mse];
    .t.eq[count g;count m`layers];
    .t.eq[count each g;count each m`params];
    .t.eq[count each raze g;count each raze m`params];
    .t.eq[count each first each raze g;count each first each raze m`params];
 }];

.t.add[`grad_softmax_needs_xent;{
    m:.qs.init[3;4 2;`tanh`softmax;1];
    .t.throws[{.qs.grad[x;enlist 1 2 3f;enlist 0 1f;`mse]};m];
 }];

/ run f[] with .qs.threads pretending there are n secondary threads
.t.withThreads:{[n;f] t:.qs.threads; .qs.threads:{[n;x] n}[n]; r:@[f;::;{(`err;x)}]; .qs.threads:t; if[`err~first r; 'r 1]; r}

/ sharding over threads gives the same gradient as one pass (peach runs serially when threads are off)
.t.add[`grad_sharded;{
    m:.qs.init[3;4 2;`tanh`softmax;1]; system "S 5";
    X:{x?1f} each 100#3; Y:.qs.data.oneHot[2;100?2];
    r:.t.withThreads[4;{[m;X;Y;u] (count .qs.shards 100;.qs.grad[m;X;Y;`xent];.qs.predict[m;X])}[m;X;Y]];
    .t.eq[r 0;3];
    .t.near[r 1;.qs.grad1[m;X;Y;`xent;(count m`layers)#1b];1e-15];
    .t.near[r 2;.qs.infer[m;X;count m`layers];1e-15];
    .t.eq[.t.withThreads[0;{count .qs.shards 100}];1];
    .t.eq[.t.withThreads[16;{count .qs.shards 40}];1];
 }];

/ the dropout option adds a dropout layer after each hidden dense layer's activation, never after the output
.t.add[`dropout_layers_inserted;{
    m:.qs.init[3;5 4 2;`tanh`relu`softmax;1];
    d:.qs.withDropout[m;0.4];
    .t.eq[.qs.kinds d 0;`dense`act`dropout`dense`act`dropout`dense`act];
    .t.eq[d 1;11011011b];
    .t.eq[(.qs.withDropout[m;0f]) 0;m];
    .t.eq[count (d 0)`params;8];
    .t.eq[(d[0]`params) 2;()];
 }];

/ with fixed dropout masks the gradient still matches finite differences of the masked network
.t.add[`grad_dropout;{
    system "S 3";
    m:.qs.init[3;5 4 2;`tanh`relu`softmax;0N]; X:{x?1f} each 6#3; Y:.qs.data.oneHot[2;6?2];
    m2:(.qs.withDropout[m;0.4]) 0;
    t:.qs.aux[m2;6];
    .t.eq[count t;count m2`layers];
    .t.eq[t 0;1b];
    .t.ok[all raze[t 2] in 0f,1%0.6;"mask values should be 0 or 1/(1-p)"];
    .t.eq[(count t 2;count first t 2);6 5];
    .t.eq[(count t 5;count first t 5);6 4];
    L:{[X;Y;t;m] .qs.loss.v[`xent][first .qs.forwardAll[m;X;t];Y]}[X;Y;t];
    g:.qs.grad1[m2;X;Y;`xent;t];
    .t.ok[1e-6>.t.gradDiffL[m2;L;g];"dropout gradient doesn't match finite differences"];
 }];

/ dropout masks are drawn before sharding, so a seeded gradient doesn't depend on the thread count
.t.add[`grad_dropout_sharded;{
    m:.qs.init[3;8 2;`relu`softmax;1]; system "S 5";
    X:{x?1f} each 100#3; Y:.qs.data.oneHot[2;100?2];
    g:{[m;X;Y;n] .t.withThreads[n;{[m;X;Y;u] system "S 9"; .qs.gradDrop[m;X;Y;`xent;0.5]}[m;X;Y]]}[m;X;Y];
    .t.near[g 4;g 0;1e-15];
    .t.ok[not (g 0)~.qs.grad[m;X;Y;`xent];"dropout had no effect"];
 }];

/ a dropout layer in the model itself (not the training option) works the same way
.t.add[`dropout_layer_in_model;{
    system "S 4";
    m:.qs.assemble[enlist 3;((`dense;`n`scheme!(6;`xavier));(`act;enlist[`f]!enlist `tanh);(`dropout;enlist[`rate]!enlist 0.5);(`dense;`n`scheme!(2;`xavier));(`act;enlist[`f]!enlist `softmax));();7];
    X:{x?1f} each 5#3;
    / not training: the layer passes values through, so predictions are repeatable
    .t.eq[.qs.predict[m;X];.qs.predict[m;X]];
    k:.qs.kinds m;
    .t.eq[k;`dense`act`dropout`dense`act];
    / training: each call draws fresh masks
    Y:.qs.data.oneHot[2;5?2];
    .t.ok[not (.qs.grad[m;X;Y;`xent])~.qs.grad[m;X;Y;`xent];"the dropout layer wasn't applied in training"];
    / and the training option adds none on top of it for the output side
    .t.eq[count .qs.kinds (.qs.withDropout[m;0f]) 0;5];
 }];
