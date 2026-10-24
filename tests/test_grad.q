/ T1: analytic gradient vs central finite differences, for every activation and loss

/ numeric gradient of loss for every weight and bias
.t.fdGrad:{[m;X;Y;loss;h]
    L:{[X;Y;loss;m] .qs.loss.v[loss][.qs.predict[m;X];Y]}[X;Y;loss];
    fd:{[L;m;h;k;i] (L[@[m;k;.[;i;+;h]]]-L[@[m;k;.[;i;-;h]]])%2*h}[L;m;h];
    ixW:raze {[l;w] raze l,/:'til[count w],/:\:til count first w}'[til count m`weight;m`weight];
    ixB:raze {[l;b] l,/:til count b}'[til count m`bias;m`bias];
    gW:{[m;ix;v] {.[x;y 0;:;y 1]}/[0f*m`weight;flip (ix;v)]}[m;ixW;fd[`weight] each ixW];
    gB:{[m;ix;v] {.[x;y 0;:;y 1]}/[0f*m`bias;flip (ix;v)]}[m;ixB;fd[`bias] each ixB];
    `weight`bias!(gW;gB)
 }

/ max abs difference between .qs.grad and finite differences
.t.gradErr:{[nIn;sizes;acts;loss]
    m:.qs.init[nIn;sizes;acts;0N];
    m[`bias]:{-0.5+(count x)?1f} each m`bias;
    X:{x?1f} each 4#nIn;
    k:last sizes;
    Y:$[loss=`xent; {x=til y}[;k] each 4?k; {x?1f} each 4#k];
    Y:"f"$Y;
    g:.qs.grad[m;X;Y;loss]; n:.t.fdGrad[m;X;Y;loss;1e-6];
    max (max abs (raze/) g[`weight]-n`weight;max abs (raze/) g[`bias]-n`bias)
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

.t.add[`grad_shapes;{
    m:.qs.init[3;4 2;`tanh`sigmoid;1];
    g:.qs.grad[m;(1 2 3f;4 5 6f);(0 1f;1 0f);`mse];
    .t.eq[count each g`weight;count each m`weight];
    .t.eq[count each first each g`weight;count each first each m`weight];
    .t.eq[count each g`bias;count each m`bias];
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
    .t.near[r 1;.qs.grad1[m;X;Y;`xent;2#1f];1e-15];
    .t.near[r 2;last last .qs.forward[m;X];1e-15];
    .t.eq[.t.withThreads[0;{count .qs.shards 100}];1];
    .t.eq[.t.withThreads[16;{count .qs.shards 40}];1];
 }];

/ with fixed dropout masks the gradient still matches finite differences of the masked network
.t.add[`grad_dropout;{
    system "S 3";
    m:.qs.init[3;5 4 2;`tanh`relu`softmax;0N]; X:{x?1f} each 6#3; Y:.qs.data.oneHot[2;6?2];
    M:.qs.dropMasks[m;6;0.4];
    .t.eq[count M;3];
    .t.eq[last M;1f];
    .t.ok[all raze[M 0] in 0f,1%0.6;"mask values should be 0 or 1/(1-p)"];
    L:{[X;Y;M;m] .qs.loss.v[`xent][last .qs.forwardMasked[m;X;M] 2;Y]}[X;Y;M];
    h:1e-6;
    fd:{[L;m;h;k;i] (L[@[m;k;.[;i;+;h]]]-L[@[m;k;.[;i;-;h]]])%2*h}[L;m;h];
    g:.qs.grad1[m;X;Y;`xent;M];
    ixW:raze {[l;w] raze l,/:'til[count w],/:\:til count first w}'[til count m`weight;m`weight];
    ixB:raze {[l;b] l,/:til count b}'[til count m`bias;m`bias];
    .t.near[{x . y}[g`weight] each ixW;fd[`weight] each ixW;1e-6];
    .t.near[{x . y}[g`bias] each ixB;fd[`bias] each ixB;1e-6];
 }];

/ dropout masks are drawn before sharding, so a seeded gradient doesn't depend on the thread count
.t.add[`grad_dropout_sharded;{
    m:.qs.init[3;8 2;`relu`softmax;1]; system "S 5";
    X:{x?1f} each 100#3; Y:.qs.data.oneHot[2;100?2];
    g:{[m;X;Y;n] .t.withThreads[n;{[m;X;Y;u] system "S 9"; .qs.gradDrop[m;X;Y;`xent;0.5]}[m;X;Y]]}[m;X;Y];
    .t.near[g 4;g 0;1e-15];
    .t.ok[not (g 0)~.qs.grad[m;X;Y;`xent];"dropout had no effect"];
 }];
