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
.t.gradOut:((`sigmoid;`mse);(`sigmoid;`bce);(`linear;`mse);(`tanh;`norm);(`softmax;`xent);(`sigmoid;`xent);(`relu;`mse));
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
