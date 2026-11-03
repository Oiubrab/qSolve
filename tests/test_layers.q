/ M1.1 and M1.3: the layer registry, and a finite-difference gradient check of every registered layer type

/ every registered type has every part
.t.add[`layers_registry_complete;{
    parts:`names`shape`init`forward`backward`check`l2`inGrad`example;
    ks:key .qs.layer.forward;
    .t.ok[all `dense`act`dropout`seq`conv1d`maxpool1d`flatten`gru in ks;"missing a ported layer type"];
    bad:raze {[ks;p] ks where not ks in key get `$".qs.layer.",string p}[ks] each parts;
    .t.ok[0=count bad;"layer types missing a part: ",-3!bad];
 }];

/ backward matches central finite differences for every parameter and for the input, for the examples each type
/ registers; a new layer type is checked as soon as it registers its examples
.t.add[`layers_gradient_check;{
    system "S 21";
    r:raze {[k] {[k;e] (k;.qs.gradCheck[k;e 0;e 1])}[k] each .qs.layer.example k} each key .qs.layer.forward;
    bad:r where 1e-6<r[;1];
    .t.ok[0=count bad;"layer gradient errors: ",-3!bad];
    .t.ok[(count distinct r[;0])=count .qs.layer.forward;"every layer type should be checked"];
 }];

/ the check has teeth: a layer whose backward is 10% off fails it
.t.add[`layers_gradient_check_detects_errors;{
    k:`dense; ex:first .qs.layer.example k;
    good:.qs.layer.backward k;
    .qs.layer.backward[k]:{[s;p;c;dZ;needIn] r:.qs.dense.backward[s;p;c;dZ;needIn]; (r 0;(key r 1)!1.1*value r 1)};
    e:@[.qs.gradCheck[k;ex 0];ex 1;{(`err;x)}];
    .qs.layer.backward[k]:good;
    if[`err~first e; 'e 1];
    .t.ok[e>1e-3;"a 10% wrong gradient passed the check: ",string e];
 }];

/ a layer type can be added from outside the library with .qs.layer.add, and the model machinery then runs it
.t.add[`layers_register_new_type;{
    / scale: multiplies by a fixed number, with one learnable gain
    .qs.layer.add[`gain;`names`shape`init`forward`backward`check`l2`inGrad`example!(
        enlist `g;{[s;inS] inS};{[s;inS] enlist[`g]!enlist enlist 1.5};
        {[s;p;A;t] (A*first p`g;A)};{[s;p;c;dO;needIn] ($[needIn;dO*first p`g;()];enlist[`g]!enlist enlist sum raze dO*c)};
        {[s;p] };`symbol$();1b;enlist (()!();enlist 4))];
    r:@[{
        m:.qs.assemble[enlist 3;((`dense;`n`scheme!(4;`xavier));(`gain;()!());(`dense;`n`scheme!(1;`xavier));(`act;enlist[`f]!enlist `sigmoid));();1];
        .t.eq[.qs.kinds m;`dense`gain`dense`act];
        X:{x?1f} each 6#3; Y:enlist each "f"$6?0 1;
        .t.ok[1e-6>.t.gradDiff[m;X;Y;`bce;.qs.grad[m;X;Y;`bce]];"gain layer gradient"];
        e:.qs.gradCheck[`gain;()!();enlist 4];
        .t.ok[e<1e-6;"gain gradCheck ",string e];
        `ok};::;{(`err;x)}];
    {n:`$".qs.layer.",string x; n set enlist[`gain] _ get n} each `names`shape`init`forward`backward`check`l2`inGrad`example;
    if[`err~first r; 'r 1];
 }];

/ hand-computed values

/ a 1-channel convolution with kernel (1;-1) is the difference of neighbours
.t.add[`layer_conv1d_known_values;{
    s:`filters`kernel`inShape`outShape!(1;2;1 4;1 3);
    p:`K`b!(enlist 1 -1f;enlist 0.5);
    o:.qs.conv1d.forward[s;p;enlist 1 2 4 8f;0b];
    .t.eq[first o;enlist -0.5 -1.5 -3.5];
    / two examples are kept apart: no tap straddles the boundary
    o2:.qs.conv1d.forward[s;p;enlist 1 2 4 8 10 10 10 10f;0b];
    .t.eq[first o2;enlist -0.5 -1.5 -3.5 0.5 0.5 0.5];
 }];

/ max pooling takes the largest of each block, routes the gradient to the first of a tie, and drops a remainder
.t.add[`layer_maxpool_values;{
    s:`pool`inShape`outShape!(2;1 4;1 2);
    A:enlist 1 3 2 2 5 4 7 7f;
    o:.qs.maxpool1d.forward[s;()!();A;1b];
    .t.eq[first o;enlist 3 2 5 7f];
    / not training: the same output, without the winners backward would need
    .t.eq[.qs.maxpool1d.forward[s;()!();A;0b];(first o;())];
    b:.qs.maxpool1d.backward[s;()!();o 1;enlist 1 1 1 1f;1b];
    .t.eq[first b;enlist 0 1 1 0 1 0 1 0f];
    s3:`pool`inShape`outShape!(3;1 5;1 1);
    .t.eq[first first .qs.maxpool1d.forward[s3;()!();enlist 1 9 2 100 100f;0b];enlist 9f];
 }];

/ flatten lists each step's channels together, and its backward undoes it
.t.add[`layer_flatten_order;{
    s:`inShape`outShape!(2 3;enlist 6);
    A:(1 2 3f;10 20 30f);
    o:.qs.flatten.forward[s;()!();A;0b];
    .t.eq[first o;enlist 1 10 2 20 3 30f];
    b:.qs.flatten.backward[s;()!();o 1;first o;1b];
    .t.eq[first b;A];
 }];

/ seq turns rows (channels one after another) into one vector per channel holding every example in turn
.t.add[`layer_seq_layout;{
    s:`channels`length`inShape`outShape!(2;3;enlist 6;2 3);
    X:(1 2 3 10 20 30f;4 5 6 40 50 60f);
    o:.qs.seq.forward[s;()!();X;0b];
    .t.eq[first o;(1 2 3 4 5 6f;10 20 30 40 50 60f)];
    .t.eq[first .qs.seq.backward[s;()!();();first o;1b];X];
 }];

/ dropout passes values through unless training, scales kept values by 1/(1-rate), and masks its gradient
.t.add[`layer_dropout_values;{
    s:`rate`inShape`outShape!(0.5;enlist 3;enlist 3);
    A:(1 2 3f;4 5 6f); mask:(2 0 2f;0 2 2f);
    .t.eq[first .qs.dropout.forward[s;()!();A;0b];A];
    o:.qs.dropout.forward[s;()!();A;mask];
    .t.eq[first o;(2 0 6f;0 10 12f)];
    .t.eq[first .qs.dropout.backward[s;()!();o 1;(1 1 1f;1 1 1f);1b];mask];
    / the mask keeps each value with probability 1-rate, scaled to keep the mean
    system "S 3"; mk:.qs.dropMask[4000;0.25;5];
    .t.ok[all raze[mk] in 0f,1%0.75;"mask values"];
    .t.near[avg raze mk;1f;0.05];
 }];

/ activations as layers equal the activation functions, on a matrix and on a channel-major sequence
.t.add[`layer_act_values;{
    s:`f`inShape`outShape!(`relu;enlist 3;enlist 3);
    A:(-1 0.5 2f;3 -4 0f);
    o:.qs.actl.forward[s;()!();A;0b];
    .t.eq[first o;(0 0.5 2f;3 0 0f)];
    .t.eq[first .qs.actl.backward[s;()!();o 1;(1 1 1f;1 1 1f);1b];(0 1 1f;1 0 0f)];
    / softmax has no layer-level derivative: the loss pairs with it
    s2:`f`inShape`outShape!(`softmax;enlist 3;enlist 3);
    .t.throws[{[s;a] .qs.actl.backward[s;()!();(a;a);a;1b]}[s2];A];
 }];
