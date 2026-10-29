/ GRU front end (G5.3)

/ backpropagation through time against central differences, for every parameter, dense ones included
.t.gruCase:{[block;hidden;sizes;acts;loss;seed]
    system "S ",string seed;
    m:.qs.initGru[2;12;block;hidden;sizes;acts;0N];
    m[`gb]:-0.3+(count m`gb)?0.6;
    X:{x?1f} each 3#24;
    k:last sizes;
    Y:"f"$$[loss=`xent; {x=til y}[;k] each 3?k; {x?1f} each 3#k];
    g:.qs.grad[m;X;Y;loss]; n:.t.fdAll[m;X;Y;loss;1e-6];
    {[g;n;k] max abs (raze/) g[k]-n k}[g;n] each .qs.params m
 }

.t.add[`gru_gradients;{
    e1:.t.gruCase[3;4;3 1;`tanh`sigmoid;`bce;1];
    .t.ok[all e1<1e-6;"GRU into tanh/sigmoid gradient error ",-3!e1];
    e2:.t.gruCase[2;3;enlist 3;`softmax;`xent;2];
    .t.ok[all e2<1e-6;"GRU into softmax gradient error ",-3!e2];
    e3:.t.gruCase[6;5;2 1;`relu`linear;`mse;3];
    .t.ok[all e3<1e-6;"GRU into relu/linear gradient error ",-3!e3];
 }];

.t.add[`gru_shapes_and_checks;{
    m:.qs.initGru[2;64;8;6;4 1;`tanh`sigmoid;1];
    .t.eq[.qs.inputSize m;128];
    .t.eq[(count m`gW;count first m`gW);18 16];
    .t.eq[(count m`gU;count first m`gU);18 6];
    .t.eq[count m`gb;18];
    / the steps read block samples of each channel in turn
    X:enlist "f"$til 128;
    st:.qs.gru.steps[X;m`gru];
    .t.eq[count st;8];
    .t.eq[first first st;"f"$(til 8),64+til 8];
    .t.eq[first last st;"f"$(56+til 8),120+til 8];
    s:.qs.summary m;
    .t.eq[s`kind;`gru`dense`dense];
    / 18x16 + 18x6 + 18 GRU parameters, then 6x4+4 and 4+1
    .t.eq[s`params;414 28 5];
    .t.throws[.qs.check;@[m;`gU;:;18 5#0f]];
    .t.throws[.qs.check;@[m;`weight;:;(4 7#0f;1 4#0f)]];
    .t.throws[{.qs.initGru[2;64;7;6;4 1;`tanh`sigmoid;1]};::];
    x:128?1f;
    .t.near[.qs.predict[m;x];first .qs.predict[m;enlist x];1e-12];
    / saved and loaded intact
    f:hsym `$(getenv[`TMPDIR],$[""~getenv`TMPDIR;"/tmp";""]),"/qsolve_gru_",(string .z.i),".qsm";
    .qs.save[f;m];
    .t.eq[.qs.load f;m];
    hdel f;
 }];

/ a task needing memory: is the first half of a sequence louder than the second? (the order matters, so a model
/ that forgets earlier steps can't do better than chance)
.t.add[`gru_learns_order;{
    system "S 11";
    mk:{[n] a:n?1f; b:n?1f; X:{[a;b] (8?a),8?b}'[a;b]; `X`Y!(X;enlist each "f"$a>b)};
    d:mk 600; v:mk 300;
    m:.qs.initGru[1;16;2;8;enlist 1;`sigmoid;5];
    r:.qs.train[m;`X`Y`Xval`Yval!(d`X;d`Y;v`X;v`Y);`epochs`batch`lr`loss`optimiser`verbose!(40;32;0.01;`bce;`adam;0)];
    acc:.qs.evaluate[r`model;v`X;v`Y;`bce]`acc;
    .t.ok[acc>0.85;"GRU accuracy ",string acc];
 }];
