/ activation values and derivatives

.t.add[`act_values;{
    .t.near[.qs.act.f[`sigmoid] 0f;0.5;1e-15];
    .t.near[.qs.act.f[`tanh] 0 0.5 -2f;0 0.46211715726000974 -0.9640275800758169;1e-12];
    .t.eq[.qs.act.f[`relu] -1 0 2f;0 0 2f];
    .t.eq[.qs.act.f[`linear] -1 0 2f;-1 0 2f];
 }];

.t.add[`act_softmax_rows;{
    s:.qs.act.f[`softmax] (1 2 3f;0 0 0f);
    .t.near[sum each s;1 1f;1e-12];
    .t.near[s 1;3#1%3;1e-12];
    .t.ok[(</) s[0;0 1];"softmax should preserve order"];
 }];

/ large inputs must not overflow into NaN
.t.add[`act_stable;{
    zs:-800 -50 0 50 800f;
    {[zs;f] .t.ok[not any null .qs.act.f[f] zs;"null from ",string f]}[zs] each `sigmoid`tanh`relu`linear;
    {[zs;f] .t.ok[not any null .qs.act.d[f][zs;.qs.act.f[f] zs];"null derivative from ",string f]}[zs] each `sigmoid`tanh`relu`linear;
    .t.ok[not any null raze .qs.act.f[`softmax] enlist 1000 0 -1000f;"softmax overflow"];
 }];

/ derivative matches central differences (away from relu's kink)
.t.add[`act_derivatives;{
    zs:-2.5 -0.7 0.3 1.9f; h:1e-6;
    {[z;h;f] fd:(.qs.act.f[f][z+h]-.qs.act.f[f][z-h])%2*h; .t.near[.qs.act.d[f][z;.qs.act.f[f] z];fd;1e-8]}[zs;h] each key .qs.act.d;
 }];
