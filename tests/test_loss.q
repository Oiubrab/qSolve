/ loss values and gradients

.t.add[`loss_values;{
    A:(0.2 0.8;0.6 0.4); Y:(0 1f;1 0f);
    .t.near[.qs.loss.v[`mse][A;Y];0.5*(0.04+0.04+0.16+0.16)%2;1e-12];
    .t.near[.qs.loss.v[`xent][A;Y];neg (log[0.8]+log 0.6)%2;1e-12];
    .t.near[.qs.loss.v[`bce][A;Y];neg (log[0.8]+log[0.8]+log[0.6]+log 0.6)%2;1e-12];
    .t.near[.qs.loss.v[`norm][A;Y];sqrt 0.4;1e-12];
 }];

/ log losses stay finite when predictions hit 0 or 1
.t.add[`loss_clipped;{
    A:(0 1f;1 0f); Y:(1 0f;0 1f);
    {.t.ok[not null .qs.loss.v[x][y;z];"null loss from ",string x]}[;A;Y] each `bce`xent;
 }];

/ dL/dA matches central differences
.t.add[`loss_gradients;{
    A:(0.2 0.7 0.4;0.6 0.3 0.9); Y:(0.1 1 0;1 0 0.5); h:1e-6;
    {[A;Y;h;l]
        ix:raze til[count A],/:\:til count first A;
        fd:{[A;Y;h;l;i] (.qs.loss.v[l][.[A;i;+;h];Y]-.qs.loss.v[l][.[A;i;-;h];Y])%2*h}[A;Y;h;l] each ix;
        .t.near[(raze .qs.loss.g[l][A;Y]);fd;1e-7]
    }[A;Y;h] each key .qs.loss.g;
 }];
