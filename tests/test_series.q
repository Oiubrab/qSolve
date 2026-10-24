/ time-series windows

.t.add[`series_windows;{
    s:10 11 13 16 20 25 31 38 46 55f;
    d:.qs.data.windows[s;3;1;0.5];
    / times 2..8, the first 3 train
    .t.eq[count each d`X`Y`Xval`Yval;3 3 4 4];
    .t.eq[d`level;13 16 20f];
    .t.eq[d`levelVal;25 31 38 46f];
    / targets put back in the series' units are the values one step after each window
    .t.near[.qs.data.unwindow[d;d`Y;d`level];16 20 25f;1e-12];
    .t.near[.qs.data.unwindow[d;d`Yval;d`levelVal];31 38 46 55f;1e-12];
    / target scaling comes from the training rows only
    .t.near[d`mu;avg 3 4 5f;1e-12];
    .t.near[avg raze d`X;0f;1e-12];
    .t.eq[count first d`X;2];
    .t.throws[{.qs.data.windows[1 2 3f;3;1;0.5]};::];
 }];

.t.add[`series_horizon;{
    s:"f"$til 20;
    d:.qs.data.windows[s;4;3;0.5];
    .t.near[.qs.data.unwindow[d;d`Y;d`level];3+d`level;1e-12];
 }];

.t.add[`series_naive;{
    s:"f"$20#0 1 2 3;
    n:.qs.data.naive[s;8;1;0.5;4];
    .t.near[n`seasonal;0f;1e-12];
    .t.ok[n[`last]>0;"last-value forecast should miss a cycle"];
    .t.ok[null .qs.data.naive[s;3;1;0.5;8]`seasonal;"period longer than the window"];
 }];

/ a model should beat both naive forecasts on a noisy seasonal series with a trend
.t.add[`series_beats_naive;{
    system "S 1"; s:.qs.data.seasonal[1500;24;0.1];
    d:.qs.data.windows[s;48;1;0.8];
    r:.qs.train[.qs.init[47;enlist 1;`linear;1];d;`loss`optimiser`lr`epochs`verbose!(`mse;`adam;0.003;60;0)];
    p:.qs.data.unwindow[d;.qs.predict[r`model;d`Xval];d`levelVal];
    mae:avg abs p-.qs.data.unwindow[d;d`Yval;d`levelVal];
    nb:.qs.data.naive[s;48;1;0.8;24];
    .t.ok[mae<0.8*nb`seasonal;"model MAE ",(string mae)," vs seasonal naive ",string nb`seasonal];
 }];

/ encoding new recent values matches the windows built in training
.t.add[`series_window_row;{
    system "S 2"; s:.qs.data.seasonal[300;24;0.1];
    d:.qs.data.windows[s;12;2;0.8];
    t:last d`times;
    w:.qs.data.windowRow[d;s til 1+t];
    .t.near[w`X;-1#d`Xval;1e-12];
    .t.eq[w`level;-1#d`levelVal];
    .t.near[.qs.data.unwindow[d;-1#d`Yval;w`level];enlist s t+2;1e-9];
    .t.throws[.qs.data.windowRow[d;];5#s];
 }];
