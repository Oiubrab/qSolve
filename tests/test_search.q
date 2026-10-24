/ hyperparameter search

.t.add[`search_grid;{
    system "S 1"; d:.qs.data.split[.qs.data.moons[300;0.1];0.8];
    r:.qs.search[d;`hidden`lr!((enlist 4;8 8);0.01 0.001);`optimiser`epochs!(`adam;10);0N];
    .t.eq[count r;4];
    .t.eq[cols r;`hidden`lr`valLoss`valAcc`valR2`bestEpoch`seconds];
    .t.eq[r`valLoss;asc r`valLoss];
    .t.eq[asc distinct r`lr;0.001 0.01];
    .t.ok[(enlist 8 8) in r`hidden;"two-layer configuration missing"];
    .t.ok[all null r`valR2;"R squared for classification"];
 }];

.t.add[`search_random;{
    system "S 1"; d:.qs.data.split[.qs.data.moons[300;0.1];0.8];
    r:.qs.search[d;`lr`batch!(0.1 0.03 0.01 0.003;16 32 64);`optimiser`epochs!(`adam;3);5];
    .t.eq[count r;5];
    .t.eq[count distinct flip `lr`batch#flip r;5];
    .t.eq[count .qs.search[d;enlist[`lr]!enlist 0.1 0.01;enlist[`epochs]!enlist 2;10];2];
 }];

/ the score is the model the search trained: rerunning the best row's configuration reproduces it
.t.add[`search_reproducible;{
    system "S 1"; d:.qs.data.split[.qs.data.moons[300;0.1];0.8];
    b:first .qs.search[d;`hidden`lr!((enlist 4;enlist 8);0.01 0.003);`optimiser`epochs!(`adam;10);0N];
    m:.qs.init[2;b[`hidden],2;`tanh`softmax;42];
    r:.qs.train[m;d;`optimiser`epochs`lr`loss`verbose!(`adam;10;b`lr;`xent;0)];
    .t.near[.qs.evaluate[r`model;d`Xval;d`Yval;`xent]`loss;b`valLoss;1e-12];
 }];

.t.add[`search_regression;{
    system "S 1"; d:.qs.data.fromTable[.qs.data.houses 300;();`price;0.8];
    r:.qs.search[d;enlist[`lr]!enlist 0.01 0.001;`optimiser`epochs!(`adam;5);0N];
    .t.ok[all null r`valAcc;"accuracy for regression"];
    .t.ok[not any null r`valR2;"missing R squared"];
    .t.throws[{.qs.search[`X`Y#x;enlist[`lr]!enlist 0.1;()!();0N]};d];
 }];

.t.add[`cross_validate;{
    system "S 1"; d:.qs.data.moons[200;0.1];
    r:.qs.crossValidate[d;4;`optimiser`epochs`lr!(`adam;20;0.01)];
    f:r`folds;
    .t.eq[f`fold;til 4];
    .t.eq[cols f;`fold`valLoss`valAcc`valR2`bestEpoch`seconds];
    .t.eq[r[`summary]`metric;`valLoss`valAcc`valR2];
    .t.near[r[`summary;0;`mean];avg f`valLoss;1e-12];
    .t.ok[0.85<r[`summary;1;`mean];"moons cross-validated accuracy too low"];
    / folds partition the rows: 7 rows in 3 folds gives 3, 3 and 1
    s:.qs.data.split[.qs.data.moons[7;0.1];0.5];
    .t.eq[count .qs.crossValidate[s;3;enlist[`epochs]!enlist 1]`folds;3];
    .t.throws[.qs.crossValidate[d;;()!()];1];
    .t.throws[.qs.crossValidate[d;;()!()];1000];
 }];
