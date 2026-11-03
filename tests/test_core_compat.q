/ M1.4: models in the layout from before the layer registry (tests/fixtures/old_format_cases, written by
/ tests/fixtures/make_old_format.q with the old library) are converted on load and give the same predictions, logits,
/ gradients and trained models as the old library did

.t.oldCases:{[] get .t.path "tests/fixtures/old_format_cases"}
.t.oldCase:{[name] c:.t.oldCases[]; first c where (c`name)=name}

/ a new-layout gradient (one list of arrays per layer) in the old layout's terms, to compare with the old numbers:
/ weight and bias for the dense layers, ckernel and cbias for conv layers, gW, gU and gb for a GRU
.t.oldGrad:{[m;g]
    K:.qs.kinds m; d:g where K=`dense; c:g where K=`conv1d; r:g where K=`gru;
    d1:`weight`bias!(d[;0];d[;1]);
    c1:$[count c; `ckernel`cbias!(c[;0];c[;1]); ()!()];
    r1:$[count r; `gW`gU`gb!(r[0;0];r[0;1];r[0;2]); ()!()];
    d1,c1,r1
 }

.t.add[`compat_old_models_convert;{
    cs:.t.oldCases[];
    {[c]
        m:.qs.check c`model;
        .t.ok[`layers`params in key m; string[c`name]," didn't convert"];
        / the same network: parameter counts agree with the old model's
        o:c`model;
        np:(sum count each raze each o`weight)+(sum count each o`bias)
            +$[`ckernel in key o; (sum count each raze each o`ckernel)+sum count each o`cbias; 0]
            +$[`gW in key o; sum (count raze o`gW;count raze o`gU;count o`gb); 0];
        .t.eq[sum .qs.summary[m]`params;np];
        / other keys survive the conversion
        .t.eq[`layers`params _ m;(`weight`bias`act`conv`ckernel`cbias`gru`gW`gU`gb inter key o) _ o];
    } each cs;
 }];

.t.add[`compat_old_predictions;{
    {[c]
        .t.near[.qs.predict[c`model;c`X];c`pred;1e-12];
        .t.near[.qs.predictRaw[c`model;c`X];c`raw;1e-12];
        / a single example works like a one-row batch
        .t.near[.qs.predict[c`model;first c`X];first c`pred;1e-12];
    } each .t.oldCases[];
 }];

.t.add[`compat_old_gradients;{
    {[c]
        m:.qs.check c`model;
        g:.t.oldGrad[m;.qs.grad[m;c`X;c`Y;c`loss]];
        o:c`grad;
        ks:key o;
        .t.ok[all ks in key g;string[c`name]," gradient is missing ",-3!ks except key g];
        {[g;o;k] .t.near[g k;o k;1e-12]}[g;o] each ks;
    } each .t.oldCases[];
 }];

/ whole training runs (adam, momentum, l2, dropout; conv and GRU front ends) end where the old library's did
.t.add[`compat_old_training;{
    {[c]
        system "S 1";
        o:(`loss`verbose!(c`loss;0)),c`opts;
        r:.qs.train[c`model;`X`Y!(c`X;c`Y);o];
        .t.near[r[`history]`trainLoss;c`trainLoss;1e-9];
        .t.eq[r`bestEpoch;c`bestEpoch];
        .t.near[.qs.predict[r`model;c`X];c`trainedPred;1e-9];
    } each .t.oldCases[];
 }];

/ a model file in the old layout loads as the new one and survives a save and load
.t.add[`compat_old_file_loads;{
    f:.t.tmpFile "qsolve_old_model";
    c:.t.oldCase`conv_pooled;
    f set c`model;
    m:.qs.load f;
    .t.ok[`layers in key m;"load didn't convert"];
    .t.near[.qs.predict[m;c`X];c`pred;1e-12];
    .qs.save[f;m]; .t.eq[.qs.load f;m];
    hdel f;
 }];

/ the real models, if their files are here, as converted: logits on random inputs against the old library's, kept in
/ tests/fixtures only for the small ones; the GW models are git-ignored, so this just checks they load and run
.t.add[`compat_real_models_run;{
    f:{[n] hsym `$.t.root,"/",n};
    fs:(f "projects/gw/models/gw_cnn_4.qsm";f "projects/gw/models/gw_gru_1.qsm";f "models/spirals_1.qsm");
    if[not all {not ()~key x} each fs; .t.skip "needs the models in projects/gw/models and models"];
    {[f] m:.qs.load f; X:{x?1f} each 5#.qs.inputSize m; p:.qs.predict[m;X]; .t.eq[count p;5]; .t.ok[all raze p within 0 1f;"outputs"]} each fs;
 }];
