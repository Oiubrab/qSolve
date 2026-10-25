/ qSolve: a small multilayer perceptron in q
/ model:  `weight`bias`act!(W;B;acts)  W[l] is n_l x n_(l-1), B[l] has n_l entries, acts one symbol per layer
/ batches are matrices: one row per example

/ activations: forward functions, and derivatives written in terms of (z;a) where a:f z
.qs.act.f:`sigmoid`tanh`relu`linear`softmax!(
    {reciprocal 1+exp neg x};
    {1-2%1+exp 2*x};
    {0f|x};
    {x};
    {{e%sum e:exp x-max x} each x})
.qs.act.d:`sigmoid`tanh`relu`linear!(
    {[z;a] a*1-a};
    {[z;a] 1-a*a};
    {[z;a] "f"$z>0};
    {[z;a] 1f+0f*z})

/ losses averaged over the batch: values .qs.loss.v[A;Y] and gradients .qs.loss.g[A;Y] = dL/dA
/ mae (mean absolute error) has the sign of the error as its gradient
/ norm is the original qSolve loss: the euclidean length of every error in the batch
.qs.clip:{1e-12|x&1-1e-12}
.qs.loss.v:`mse`mae`bce`xent`norm!(
    {[A;Y] 0.5*(sum raze e*e:A-Y)%count Y};
    {[A;Y] (sum raze abs A-Y)%count Y};
    {[A;Y] A:.qs.clip A; neg (sum raze (Y*log A)+(1-Y)*log 1-A)%count Y};
    {[A;Y] neg (sum raze Y*log .qs.clip A)%count Y};
    {[A;Y] sqrt sum raze e*e:A-Y})
.qs.loss.g:`mse`mae`bce`xent`norm!(
    {[A;Y] (A-Y)%count Y};
    {[A;Y] (signum A-Y)%count Y};
    {[A;Y] A:.qs.clip A; ((A-Y)%A*1-A)%count Y};
    {[A;Y] neg (Y%.qs.clip A)%count Y};
    {[A;Y] e%sqrt sum raze e*e:A-Y})

/ output activation and loss pairs whose combined gradient simplifies to (A-Y)/n
.qs.paired:((`softmax;`xent);(`sigmoid;`bce))

/ signal unless the model's shapes and activations are consistent
.qs.check:{[m]
    if[not all `weight`bias`act in key m; '"model needs weight, bias and act"];
    if[not (count m`weight)=count m`bias; '"weight and bias have different numbers of layers"];
    if[not (count m`weight)=count m`act; '"need one activation per layer"];
    if[not all m[`act] in key .qs.act.f; '"unknown activation: ",-3!m[`act] except key .qs.act.f];
    if[`softmax in -1_m`act; '"softmax is only allowed on the output layer"];
    if[not (count each m`weight)~count each m`bias; '"bias length must match the number of nodes"];
    if[not (-1_count each m`weight)~1_count each first each m`weight; '"layer sizes don't chain"];
    m
 }

/ new model with nIn inputs and one layer per entry of sizes
/ acts is one activation per layer, or a single one for every layer; seed is 0N to leave the generator alone
/ weights are Xavier-uniform (He-uniform for relu), biases are zero
.qs.init:{[nIn;sizes;acts;seed]
    sizes:(),sizes;
    acts:$[1=count acts:(),acts; count[sizes]#acts; acts];
    if[not null seed; system "S ",string seed];
    fanIn:nIn,-1_sizes;
    lim:?[acts=`relu; sqrt 6%fanIn; sqrt 6%fanIn+sizes];
    W:{[o;i;l] (o;i)#l*-1+2*(o*i)?1f}'[sizes;fanIn;lim];
    .qs.check `weight`bias`act!(W;sizes#\:0f;acts)
 }

/ forward pass over a batch matrix X, returns (Z;A): pre-activations per layer and activations with A[0]=X
.qs.forward:{[m;X] f:.qs.forwardMasked[m;X;(count m`weight)#1f]; (f 0;f 2)}

/ forward pass with each layer's activations multiplied by a mask (a matrix, or 1f for none)
/ returns (Z;U;A): pre-activations, unmasked activations, and masked activations with A[0]=X
.qs.forwardMasked:{[m;X;M]
    {[s;w;b;f;k] z:(last[s 2] mmu flip w)+\:b; u:.qs.act.f[f] z; (s[0],enlist z;s[1],enlist u;s[2],enlist $[k~1f;u;u*k])}/[(();();enlist X);m`weight;m`bias;m`act;M]
 }

/ inverted dropout masks for n rows: each hidden node kept with probability 1-p and scaled by 1/(1-p), none on the output layer
.qs.dropMasks:{[m;n;p]
    $[p=0; (count m`weight)#1f; ({[n;p;k] (1%1-p)*p<(n;k)#(n*k)?1f}[n;p] each -1_count each m`bias),1f]
 }

/ row indices split into one shard per secondary thread, each at least .qs.minShard rows (one shard if threads are off)
.qs.minShard:32
.qs.threads:{system "s"}
.qs.shards:{[n] p:1|.qs.threads[]&n div .qs.minShard; ceiling[n%p] cut til n}

/ model outputs for a batch matrix X, or for a single example vector; large batches are spread over threads
.qs.predict:{[m;X]
    v:0h<type X; X:"f"$$[v;enlist X;X];
    out:{[m;X] last last .qs.forward[m;X]}[m];
    A:$[1<count i:.qs.shards count X; raze {[f;X;i] f X i}[out;X] peach i; out X];
    $[v;first A;A]
 }

/ gradient of loss over batch (X;Y), returns `weight`bias!(gW;gB) shaped like the model
.qs.grad:{[m;X;Y;loss] .qs.gradDrop[m;X;Y;loss;0f]}

/ gradient with dropout rate p on the hidden layers
/ large batches are spread over threads, each shard's mean gradient weighted by its share of the rows
/ (not for norm, whose gradient depends on the whole batch at once); masks are drawn first so results are seeded
.qs.gradDrop:{[m;X;Y;loss;p]
    M:.qs.dropMasks[m;count X;p];
    if[(loss=`norm) or 2>count i:.qs.shards count X; :.qs.grad1[m;X;Y;loss;M]];
    (+/){[m;X;Y;loss;M;i] (count[i]%count X)*.qs.grad1[m;X i;Y i;loss;{$[x~1f;x;x y]}[;i] each M]}[m;X;Y;loss;M] peach i
 }

/ gradient of loss over a single shard, with one mask per layer as for .qs.forwardMasked
.qs.grad1:{[m;X;Y;loss;M]
    za:.qs.forwardMasked[m;X;M]; Z:za 0; U:za 1; A:za 2; out:last A; f:last m`act;
    dOut:$[any (f;loss)~/:.qs.paired; (out-Y)%count Y;
        f=`softmax; '"softmax output needs xent loss";
        .qs.loss.g[loss][out;Y]*.qs.act.d[f][last Z;out]];
    / deltas from the output layer back: d_l = (d_(l+1) mmu W[l+1]) * f_l'(Z[l]) * mask_l
    D:{[ds;w;z;u;k;f] (enlist k*.qs.act.d[f][z;u]*first[ds] mmu w),ds}/[enlist dOut;reverse 1_m`weight;reverse -1_Z;reverse -1_U;reverse -1_M;reverse -1_m`act];
    `weight`bias!({flip[x] mmu y}'[D;-1_A];sum each D)
 }

/ predicted class per row: argmax, or a 0.5 threshold for single-output models
.qs.classes:{[A] $[1=count first A; "j"$0.5<=first each A; {x?max x} each A]}

/ fraction of rows whose predicted class matches the target's class
.qs.accuracy:{[A;Y] avg .qs.classes[A]=.qs.classes Y}

/ true when targets are class labels: one-hot rows, or a single 0/1 column
.qs.isClass:{[Y] all raze[Y] in 0 1f}

/ regression errors over every output: mean absolute, root mean square, and R squared
.qs.regression:{[A;Y] e:raze A-Y; d:y-avg y:raze Y; `mae`rmse`r2!(avg abs e;sqrt avg e*e;1-(sum e*e)%sum d*d)}

/ loss, accuracy (null for regression targets), and regression errors of a model on (X;Y)
.qs.evaluate:{[m;X;Y;loss]
    A:.qs.predict[m;"f"$X]; Y:"f"$Y;
    (`loss`acc!(.qs.loss.v[loss][A;Y];$[.qs.isClass Y;.qs.accuracy[A;Y];0n])),.qs.regression[A;Y]
 }

/ confusion matrix: row is the true class, column the predicted class
.qs.confusion:{[m;X;Y]
    t:.qs.classes Y; p:.qs.classes .qs.predict[m;"f"$X]; k:max 2,count first Y;
    {.[x;y;+;1]}/[(k;k)#0;flip (t;p)]
 }

/ optimisers: each takes (opts;state;grad) and returns (update;state), the update being subtracted from the model
.qs.opt.init:`sgd`momentum`adam!(
    {[g] ::};
    {[g] 0f*g};
    {[g] `t`m1`m2!(0;0f*g;0f*g)})
.qs.opt.step:`sgd`momentum`adam!(
    {[o;s;g] (o[`lr]*g;s)};
    {[o;s;g] v:(o[`momentum]*s)+g; (o[`lr]*v;v)};
    {[o;s;g]
        t:1+s`t; b1:o`beta1; b2:o`beta2;
        m1:(b1*s`m1)+(1-b1)*g;
        m2:(b2*s`m2)+(1-b2)*g*g;
        ((o[`lr]*m1%1-b1 xexp t)%1e-8+sqrt m2%1-b2 xexp t;`t`m1`m2!(t;m1;m2))})

/ learning rate for epoch e (counting from 1) under each schedule
.qs.pi:acos -1
.qs.schedule:`none`step`cosine!(
    {[o;e] o`lr};
    {[o;e] o[`lr]*o[`decay] xexp floor (e-1)%o`stepEvery};
    {[o;e] o[`minLr]+0.5*(o[`lr]-o`minLr)*1+cos .qs.pi*(e-1)%o`epochs})

/ training options
/ verbose is 0 for silent, or log every n epochs; dropout is the rate on hidden layers during training
/ schedule step multiplies lr by decay every stepEvery epochs, cosine anneals it to minLr
/ patience stops after that many epochs without a lower monitored loss (validation loss if there is
/ validation data, else training loss); keepBest returns the model from the epoch with the lowest one
/ onEpoch is called with each epoch's history row, e.g. to report progress
.qs.defaults:`epochs`batch`lr`loss`optimiser`momentum`beta1`beta2`l2`dropout`schedule`decay`stepEvery`minLr`patience`keepBest`shuffle`seed`verbose`onEpoch!(
    10;32;0.1;`mse;`sgd;0.9;0.9;0.999;0f;0f;`none;0.5;10;0f;0N;0b;1b;42;1;{[r]})

/ one gradient step on batch rows b; s holds `model`state (and anything else, passed through)
.qs.step:{[o;X;Y;s;b]
    m:s`model;
    g:.qs.gradDrop[m;X b;Y b;o`loss;o`dropout];
    if[o[`l2]>0; g[`weight]+:o[`l2]*m`weight];
    r:.qs.opt.step[o`optimiser][o;s`state;g];
    m[`weight]-:r[0]`weight; m[`bias]-:r[0]`bias;
    s[`model]:m; s[`state]:r 1;
    s
 }

/ one epoch: shuffle, then a step for every batch, including a smaller final one
.qs.epoch:{[o;X;Y;s]
    i:$[o`shuffle; neg[n]?n:count X; til count X];
    .qs.step[o;X;Y]/[s;o[`batch] cut i]
 }

/ signal unless the training data and options make sense for model m
.qs.checkTrain:{[m;X;Y;o]
    if[not count[X]=count Y; '"X and Y have different numbers of rows"];
    if[not count[first X]=count first first m`weight; '"X has ",(string count first X)," columns, the model expects ",string count first first m`weight];
    if[not count[first Y]=count last m`bias; '"Y has ",(string count first Y)," columns, the model outputs ",string count last m`bias];
    if[not o[`optimiser] in key .qs.opt.step; '"unknown optimiser: ",string o`optimiser];
    if[not o[`loss] in key .qs.loss.v; '"unknown loss: ",string o`loss];
    if[not o[`schedule] in key .qs.schedule; '"unknown schedule: ",string o`schedule];
    if[not o[`dropout] within 0 0.99; '"dropout must be in [0,1)"];
    if[(not null o`patience) and 1>o`patience; '"patience must be at least 1"];
 }

/ train model m on data (`X`Y and optionally `Xval`Yval) with options merged over .qs.defaults
/ instead of X and Y, data may hold gen, a function of the epoch number (from 1) returning `X`Y: each epoch then
/ trains on freshly generated rows (e.g. new noise and injections), while Xval and Yval stay fixed
/ returns `model`history`bestEpoch, history being one row per epoch run
.qs.train:{[m;data;opts]
    o:.qs.defaults,opts;
    m:.qs.check m;
    if[not null o`seed; system "S ",string o`seed];
    gen:`gen in key data;
    b:$[gen; data[`gen] 1; data];
    X:"f"$b`X; Y:"f"$b`Y;
    .qs.checkTrain[m;X;Y;o];
    val:all `Xval`Yval in key data;
    s:`model`state`history`best`bestLoss`bestEpoch`wait`stop!(m;.qs.opt.init[o`optimiser] `weight`bias#m;();m;0w;0N;0;0b);
    s:{[o;X;Y;data;val;s;e]
        if[s`stop; :s];
        t0:.z.p; lr:.qs.schedule[o`schedule][o;e];
        if[(`gen in key data) and e>1; b:data[`gen] e; X:"f"$b`X; Y:"f"$b`Y];
        s:.qs.epoch[@[o;`lr;:;lr];X;Y;s];
        tr:.qs.evaluate[s`model;X;Y;o`loss];
        va:$[val; .qs.evaluate[s`model;data`Xval;data`Yval;o`loss]; `loss`acc!0n 0n];
        row:`epoch`lr`trainLoss`trainAcc`valLoss`valAcc`seconds!(e;lr;tr`loss;tr`acc;va`loss;va`acc;1e-9*.z.p-t0);
        s[`history],:enlist row;
        o[`onEpoch] row;
        l:$[val;va`loss;tr`loss];
        $[l<s`bestLoss; s[`best`bestLoss`bestEpoch`wait]:(s`model;l;e;0); s[`wait]+:1];
        s[`stop]:(not null o`patience) and s[`wait]>=o`patience;
        if[(o[`verbose]>0) and (0=e mod o`verbose) or (e=o`epochs) or s`stop; -1 .qs.fmt row];
        if[(o[`verbose]>0) and s`stop; -1 "stopped: no improvement for ",(string o`patience)," epochs, best was epoch ",string s`bestEpoch];
        s
    }[o;X;Y;data;val]/[s;1+til o`epochs];
    `model`history`bestEpoch!($[o`keepBest;s`best;s`model];s`history;s`bestEpoch)
 }

/ one log line for a history row
.qs.fmt:{[r]
    f:{.Q.fmt[x+1+count string floor y;x;y]};
    a:{[f;x] $[null x; ""; " acc ",f[4;x]]}[f];
    "epoch ",(string r`epoch),"  train loss ",f[5;r`trainLoss],a[r`trainAcc],
        $[null r`valLoss; ""; "  val loss ",f[5;r`valLoss],a r`valAcc],"  ",f[2;r`seconds],"s"
 }

/ one row per layer: inputs, outputs, activation and number of parameters
.qs.summary:{[m]
    ([] layer:til count m`weight; inputs:count each first each m`weight; outputs:count each m`weight; act:m`act;
        params:{(count[x]*count first x)+count y}'[m`weight;m`bias])
 }

/ save and load models with q's native serialisation
.qs.save:{[file;m] (hsym file) set .qs.check m}
.qs.load:{[file] .qs.check get hsym file}

/ a configuration for data d: base (model keys hidden, act, out and training options) filled in with
/ defaults, the output activation and loss following the targets as train.q does
.qs.spec:{[d;base]
    base:(`hidden`act`verbose!(enlist 16;`tanh;0)),base;
    nOut:count first d`Y;
    base:(enlist[`out]!enlist $[not .qs.isClass d`Y;`linear;nOut>1;`softmax;`sigmoid]),base;
    (enlist[`loss]!enlist $[base[`out]=`softmax;`xent;base[`out]=`sigmoid;`bce;`mse]),base
 }

/ train a new model to configuration c on d and score it on d's validation data
/ returns `valLoss`valAcc`valR2`bestEpoch`seconds, valAcc for classes and valR2 for numeric targets
.qs.fitScore:{[d;c]
    m:.qs.init[count first d`X;c[`hidden],count first d`Y;((count c`hidden)#c`act),c`out;(.qs.defaults,c)`seed];
    t0:.z.p; r:.qs.train[m;d;`hidden`act`out _ c];
    e:.qs.evaluate[r`model;d`Xval;d`Yval;c`loss];
    `valLoss`valAcc`valR2`bestEpoch`seconds!(e`loss;e`acc;$[null e`acc;e`r2;0n];r`bestEpoch;1e-9*.z.p-t0)
 }

/ the configurations of a search space (option -> candidate values): every combination, or n of them sampled
/ without replacement when n is not null; each is a dictionary option -> value
.qs.grid:{[space;n]
    k:key space; v:value space;
    ix:{raze x,/:\:y}/[enlist `long$();til each count each v];
    if[not null n; ix:ix neg[n&count ix]?count ix];
    {[k;v;i] k!v@'i}[k;v] each ix
 }

/ hyperparameter search: train one model per configuration and rank them by validation loss
/ space maps option names to candidate values; hidden (a list of layer-size lists) and act (hidden activation)
/ shape the model, everything else is a training option; base holds fixed values for both, including the
/ output activation out (default: softmax for several outputs, sigmoid for one) and loss
/ n is 0N for the full grid, or a number of configurations sampled from it at random
/ returns a table: one row per configuration, its values, then valLoss, valAcc (classification) or valR2
/ (regression), bestEpoch, seconds; configurations run one after another, each using threads as usual
.qs.search:{[d;space;base;n]
    if[not all `Xval`Yval in key d; '"search needs validation data (Xval and Yval)"];
    if[not null s:(.qs.defaults,base)`seed; system "S ",string s];
    g:.qs.grid[space;n];
    base:.qs.spec[d;base];
    `valLoss xasc {[d;base;c] c,.qs.fitScore[d;base,c]}[d;base] each g
 }

/ k-fold cross-validation of configuration base on d (`X`Y; any Xval/Yval are added in first)
/ rows are shuffled and cut into k folds; each fold validates a model trained on the others
/ returns `folds`summary: one row per fold, and the mean and sd of each score over the folds
/ note: data from .qs.data.fromTable was standardised on its training split, so this is slightly optimistic
.qs.crossValidate:{[d;k;base]
    X:d[`X],$[`Xval in key d;d`Xval;()]; Y:d[`Y],$[`Yval in key d;d`Yval;()];
    if[k<2; '"need at least 2 folds"]; if[k>count X; '"more folds than rows"];
    if[not null s:(.qs.defaults,base)`seed; system "S ",string s];
    f:{x where 0<count each x} (ceiling count[X]%k) cut neg[n]?n:count X;
    base:.qs.spec[`X`Y!(X;Y);base];
    t:{[X;Y;f;base;j] i:raze f _ j; v:f j; (enlist[`fold]!enlist j),.qs.fitScore[`X`Y`Xval`Yval!(X i;Y i;X v;Y v);base]}[X;Y;f;base] each til count f;
    s:`valLoss`valAcc`valR2;
    `folds`summary!(t;([] metric:s; mean:avg each t s; sd:dev each t s))
 }

