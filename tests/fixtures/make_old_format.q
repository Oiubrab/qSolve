/ writes tests/fixtures/old_format_cases: models in the pre-M1 layout (weight/bias/act, conv, gru) and what the
/ library computed for them then: predictions, logits, gradients and short training runs. tests/test_core_compat.q
/ checks that the layer-registry core (M1) converts the models and reproduces every number.
/ Run it only with the library as it was before M1 (commit f162ffd, src/defineEquation.q); it can't run on the new core,
/ which no longer produces this layout. From the repository root: q tests/fixtures/make_old_format.q
system "l src/defineEquation.q";
system "S 1";
rnd:{[n;d] {x?1f} each n#d};
oneHot:{[k;n] "f"$(til k)=/:n?k};
mk:{[name;m;X;Y;loss;opts]
    g:.qs.grad[m;X;Y;loss];
    r:.qs.train[m;`X`Y!(X;Y);(`loss`verbose!(loss;0)),opts];
    `name`model`X`Y`loss`opts`pred`raw`grad`trained`trainedPred`trainLoss`bestEpoch!(name;m;X;Y;loss;opts;.qs.predict[m;X];.qs.predictRaw[m;X];g;r`model;.qs.predict[r`model;X];r[`history]`trainLoss;r`bestEpoch)
 };
cases:();
/ MLPs: relu hidden layer (He init), softmax and xent, adam with l2 and dropout
X:rnd[20;3]; cases,:enlist mk[`mlp_softmax;.qs.init[3;5 4 2;`tanh`relu`softmax;11];X;oneHot[2;20];`xent;`epochs`batch`optimiser`lr`l2`dropout`seed!(4;8;`adam;0.01;0.01;0.3;5)];
cases,:enlist mk[`mlp_sigmoid;.qs.init[3;4 1;`relu`sigmoid;12];X;enlist each "f"$20?0 1;`bce;`epochs`batch`optimiser`lr`seed!(4;5;`momentum;0.05;6)];
cases,:enlist mk[`mlp_linear_sgd;.qs.init[3;4 1;`tanh`linear;13];X;enlist each 20?1f;`mse;`epochs`batch`lr`seed!(3;7;0.1;7)];
/ conv: two conv layers with pooling; and a pool-1 layer, relu, straight into a linear output
X:rnd[12;48]; Y:enlist each "f"$12?0 1;
cases,:enlist mk[`conv_pooled;.qs.initConv[2;24;([] filters:3 2; kernel:5 3; pool:2 2; act:`tanh`sigmoid);4 1;`tanh`sigmoid;21];X;Y;`bce;`epochs`batch`optimiser`lr`l2`seed!(3;4;`adam;0.01;0.01;8)];
cases,:enlist mk[`conv_pool1;.qs.initConv[2;24;([] filters:2 3; kernel:4 2; pool:1 2; act:`relu`relu);2 1;`relu`sigmoid;22];X;Y;`bce;`epochs`batch`optimiser`lr`seed!(3;4;`adam;0.01;9)];
/ GRU front end
cases,:enlist mk[`gru;.qs.initGru[2;24;4;5;4 1;`tanh`sigmoid;23];X;Y;`bce;`epochs`batch`optimiser`lr`l2`seed!(3;4;`adam;0.01;0.01;10)];
/ a model that carries a table encoding and an info dict, as the dashboard saves (extra keys must survive)
m:.qs.init[3;4 2;`tanh`softmax;14]; m[`enc]:([] col:`a`b; sd:1 2f); m[`info]:`kind`note!(`table;"kept");
X:rnd[20;3]; cases,:enlist mk[`mlp_extras;m;X;oneHot[2;20];`xent;`epochs`batch`seed!(2;5;3)];
`:tests/fixtures/old_format_cases set cases;
-1 "wrote ",(string count cases)," cases: ",", " sv string cases`name;
exit 0
