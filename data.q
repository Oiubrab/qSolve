/ qSolve datasets: generators and loaders, each returning `X`Y (and `Xval`Yval where there is a split)
/ X is a matrix with one row per example, Y one-hot (or a single column for binary and regression targets)
/ generators use q's random state: set it with system "S <seed>" for reproducible data

.qs.data.pi:acos -1

/ n draws from the standard normal distribution (Box-Muller)
.qs.data.normal:{[n] sqrt[-2*log 1-n?1f]*cos 2*.qs.data.pi*n?1f}

/ one-hot rows for integer labels in 0..k-1
.qs.data.oneHot:{[k;labels] "f"$labels=\:til k}

/ the four rows of a two-input logic gate f, e.g. gate[`xor]
.qs.data.gates:`and`or`xor`nand!(&;|;<>;{not x&y})
.qs.data.gate:{[f]
    X:(0 0f;0 1f;1 0f;1 1f);
    `X`Y!(X;enlist each "f"$.qs.data.gates[f] .' "b"$X)
 }

/ the original qSolve task: 4 random bits, a fifth of rows forced to 1100 or 0011
/ classes are 1100, 0011, and neither (most rows), one-hot over three columns
.qs.data.fourBit:{[n]
    X:"f"$(n;4)#(4*n)?0 1;
    i:5*til ceiling n%5;
    X[i]:(count i)?(1 1 0 0f;0 0 1 1f);
    c:{$[x~1 1 0 0f;0;x~0 0 1 1f;1;2]} each X;
    `X`Y!(X;.qs.data.oneHot[3;c])
 }

/ regression: y = sin x for x in [0,2pi], with X scaled to [-1,1] and Y to [0,1]
.qs.data.sine:{[n]
    x:n?2*.qs.data.pi;
    `X`Y!(enlist each -1+x%.qs.data.pi;enlist each 0.5+0.5*sin x)
 }

/ binary 2-D datasets: n points, half per class, gaussian noise of sd noise added to each coordinate
.qs.data.noisy2D:{[pts;labels;noise] `X`Y!(pts+noise*(count pts;2)#.qs.data.normal 2*count pts;.qs.data.oneHot[2;labels])}

/ two concentric circles of radius 1 (class 0) and 0.5 (class 1)
.qs.data.circles:{[n;noise]
    k:n div 2; t:n?2*.qs.data.pi; r:?[n#(k#1b),(n-k)#0b;1f;0.5];
    .qs.data.noisy2D[flip (r*cos t;r*sin t);"j"$r=0.5;noise]
 }

/ two interleaving half circles
.qs.data.moons:{[n;noise]
    k:n div 2; t:n?.qs.data.pi; c:n#(k#0),(n-k)#1;
    x:?[c=0;cos t;1-cos t]; y:?[c=0;sin t;0.5-sin t];
    .qs.data.noisy2D[flip (x;y);c;noise]
 }

/ two interleaved spirals, 1.5 turns each, scaled to about [-1,1]
.qs.data.spirals:{[n;noise]
    k:n div 2; t:3*.qs.data.pi*sqrt n?1f; c:n#(k#0),(n-k)#1;
    s:?[c=0;1f;-1f]*t%3*.qs.data.pi;
    .qs.data.noisy2D[flip (s*cos t;s*sin t);c;noise]
 }

/ shuffle and split data into training and validation parts, frac of rows going to training
.qs.data.split:{[d;frac]
    i:neg[n]?n:count d`X; k:floor frac*n;
    `X`Y`Xval`Yval!(d[`X] k#i;d[`Y] k#i;d[`X] k _i;d[`Y] k _i)
 }

/ column means and standard deviations, and scaling by them (sd 0 columns are left unscaled)
.qs.data.stats:{[X] sd:dev X; `mu`sd!(avg X;?[sd=0;1f;sd])}
.qs.data.scale:{[st;X] (X-\:st`mu)%\:st`sd}

/ standardise X and Xval using the training columns' statistics
.qs.data.standardise:{[d]
    st:.qs.data.stats d`X;
    d[`X]:.qs.data.scale[st;d`X];
    if[`Xval in key d; d[`Xval]:.qs.data.scale[st;d`Xval]];
    d
 }

/ read an IDX file of unsigned bytes (the MNIST format): a vector for 1-D files, otherwise one row per item
.qs.data.idx:{[file]
    b:read1 hsym file;
    if[not (0x0000~2#b) and 0x08=b 2; '"not an unsigned-byte IDX file: ",string file];
    dims:{"j"$0x0 sv x} each 4 cut (4*"j"$b 3)#4_b;
    v:(4+4*count dims)_b;
    if[not (prd dims)=count v; '"IDX file is truncated: ",string file];
    $[1=count dims; "j"$v; (first dims;prd 1_dims)#v]
 }

/ MNIST-format directory (MNIST or Fashion-MNIST): pixels scaled to [0,1], labels one-hot, test set as validation
.qs.data.mnist:{[dir]
    f:{[dir;x] .qs.data.idx `$(string dir),"/",x}[dir];
    `X`Y`Xval`Yval!(
        (1%255)*"f"$f "train-images-idx3-ubyte";
        .qs.data.oneHot[10;f "train-labels-idx1-ubyte"];
        (1%255)*"f"$f "t10k-images-idx3-ubyte";
        .qs.data.oneHot[10;f "t10k-labels-idx1-ubyte"])
 }

/ the first n training and nval validation rows of a dataset
.qs.data.take:{[d;n;nval] d:@[d;`X`Y;n sublist]; $[`Xval in key d; @[d;`Xval`Yval;nval sublist]; d]}

/ Iris CSV (UCI iris.data): 4 standardised features, 3 one-hot classes, shuffled 80/20 split
.qs.data.iris:{[file]
    t:flip `sl`sw`pl`pw`species!("FFFFS";",") 0: hsym file;
    t:select from t where not null species;
    names:asc distinct t`species;
    d:`X`Y!(flip t`sl`sw`pl`pw;.qs.data.oneHot[3;names?t`species]);
    .qs.data.standardise .qs.data.split[d;0.8]
 }

/ directory holding downloaded data (see scripts/fetch_data.sh)
.qs.data.dir:"data"

/ datasets by name, split into training and validation: n rows and noise are used by the generated ones
/ the logic gates validate on their own four rows
.qs.data.names:`and`or`xor`nand`fourBit`sine`circles`moons`spirals`iris`mnist`fashion
.qs.data.load:{[name;n;noise]
    $[name in key .qs.data.gates; {x,`Xval`Yval!x`X`Y} .qs.data.gate name;
        name in `fourBit`sine; .qs.data.split[.qs.data[name] n;0.8];
        name in `circles`moons`spirals; .qs.data.split[.qs.data[name][n;noise];0.8];
        name=`iris; .qs.data.iris `$":",.qs.data.dir,"/iris/iris.data";
        name in `mnist`fashion; .qs.data.mnist `$":",.qs.data.dir,"/",string name;
        '"unknown dataset: ",(string name),", expected one of ",", " sv string .qs.data.names]
 }
