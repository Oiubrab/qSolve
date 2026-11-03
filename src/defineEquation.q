/ qSolve: a small neural network library in q
/ a model is a list of layers (see the layer registry below); dense, conv and GRU networks are all built from it

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

/ ---------- layers (M1) ----------
/ A layer type is one entry in each of the dictionaries .qs.layer.<part>, keyed by its kind (a symbol); .qs.layer.add
/ registers a type from a dictionary of its parts. Parts of a type:
/   names    the names of its parameter arrays, in order (empty for a layer without any)
/   shape    [spec;inShape] -> outShape, signalling if it can't take that input
/   init     [spec;inShape] -> dictionary names -> arrays (may draw random numbers)
/   forward  [spec;p;A;t]   -> (output;cache); p is the dictionary of parameters, t is 0b when not training, else
/            the layer's training input (1b, or a dropout layer's mask)
/   backward [spec;p;cache;dOut;needIn] -> (dIn;gradients dictionary); dIn is () unless needIn
/   check    [spec;p]       -> signals if p doesn't fit the spec
/   l2       the parameters that weight decay (l2) applies to
/   inGrad   1b if backward can give the gradient at the input
/   example  a list of (spec;inShape) pairs that .qs.gradCheck runs, for the tests
/ Shapes: ,d is a batch matrix (n rows of d values); (c;t) is a sequence in the channel-major "long" layout, c
/ vectors each holding every example's t steps in turn. Parameters travel as a list of arrays per layer.
/ q turns a list of dictionaries with the same keys into a table, so a model never holds one: layers are
/ (kind;spec) pairs, parameters and caches are lists.
.qs.layer.names:(`symbol$())!()
.qs.layer.shape:(`symbol$())!()
.qs.layer.init:(`symbol$())!()
.qs.layer.forward:(`symbol$())!()
.qs.layer.backward:(`symbol$())!()
.qs.layer.check:(`symbol$())!()
.qs.layer.l2:(`symbol$())!()
.qs.layer.inGrad:(`symbol$())!()
.qs.layer.example:(`symbol$())!()
.qs.layer.add:{[k;d] {[k;d;n] v:`$".qs.layer.",string n; v set (get v),enlist[k]!enlist d n}[k;d] each key d; (::)}

/ a layer's parameters as a dictionary, from the list of arrays the model holds
.qs.pdict:{[k;p] .qs.layer.names[k]!p}

/ dense: a fully connected layer, W is n x nin and b has n values (Xavier-uniform, or He-uniform for scheme he)
.qs.dense.shape:{[s;inS]
    if[not 1=count inS; '"dense needs a flat input (flatten the conv layers first)"];
    enlist "j"$s`n
 }
.qs.dense.init:{[s;inS]
    nin:first inS; n:s`n;
    lim:$[`he=$[`scheme in key s; s`scheme; `xavier]; sqrt 6%nin; sqrt 6%nin+n];
    `W`b!((n;nin)#lim*-1+2*(n*nin)?1f;n#0f)
 }
.qs.dense.forward:{[s;p;A;t] ((A mmu flip p`W)+\:p`b;A)}
.qs.dense.backward:{[s;p;c;dZ;needIn] ($[needIn;dZ mmu p`W;()];`W`b!(flip[dZ] mmu c;sum dZ))}
.qs.dense.check:{[s;p]
    if[not (s[`n],first s`inShape)~(count p`W;count first p`W); '"weights don't match the layer's sizes"];
    if[not (s`n)=count p`b; '"bias length must match the number of nodes"];
 }

/ act: an activation function, elementwise (softmax acts on rows, so only on a flat output)
.qs.actl.shape:{[s;inS] inS}
.qs.actl.init:{[s;inS] ()!()}
.qs.actl.forward:{[s;p;A;t] U:.qs.act.f[s`f] A; (U;(A;U))}
.qs.actl.backward:{[s;p;c;dU;needIn]
    if[not (s`f) in key .qs.act.d; '(string s`f)," needs a loss paired with it (softmax with xent)"];
    (dU*.qs.act.d[s`f][c 0;c 1];()!())
 }
.qs.actl.check:{[s;p] if[not (s`f) in key .qs.act.f; e:"unknown activation: ",string s`f; 'e]}

/ dropout: zeroes each value with probability rate while training, scaling the rest by 1/(1-rate). The mask is the
/ layer's training input, drawn for the whole batch by .qs.aux before the batch is split over threads
.qs.dropMask:{[n;p;k] (1%1-p)*p<(n;k)#(n*k)?1f}
.qs.dropout.shape:{[s;inS] inS}
.qs.dropout.init:{[s;inS] ()!()}
.qs.dropout.forward:{[s;p;A;t] $[t~0b; (A;()); (A*t;t)]}
.qs.dropout.backward:{[s;p;c;dA;needIn] (dA*c;()!())}
.qs.dropout.check:{[s;p] if[not (s`rate)>=0; '"dropout rate must be at least 0"]; if[not (s`rate)<1; '"dropout rate must be below 1"]}

/ seq: reads a flat batch (each row holds its channels one after another) as channels x (n*length), the layout
/ conv layers work in. Always the first layer; its gradient is never needed
.qs.conv.toLong:{[X;c;len] {[X;len;ch] raze X[;(ch*len)+til len]}[X;len] each til c}
.qs.seq.shape:{[s;inS]
    if[not (enlist s[`channels]*s`length)~inS; '"seq: the input size isn't channels x length"];
    "j"$(s`channels;s`length)
 }
.qs.seq.init:{[s;inS] ()!()}
.qs.seq.forward:{[s;p;A;t] (.qs.conv.toLong[A;s`channels;s`length];())}
.qs.seq.backward:{[s;p;c;dA;needIn]
    if[needIn; len:s`length; n:(count first dA) div len; :((,'/) {[n;len;v] (n;len)#v}[n;len] each dA;()!())];
    ((); ()!())
 }
.qs.seq.check:{[s;p] }

/ conv1d: valid 1-D convolution (no padding, stride 1) over the channels. K is filters x (kernel*cin): row f holds
/ filter f's weights for tap 0 across the channels, then tap 1, and so on. A convolution is a gather of shifted
/ copies of the input (Q, one block of cin rows per tap) and one mmu
.qs.conv1d.shape:{[s;inS]
    if[not 2=count inS; '"conv1d needs a (channels;length) input: put seq first"];
    t:(inS 1)-s[`kernel]-1;
    if[1>t; e:"conv layers shrink the input to nothing: kernel ",(string s`kernel)," on length ",string inS 1; 'e];
    "j"$(s`filters;t)
 }
.qs.conv1d.init:{[s;inS]
    fan:(s`kernel)*first inS; f:s`filters; lim:sqrt 6%fan;
    `K`b!((f;fan)#lim*-1+2*(f*fan)?1f;f#0f)
 }
/ cache: Q, the gather indices, the output length (tn) and the input length
.qs.conv1d.forward:{[s;p;A;t]
    k:s`kernel; ti:(s`inShape) 1; tn:ti-k-1; n:(count first A) div ti;
    base:raze (ti*til n)+\:til tn;
    Q:raze {[A;base;j] A[;base+j]}[A;base] each til k;
    ((p[`K] mmu Q)+p`b;(Q;base;tn;ti))
 }
.qs.conv1d.backward:{[s;p;c;dZ;needIn]
    K:p`K; k:s`kernel; cin:first s`inShape; Q:c 0; base:c 1; tn:c 2; ti:c 3;
    g:`K`b!({[Q;z] z$/:Q}[Q] each dZ;sum each dZ);
    dA:$[needIn;
        [dP:(flip K) mmu dZ; n:count[base] div tn;
         {[dP;base;cin;dA;j] .[dA;(::;base+j);+;dP (j*cin)+til cin]}[dP;base;cin]/[(cin;n*ti)#0f;til k]];
        ()];
    (dA;g)
 }
.qs.conv1d.check:{[s;p]
    if[not (s`filters;(s`kernel)*first s`inShape)~(count p`K;count first p`K); '"conv kernel shapes don't match the layer"];
    if[not (s`filters)=count p`b; '"conv bias length doesn't match the filters"];
 }

/ maxpool1d: max over non-overlapping blocks of pool steps (a remainder is dropped)
.qs.maxpool1d.shape:{[s;inS]
    if[not 2=count inS; '"maxpool1d needs a (channels;length) input"];
    r:(inS 1) div s`pool;
    if[1>r; '"pooling shrinks the input to nothing"];
    "j"$(inS 0;r)
 }
.qs.maxpool1d.init:{[s;inS] ()!()}
/ cache: for each output, which position in its block won (the first, on a tie), and the gather indices (not worked out
/ when not training). The winner is set by arithmetic, as ?[...] needs simple lists
.qs.maxpool1d.forward:{[s;p;A;t]
    pool:s`pool; ti:(s`inShape) 1; tp:ti div pool; n:(count first A) div ti;
    pbase:raze (ti*til n)+\:pool*til tp;
    V:{[U;pbase;i] U[;pbase+i]}[A;pbase] each til pool;
    O:(|/) V;
    if[t~0b; :(O;())];
    arg:{[V;O;arg;i] arg+(i-arg)*V[i]=O}[V;O]/[(count O;count first O)#pool-1;reverse til pool];
    (O;(arg;pbase))
 }
.qs.maxpool1d.backward:{[s;p;c;dO;needIn]
    pool:s`pool; ti:(s`inShape) 1; n:(count first dO) div ti div pool;
    dU:{[dO;c;dU;i] .[dU;(::;c[1]+i);+;dO*c[0]=i]}[dO;c]/[(count dO;n*ti)#0f;til pool];
    (dU;()!())
 }
.qs.maxpool1d.check:{[s;p] if[1>s`pool; '"pool must be at least 1"]}

/ flatten: (c;t) sequences to flat rows, time-major (each step's channels together)
.qs.flatten.shape:{[s;inS]
    if[not 2=count inS; '"flatten needs a (channels;length) input"];
    enlist "j"$prd inS
 }
.qs.flatten.init:{[s;inS] ()!()}
.qs.flatten.forward:{[s;p;A;t]
    tp:(s`inShape) 1; n:(count first A) div tp;
    ({[O;tp;e] raze flip O[;(e*tp)+til tp]}[A;tp] each til n;())
 }
.qs.flatten.backward:{[s;p;c;dF;needIn]
    f:first s`inShape;
    ($[needIn; flip raze {[f;x] ((count x) div f;f)#x}[f] each dF; ()];()!())
 }
.qs.flatten.check:{[s;p] }

/ gru: a gated recurrent unit over the flat input, whose rows hold channels x length values (channel-major). Each of
/ the length div block steps reads the next block samples of every channel (channels*block inputs), so long
/ sequences take few steps without discarding samples. The final hidden state is the output. Parameters: W (3*hidden
/ x channels*block; update, reset and candidate rows in that order), U (3*hidden x hidden) and b (3*hidden). One step,
/ from state h (zero at the start):
/   z = sigmoid(x Wz + h Uz + bz), r = sigmoid(x Wr + h Ur + br), n = tanh(x Wn + r*(h Un) + bn), h' = (1-z) n + z h
.qs.gru.sig:.qs.act.f`sigmoid
.qs.gru.shape:{[s;inS]
    if[not (enlist s[`channels]*s`length)~inS; '"gru: the input size isn't channels x length"];
    if[not 0=s[`length] mod s`block; '"the GRU block must divide the length"];
    enlist "j"$s`hidden
 }
/ input weights are Xavier-uniform, recurrent ones uniform in +-1/sqrt(hidden), biases zero (W is drawn first)
.qs.gru.init:{[s;inS]
    H:s`hidden; nin:(s`channels)*s`block; a:sqrt 6%nin+H; b:1%sqrt H;
    W:(3*H;nin)#a*-1+2*(3*H*nin)?1f;
    U:(3*H;H)#b*-1+2*(3*H*H)?1f;
    `W`U`b!(W;U;(3*H)#0f)
 }
/ the inputs of each step for batch X: one (rows x channels*block) matrix per step
.qs.gru.steps:{[X;s]
    c:s`channels; len:s`length; k:s`block;
    {[X;c;len;k;t] raze each flip {[X;len;k;t;ch] X[;(ch*len)+(t*k)+til k]}[X;len;k;t] each til c}[X;c;len;k] each til len div k
 }
/ cache: each step's state and gates, and the steps' inputs
.qs.gru.forward:{[s;p;X;t]
    xs:.qs.gru.steps[X;s]; H:s`hidden;
    q:`W`U`b`jz`jr`jn!(p`W;p`U;p`b;til H;H+til H;(2*H)+til H);
    one:{[q;st;x]
        h:st 0; a:(x mmu flip q`W)+\:q`b; u:h mmu flip q`U;
        z:.qs.gru.sig a[;q`jz]+u[;q`jz]; r:.qs.gru.sig a[;q`jr]+u[;q`jr]; hn:u[;q`jn];
        n:.qs.act.f[`tanh] a[;q`jn]+r*hn;
        (((1-z)*n)+z*h;st[1],enlist (h;z;r;n;hn))}[q];
    r:one/[((count X;H)#0f;());xs];
    (r 0;(r 1;xs))
 }
/ gradients by backpropagation through time, from the gradient at the final state
.qs.gru.backward:{[s;p;c;dh;needIn]
    if[needIn; '"the GRU's input gradient isn't implemented: it has to be the first layer"];
    one:{[U;st;cc;x]
        dh:st 0; h:cc 0; z:cc 1; r:cc 2; n:cc 3; hn:cc 4;
        dn:dh*1-z; dz:dh*h-n; dan:dn*1-n*n;
        daz:dz*z*1-z; dar:(dan*hn)*r*1-r; dhn:dan*r;
        / gradients at the input and recurrent pre-activations (rows x 3*hidden)
        dA:daz,'dar,'dan; dU:daz,'dar,'dhn;
        ((dh*z)+dU mmu U;st[1]+flip[dA] mmu x;st[2]+flip[dU] mmu h;st[3]+sum dA)}[p`U];
    r:one/[(dh;0f*p`W;0f*p`U;0f*p`b);reverse c 0;reverse c 1];
    (();`W`U`b!(r 1;r 2;r 3))
 }
.qs.gru.check:{[s;p]
    H:s`hidden; nin:(s`channels)*s`block;
    if[not ((3*H),nin)~(count p`W;count first p`W); '"GRU input weights must be 3*hidden x channels*block"];
    if[not ((3*H),H)~(count p`U;count first p`U); '"GRU recurrent weights must be 3*hidden x hidden"];
    if[not (3*H)=count p`b; '"GRU bias must have 3*hidden values"];
 }

.qs.layer.add[`dense;`names`shape`init`forward`backward`check`l2`inGrad`example!(`W`b;.qs.dense.shape;.qs.dense.init;.qs.dense.forward;.qs.dense.backward;.qs.dense.check;enlist `W;1b;enlist ((enlist[`n]!enlist 3);enlist 4))]
.qs.layer.add[`act;`names`shape`init`forward`backward`check`l2`inGrad`example!(`symbol$();.qs.actl.shape;.qs.actl.init;.qs.actl.forward;.qs.actl.backward;.qs.actl.check;`symbol$();1b;((enlist[`f]!enlist `tanh;enlist 5);(enlist[`f]!enlist `sigmoid;2 4);(enlist[`f]!enlist `relu;2 4)))]
.qs.layer.add[`dropout;`names`shape`init`forward`backward`check`l2`inGrad`example!(`symbol$();.qs.dropout.shape;.qs.dropout.init;.qs.dropout.forward;.qs.dropout.backward;.qs.dropout.check;`symbol$();1b;enlist ((enlist[`rate]!enlist 0.4);enlist 6))]
.qs.layer.add[`seq;`names`shape`init`forward`backward`check`l2`inGrad`example!(`symbol$();.qs.seq.shape;.qs.seq.init;.qs.seq.forward;.qs.seq.backward;.qs.seq.check;`symbol$();1b;enlist (`channels`length!2 5;enlist 10))]
.qs.layer.add[`conv1d;`names`shape`init`forward`backward`check`l2`inGrad`example!(`K`b;.qs.conv1d.shape;.qs.conv1d.init;.qs.conv1d.forward;.qs.conv1d.backward;.qs.conv1d.check;enlist `K;1b;enlist (`filters`kernel!3 4;2 10))]
.qs.layer.add[`maxpool1d;`names`shape`init`forward`backward`check`l2`inGrad`example!(`symbol$();.qs.maxpool1d.shape;.qs.maxpool1d.init;.qs.maxpool1d.forward;.qs.maxpool1d.backward;.qs.maxpool1d.check;`symbol$();1b;((enlist[`pool]!enlist 2;3 8);(enlist[`pool]!enlist 3;2 8)))]
.qs.layer.add[`flatten;`names`shape`init`forward`backward`check`l2`inGrad`example!(`symbol$();.qs.flatten.shape;.qs.flatten.init;.qs.flatten.forward;.qs.flatten.backward;.qs.flatten.check;`symbol$();1b;enlist (()!();3 4))]
.qs.layer.add[`gru;`names`shape`init`forward`backward`check`l2`inGrad`example!(`W`U`b;.qs.gru.shape;.qs.gru.init;.qs.gru.forward;.qs.gru.backward;.qs.gru.check;`W`U;0b;enlist (`channels`length`block`hidden!2 8 2 3;enlist 16))]

/ the index of every element of a list of arrays: (i;j) for a matrix or a list of vectors, ,i for a vector
.qs.paths:{[v] $[0h=type v; raze {[v;i] i,/:.qs.paths v i}[v] each til count v; enlist each til count v]}

/ largest error between a layer's backward pass and central finite differences (M1.3): for L = the sum of the output
/ times a fixed random matrix, over every parameter and, where the layer has one, the input gradient. spec is the
/ layer's spec and inS its input shape; 3 examples per batch
.qs.gradCheck:{[k;spec;inS]
    s:spec,`inShape`outShape!(inS;.qs.layer.shape[k][spec;inS]);
    n:3; h:1e-6;
    p:.qs.layer.init[k][s;inS];
    A:-0.5+$[1=count inS; {x?1f} each n#first inS; {x?1f} each (first inS)#n*inS 1];
    t:$[k=`dropout; .qs.dropMask[n;s`rate;first inS]; 1b];
    o:.qs.layer.forward[k][s;p;A;t];
    R:{x?1f} each (count o 0)#count first o 0;
    b:.qs.layer.backward[k][s;p;o 1;R;.qs.layer.inGrad k];
    / L takes the parameters and the input
    L:{[k;s;t;R;p;A] sum raze R*first .qs.layer.forward[k][s;p;A;t]}[k;s;t;R];
    pe:{[L;h;p;A;g;nm]
        e:{[L;h;p;A;g;nm;i] ((L[.[p;(nm),i;+;h];A]-L[.[p;(nm),i;-;h];A])%2*h)-g . (nm),i}[L;h;p;A;g;nm] each .qs.paths p nm;
        max abs e}[L;h;p;A;b 1] each key p;
    ie:$[.qs.layer.inGrad k;
        max abs {[L;h;p;A;g;i] ((L[p;.[A;i;+;h]]-L[p;.[A;i;-;h]])%2*h)-g . i}[L;h;p;A;b 0] each .qs.paths A;
        0f];
    max 0f,pe,ie
 }

/ ---------- models ----------
/ model: `layers`params (and any extra keys a caller adds, such as a table encoding)
/   layers  a list of (kind;spec) pairs; each spec records the layer's inShape and outShape
/   params  one list of arrays per layer, in the order of .qs.layer.names[kind] (empty for layers without any)
/ batches are matrices: one row per example

/ the layers' kinds, one symbol each
.qs.kinds:{[m] first each m`layers}

/ the number of inputs a model takes per row, and of outputs it gives
.qs.inputSize:{[m] L:(.qs.norm m)`layers; first L[0;1;`inShape]}
.qs.outputSize:{[m] L:(.qs.norm m)`layers; prd L[-1+count L;1;`outShape]}

/ the width of each dense layer, and each activation layer's function, in order
.qs.sizes:{[m] m:.qs.norm m; {[L;i] L[i;1;`n]}[m`layers] each where `dense=.qs.kinds m}
.qs.acts:{[m] m:.qs.norm m; {[L;i] L[i;1;`f]}[m`layers] each where `act=.qs.kinds m}

/ the shape of the data after each layer
.qs.shapes:{[m] m:.qs.norm m; {[L;i] L[i;1;`outShape]}[m`layers] each til count m`layers}

/ a model from layer descriptions: a list of (kind;spec) pairs without shapes, for input shape inS. Each layer's
/ shapes are worked out in turn and recorded in its spec. P is the list of parameter lists (aligned with the
/ layers) or () to initialise them: seed is 0N to leave the random generator alone
.qs.assemble:{[inS;layers;P;seed]
    if[not null seed; system "S ",string seed];
    / (a spec whose values are all symbols can't take a vector by amending, so the shapes are joined on)
    r:{[st;l] k:l 0; o:.qs.layer.shape[k][l 1;st 0]; s:(l 1),`inShape`outShape!(st 0;o);
        (o;st[1],enlist (k;s))}/[(inS;());layers];
    L:r 1;
    P:$[count P; P; {[L;i] k:L[i;0]; value .qs.layer.init[k][L[i;1];L[i;1]`inShape]}[L] each til count L];
    .qs.check `layers`params!(L;P)
 }

/ dense layers each followed by its activation: the He-uniform scheme for relu, Xavier-uniform otherwise
.qs.denseLayers:{[sizes;acts]
    raze {[n;a] ((`dense;`n`scheme!(n;$[a=`relu;`he;`xavier]));(`act;enlist[`f]!enlist a))}'["j"$sizes;acts]
 }

/ new model with nIn inputs and one dense layer per entry of sizes, each followed by its activation
/ acts is one activation per layer, or a single one for every layer; seed is 0N to leave the generator alone
/ weights are Xavier-uniform (He-uniform for relu), biases are zero
.qs.init:{[nIn;sizes;acts;seed]
    sizes:(),sizes;
    acts:$[1=count acts:(),acts; count[sizes]#acts; acts];
    .qs.assemble[enlist "j"$nIn;.qs.denseLayers[sizes;acts];();seed]
 }

/ the layers that turn a flat batch (channels one after another) into conv features: seq, then each conv layer's
/ conv1d, activation and (for pool above 1) maxpool1d, then flatten. layers has filters, kernel, pool and act
.qs.convLayers:{[channels;length;layers]
    body:raze {[f;k;p;a] ((`conv1d;`filters`kernel!(f;k));(`act;enlist[`f]!enlist a)),$[p>1;enlist (`maxpool1d;enlist[`pool]!enlist p);()]}'[layers`filters;layers`kernel;layers`pool;layers`act];
    (enlist (`seq;`channels`length!(channels;length))),body,enlist (`flatten;()!())
 }

/ a model with conv layers: channels x length inputs, conv layers (a table or list of dictionaries of filters,
/ kernel, pool and act), then dense layers as .qs.init (sizes and acts). Kernels are He-uniform, biases zero
.qs.initConv:{[channels;length;layers;sizes;acts;seed]
    layers:([] filters:"j"$layers@\:`filters; kernel:"j"$layers@\:`kernel; pool:"j"$layers@\:`pool; act:`$string layers@\:`act);
    sizes:(),sizes;
    acts:$[1=count acts:(),acts; count[sizes]#acts; acts];
    .qs.assemble[enlist "j"$channels*length;.qs.convLayers["j"$channels;"j"$length;layers],.qs.denseLayers[sizes;acts];();seed]
 }

/ a model with a GRU layer: channels x length inputs read block samples per channel a step, hidden units, then
/ dense layers as .qs.init (sizes and acts)
.qs.initGru:{[channels;length;block;hidden;sizes;acts;seed]
    if[not 0=length mod block; '"the GRU block must divide the length"];
    sizes:(),sizes;
    acts:$[1=count acts:(),acts; count[sizes]#acts; acts];
    g:(`gru;`channels`length`block`hidden!"j"$(channels;length;block;hidden));
    .qs.assemble[enlist "j"$channels*length;enlist[g],.qs.denseLayers[sizes;acts];();seed]
 }

/ a model from given weights, biases and activations (one dense layer each): W[l] is n_l x n_(l-1), B[l] has n_l
/ entries, acts one symbol per layer
.qs.fromWeights:{[W;B;acts]
    acts:$[1=count acts:(),acts; count[W]#acts; acts];
    P:raze {[w;b] ((w;b);())}'[W;B];
    .qs.assemble[enlist count first first W;.qs.denseLayers[count each W;acts];P;0N]
 }

/ models saved before the layer registry (keys weight, bias, act and for conv or GRU front ends conv, ckernel, cbias or
/ gru, gW, gU, gb) become the same network in the new layout; other keys are kept
.qs.fromOld:{[m]
    if[not all `weight`bias`act in key m; '"model needs layers and params, or weight, bias and act"];
    W:m`weight; B:m`bias; a:m`act;
    ex:(`weight`bias`act`conv`ckernel`cbias`gru`gW`gU`gb inter key m) _ m;
    c:$[`conv in key m; m`conv; `gru in key m; m`gru; ()];
    head:$[`conv in key m; .qs.convLayers[c`channels;c`length;c`layers]; `gru in key m; enlist (`gru;c); ()];
    inS:enlist $[count c; "j"$(c`channels)*c`length; count first first W];
    dl:head,.qs.denseLayers[count each W;a];
    K:first each dl;
    ci:-1+sums K=`conv1d; di:-1+sums K=`dense;
    gp:$[`gru in key m; (m`gW;m`gU;m`gb); ()];
    P:{[ck;cb;W;B;gp;k;c;d] $[k=`conv1d; (ck c;cb c); k=`dense; (W d;B d); k=`gru; gp; ()]}[$[`conv in key m;m`ckernel;()];$[`conv in key m;m`cbias;()];W;B;gp]'[K;ci;di];
    ex,.qs.assemble[inS;dl;P;0N]
 }

/ a model in the layer layout: old ones are converted
.qs.norm:{[m] $[`layers in key m; m; .qs.fromOld m]}

/ signal unless the model's layers, shapes and parameters are consistent; returns the model (converted, if it was
/ in the old layout)
.qs.check:{[m]
    m:.qs.norm m;
    if[not all `layers`params in key m; '"model needs layers and params"];
    L:m`layers; P:m`params; n:count L;
    if[0=n; '"the model has no layers"];
    if[not n=count P; '"need one set of parameters per layer"];
    K:first each L;
    if[count bad:K except key .qs.layer.forward; e:"unknown layer type: ",string first bad; 'e];
    if[not 1=count L[0;1;`inShape]; '"the model's input must be flat"];
    .qs.checkLayer[L;P] each til n;
    a:where K=`act;
    if[any ({[L;i] L[i;1;`f]}[L] each a)[where a<n-1]=`softmax; '"softmax is only allowed on the output layer"];
    m
 }
.qs.checkLayer:{[L;P;i]
    k:L[i;0]; s:L[i;1]; e:" (layer ",(string i),", ",(string k),")";
    if[not (.qs.layer.shape[k][s;s`inShape])~s`outShape; '"wrong output shape recorded",e];
    if[i>0; if[not s[`inShape]~L[i-1;1;`outShape]; '"input doesn't match the layer before",e]];
    if[not (count P i)=count .qs.layer.names k; '"wrong number of parameter arrays",e];
    .qs.layer.check[k][s;.qs.pdict[k;P i]]
 }

/ ---------- running a model ----------

/ the layer sequence run on a batch, keeping each layer's cache for backpropagation. t is each layer's training
/ input (see .qs.layer): returns (output;caches)
.qs.fwdStep:{[L;P;t;s;i]
    k:L[i;0]; r:.qs.layer.forward[k][L[i;1];.qs.pdict[k;P i];s 0;t i];
    (r 0;s[1],enlist r 1)
 }
.qs.forwardAll:{[m;X;t] .qs.fwdStep[m`layers;m`params;t]/[(X;());til count m`layers]}

/ the output of the first upto layers for batch X, not training and keeping nothing but the activations
.qs.inferStep:{[L;P;A;i] k:L[i;0]; first .qs.layer.forward[k][L[i;1];.qs.pdict[k;P i];A;0b]}
.qs.infer:{[m;X;upto] .qs.inferStep[m`layers;m`params]/[X;til upto]}

/ conv and GRU models work through bounded blocks of rows: their intermediates grow with rows x length
.qs.blocked:{[m] any (.qs.kinds m) in `conv1d`gru}

/ rows per block when predicting with a conv or GRU model
.qs.predictBlock:128

/ row indices split into one shard per secondary thread, each at least .qs.minShard rows (one shard if threads are off)
.qs.minShard:32
.qs.threads:{system "s"}
.qs.shards:{[n] p:1|.qs.threads[]&n div .qs.minShard; ceiling[n%p] cut til n}

/ the output of the first upto layers, for a batch matrix X or a single example vector; large batches are spread
/ over threads
.qs.run:{[m;X;upto]
    m:.qs.norm m;
    v:0h<type X; X:"f"$$[v;enlist X;X];
    out:$[.qs.blocked m;
        {[m;upto;X] raze {[m;upto;X;i] .qs.infer[m;X i;upto]}[m;upto;X] each .qs.predictBlock cut til count X}[m;upto];
        {[m;upto;X] .qs.infer[m;X;upto]}[m;upto]];
    A:$[1<count i:.qs.shards count X; raze {[f;X;i] f X i}[out;X] peach i; out X];
    $[v;first A;A]
 }

/ model outputs for a batch matrix X, or for a single example vector
.qs.predict:{[m;X] .qs.run[m;X;count (.qs.norm m)`layers]}

/ the output layer's input to its activation (the logit, for a sigmoid output), for a batch matrix X: ranking by
/ it keeps the order among examples whose probabilities all round to 1
.qs.predictRaw:{[m;X] m:.qs.norm m; n:count m`layers; .qs.run[m;X;n-`act=last .qs.kinds m]}

/ ---------- gradients ----------

/ each layer's training input for a batch of n rows: dropout layers get a mask, drawn here once for the whole batch
/ (in layer order, so seeded runs repeat), the others 1b
.qs.aux:{[m;n] {[L;n;i] $[`dropout=L[i;0]; .qs.dropMask[n;L[i;1;`rate];prd L[i;1;`inShape]]; 1b]}[m`layers;n] each til count m`layers}

/ the rows i of each matrix training input (a dropout mask); others are left alone
.qs.sliceAux:{[t;i] {[i;x] $[0h=type x; x i; x]}[i] each t}

/ model m with dropout at rate p after each hidden dense layer's activation (not the output), as the dropout
/ training option asks, and which of its layers were in m
.qs.withDropout:{[m;p]
    L:m`layers; P:m`params; K:first each L; nl:count K;
    if[p=0; :(m;nl#1b)];
    h:(K=`act) and 0b,-1_K=`dense;
    h:h and (til nl)<nl-1;
    L2:raze {[L;h;p;i] $[h i; (L i;(`dropout;`rate`inShape`outShape!(p;L[i;1;`outShape];L[i;1;`outShape]))); enlist L i]}[L;h;p] each til nl;
    P2:raze {[P;h;i] $[h i; (P i;()); enlist P i]}[P;h] each til nl;
    k:raze {[h;i] $[h i; 10b; enlist 1b]}[h] each til nl;
    (m,`layers`params!(L2;P2);k)
 }

/ one layer's backward step, run from the last layer to the first: s is (the gradient at the layer's output;
/ the gradients so far, last layer first)
.qs.bwdStep:{[L;P;C;s;i]
    k:L[i;0]; r:.qs.layer.backward[k][L[i;1];.qs.pdict[k;P i];C i;s 0;i>0];
    (r 0;s[1],enlist value r 1)
 }

/ gradient of loss over batch (X;Y), returns one list of arrays per layer shaped like the model's params
.qs.grad:{[m;X;Y;loss] .qs.gradDrop[m;X;Y;loss;0f]}

/ gradient with dropout rate p on the hidden dense layers (and any dropout layers the model has)
/ large batches are spread over threads, each shard's mean gradient weighted by its share of the rows
/ (not for norm, whose gradient depends on the whole batch at once); masks are drawn first so results are seeded
.qs.gradDrop:{[m;X;Y;loss;p]
    m:.qs.norm m;
    d:.qs.withDropout[m;p]; m2:d 0;
    t:.qs.aux[m2;count X];
    g:$[(loss=`norm) or 2>count i:.qs.shards count X; .qs.grad1[m2;X;Y;loss;t];
        (+/){[m;X;Y;loss;t;i] (count[i]%count X)*.qs.grad1[m;X i;Y i;loss;.qs.sliceAux[t;i]]}[m2;X;Y;loss;t] peach i];
    g where d 1
 }

/ gradient of loss over a single shard, with one training input per layer as for .qs.forwardAll
.qs.grad1:{[m;X;Y;loss;t]
    fw:.qs.forwardAll[m;X;t]; out:fw 0; C:fw 1;
    L:m`layers; P:m`params; nl:count L;
    la:`act=L[nl-1;0]; f:$[la; L[nl-1;1;`f]; `];
    pr:la and any (f;loss)~/:.qs.paired;
    if[la and (f=`softmax) and not pr; '"softmax output needs xent loss"];
    / a paired output activation and loss have the gradient (A-Y)/n at the activation's input
    dOut:$[pr; (out-Y)%count Y; .qs.loss.g[loss][out;Y]];
    top:nl-"j"$pr;
    r:.qs.bwdStep[L;P;C]/[(dOut;());reverse til top];
    (reverse r 1),("j"$pr)#enlist ()
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

/ the model's parameters: one list of arrays per layer
.qs.params:{[m] (.qs.norm m)`params}

/ weight decay: adds l2 times each decayed parameter (see .qs.layer.l2) to its gradient; the others are multiplied by 0
.qs.addL2:{[l2;m;g]
    K:.qs.kinds m; P:m`params;
    D:{[K;P;i] P[i]*"f"$.qs.layer.names[K i] in .qs.layer.l2 K i}[K;P] each til count K;
    g+l2*D
 }

/ one gradient step on batch rows b; s holds `model`state (and anything else, passed through)
.qs.step:{[o;X;Y;s;b]
    m:s`model;
    g:.qs.gradDrop[m;X b;Y b;o`loss;o`dropout];
    if[o[`l2]>0; g:.qs.addL2[o`l2;m;g]];
    r:.qs.opt.step[o`optimiser][o;s`state;g];
    m[`params]:m[`params]-r 0;
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
    if[not count[first X]=.qs.inputSize m; '"X has ",(string count first X)," columns, the model expects ",string .qs.inputSize m];
    if[not count[first Y]=.qs.outputSize m; '"Y has ",(string count first Y)," columns, the model outputs ",string .qs.outputSize m];
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
    s:`model`state`history`best`bestLoss`bestEpoch`wait`stop!(m;.qs.opt.init[o`optimiser] m`params;();m;0w;0N;0;0b);
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

/ one row per layer that has parameters: its inputs, outputs, activation and number of parameters. Activation and
/ pooling layers after it are folded into its row. A model of dense layers only has the columns layer, inputs,
/ outputs (counts), act and params; with conv or GRU layers there is a kind column too, and inputs and outputs are
/ shapes written channelsxlength
.qs.shapeStr:{[s] $[1=count s; string first s; "x" sv string s]}
.qs.summaryRow:{[L;P;K;a;b]
    idx:a+til b-a; ai:idx where K[idx]=`act; lastIdx:last idx where K[idx] in `dense`conv1d`gru`act`maxpool1d;
    s:L[a;1];
    (K a;$[K[a]=`gru; "j"$(s`channels;s`length); s`inShape];L[lastIdx;1;`outShape];
        $[K[a]=`gru; `tanh; count ai; L[first ai;1;`f]; `linear];sum {count raze x} each P a)
 }
.qs.summary:{[m]
    m:.qs.norm m; L:m`layers; P:m`params; K:first each L;
    hd:where K in `dense`conv1d`gru;
    r:flip .qs.summaryRow[L;P;K]'[hd;(1_hd),count K];
    kind:{$[x=`conv1d; `conv; x]} each r 0;
    $[all `dense=kind;
        flip `layer`inputs`outputs`act`params!(til count hd;first each r 1;first each r 2;r 3;r 4);
        flip `layer`kind`inputs`outputs`act`params!(til count hd;kind;.qs.shapeStr each r 1;.qs.shapeStr each r 2;r 3;r 4)]
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

