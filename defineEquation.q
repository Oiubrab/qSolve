 / building the solver

/ sigmoid definition
sigmoid:{1%1+exp[-1.0*x]}

/ sigmoid gradient factor
mSig:{exp[x]%1+xexp[x;2]}

/ takes a set of weights and biases and feeds through input
equate:{[weightsBiases;inputs;direction]
    linearly:{sigmoid z + y mmu x};
    $[direction=`forward;
        linearly/[inputs;weightsBiases`weight;weightsBiases`bias];
        direction=`backward;
        [
            results:linearly\[inputs;weightsBiases`weight;weightsBiases`bias]
        ];
        "Direction must be either `forward or `backward"
    ]
 }

/ generates a set of random weights and biases
weightBiasGen:{
    gen:{0.1*x?10};
    / start with first layer
    weight:enlist gen each x[0]#x[0];
    weight,:{y each x[z]#x[z-1]}[x;gen;] each 1 + til -1 + count x;
    bias:gen each x;
    `weight`bias!(weight;bias)
 }
