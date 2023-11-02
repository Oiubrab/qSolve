 / building the solver

/ sigmoid definition
sigmoid:{reciprocal[1+exp[-1.0*x]]}

/ sigmoid gradient factor
mSig:{exp[x]%xexp[1+exp[x];2]}

/ build a linear function
linear:{z + y mmu x};

/ common pythagorean gradient factor
mGoras:{reciprocal[sqrt sum xexp[;2] x]}

/ takes a set of weights and biases and feeds through input
useModel:{[weightsBiases;inputs] {sigmoid linear[x;y;z]}/[inputs;weightsBiases`weight;weightsBiases`bias]}

backPropogation:{[weightsBiases;inputs;expected]
    results:{sigmoid linear[x;y;z]}\[inputs;weightsBiases`weight;weightsBiases`bias];
    common:mGoras[results[-1 + count results] - expected];

    pythagoreanFactors:common * results[-1 + count results] - expected;
    resultInput:(enlist inputs) , results[til -1 + count results];
    sigmoidFactors:mSig linear[resultInput;weightsBiases`weight;weightsBiases`bias];

    / for the result nodes, the weight and bias derivatives are just the straight result direction

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
