 / building the solver

\l sigmoid.q

/ takes a set of weights and biases and feeds through input
forwardEquate:{[weightsBiases;inputs] {sigmoid z + y mmu x}/[inputs;weightsBiases`weight;weightsBiases`bias]}

/ generates a set of random weights and biases
weightBiasGen:{
    gen:{0.1*x?10};
    / start with first layer
    weight:enlist gen each x[0]#x[0];
    weight,:{y each x[z]#x[z-1]}[x;gen;] each 1 + til -1 + count x;
    bias:gen each x;
    `weight`bias!(weight;bias)
 }
