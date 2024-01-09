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

/calculate the grad
grad:{[weightsBiases;sigFactors;pyFactors;rezza;index]
    /calculate the grad for the weights
    weightGradList:{[weights;sigFactors;pyFactors;rezza;index]
        sum { [weights;index;factors]
            sigFactors:factors[0];
            pyFactors:factors[1];
            rezza:factors[2];
            $[index[0]=-1 + count weights;
                (last rezza)[index[2]]*(last sigFactors)[index[1]]*pyFactors[index[1]];
                index[0]=-2 + count weights;
                [
                    bottomGrad:rezza[index[0];index[2]]*sigFactors[index[0];index[1]];
                    sum[bottomGrad*weights[index[0]+1;til count weights[index[0]+1];index[1]]*sigFactors[index[0]+1]*pyFactors]
                ];
                [
                    bottomGrad:rezza[index[0];index[2]]*sigFactors[index[0];index[1]];
                    secondBottomGrad:bottomGrad*weights[index[0]+1;til count weights[index[0]+1];index[1]]*sigFactors[index[0]+1];
                    finalSigGrad:{z*sum each y*(count y)#enlist x}/[secondBottomGrad;weights[(index[0] + 2) + til (count weights) - 2 + index[0]];sigFactors[(index[0] + 2) + til (count weights) - 2 + index[0]]];
                    sum[finalSigGrad*pyFactors]
                ]
            ]
        }[weights;index;] each ((enlist each sigFactors) ,' (enlist each pyFactors) ,' (enlist each rezza))
    }[weightsBiases`weight;sigFactors;pyFactors;rezza;] peach index`weight;
    weightGrad:{x[y[0];y[1];y[2]]:y[3];x}/[weightsBiases`weight;(index`weight),'weightGradList];

    /calculate the grad for the biases
    biasGradList:{[weights;sigFactors;pyFactors;index]
        sum { [weights;index;factors]
            sigFactors:factors[0];
            pyFactors:factors[1];
            rezza:factors[2];
            $[index[0]=-1 + count weights;
                (last sigFactors)[index[1]]*pyFactors[index[1]];
            index[0]=-2 + count weights;
                [
                    bottomGrad:sigFactors[index[0];index[1]];
                    sum[bottomGrad*weights[index[0]+1;til count weights[index[0]+1];index[1]]*sigFactors[index[0]+1]*pyFactors]
                ];
                [
                    bottomGrad:sigFactors[index[0];index[1]];
                    secondBottomGrad:bottomGrad*weights[index[0]+1;til count weights[index[0]+1];index[1]]*sigFactors[index[0]+1];
                    finalSigGrad:{z*sum each y*(count y)#enlist x}/[secondBottomGrad;weights[(index[0] + 2) + til (count weights) - 2 + index[0]];sigFactors[(index[0] + 2) + til (count weights) - 2 + index[0]]];
                    sum[finalSigGrad*pyFactors]
                ]
            ]
        }[weights;index;] each ((enlist each sigFactors) ,' (enlist each pyFactors))
    }[weightsBiases`weight;sigFactors;pyFactors;] peach index`bias;
    biasGrad:{x[y[0];y[1]]:y[2];x}/[weightsBiases`bias;(index`bias),'biasGradList];
    /build the grad dic
    `weight`bias!(weightGrad;biasGrad)
 }

gradBuild:{[weightsBiases;inputs;expected]
    /calculate preliminary factors
    results:{sigmoid linear[x;y;z]}\[;weightsBiases`weight;weightsBiases`bias] each inputs;
    common:mGoras[raze (last each results) - expected];
    /calculate common factors
    pythagoreanFactors:common * (last each results) - expected;
    resultInput:(enlist each inputs) ,' _[-1;] each results;
    sigmoidFactors:mSig linear[;weightsBiases`weight;weightsBiases`bias] each resultInput;

    /will need to build all the weight grad scalars
    indexing:indexGen[count first inputs;weightsBiases];

    /produce the grad
    grad[weightsBiases;sigmoidFactors;pythagoreanFactors;resultInput;indexing]
 }

/ generates a set of random weights and biases
weightBiasGen:{[noOfInputs;nodes]
    gen:{0.1 + 0.1*x?10};
    / start with first layer
    weight:enlist gen each nodes[0]#noOfInputs;
    weight,:{y each x[z]#x[z-1]}[nodes;gen;] each 1 + til -1 + count nodes;
    bias:gen each nodes;
    `weight`bias!(weight;bias)
 }

/ generate combinations of all indexes for the 3D weight list
indexGen:{[noOfInputs;weightBias]
    structure:count each weightBias`weight;
    weightCombination:{,[x;] each raze({,[x;] each til y}[;z] each til y)}[0;first structure;noOfInputs];
    weightCombination,:raze{{,[x;] each raze({,[x;] each til y}[;z] each til y)}[x;y[x];y[x-1]]}[;structure] each 1 + til -1 + count structure;
    biasCombination:raze {{(x;y)}[x[0];] each 1_x} each (til count structure),'til each structure;
    `weight`bias!(weightCombination;biasCombination)
 }
