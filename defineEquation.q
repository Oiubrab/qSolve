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

backPropogation:{[weightsBiases;inputs;expected;scaling]
    /calculate preliminary factors
    results:{sigmoid linear[x;y;z]}\[inputs;weightsBiases`weight;weightsBiases`bias];
    common:mGoras[(last results) - expected];
    /calculate common factors
    pythagoreanFactors:common * (last results) - expected;
    resultInput:(enlist inputs) , -1_results;
    sigmoidFactors:mSig linear[resultInput;weightsBiases`weight;weightsBiases`bias];

    /calculate the grad
    grad:{[weightsBiases;sigFactors;pyFactors;rezza]
        /will need to build all the weight grad scalars
        index:indexGen[weightsBiases];
        /calculate the grad for the weights
        weightGradList:{[weights;sigFactors;pyFactors;rezza;index]
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
        }[weightsBiases`weight;sigFactors;pyFactors;rezza;] each index`weight;
        weightGrad:{x[y[0];y[1];y[2]]:y[3];x}/[weightsBiases`weight;(index`weight),'weightGradList];

        /calculate the grad for the biases
        biasGradList:{[weights;sigFactors;pyFactors;index]
            $[index[0]=-1 + count weights;
                (last sigFactors)[index[1]]*pyFactors[index[1]]
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
        }[weightsBiases`weight;sigFactors;pyFactors;] each index`bias;
        biasGrad:{x[y[0];y[1]]:y[2];x}/[weightsBiases`bias;(index`bias),'biasGradList];
        /build the grad dic
        `weight`bias!(weightGrad;biasGrad)
    }[weightsBiases;sigmoidFactors;pythagoreanFactors;resultInput];

    /use the grad to produce a new set of weights and biases that should, theoretically, be closer to the global minimum
    weightsBiases - scaling*grad
 }

/ generates a set of random weights and biases
weightBiasGen:{
    gen:{0.1 + 0.1*x?10};
    / start with first layer
    weight:enlist gen each x[0]#x[0];
    weight,:{y each x[z]#x[z-1]}[x;gen;] each 1 + til -1 + count x;
    bias:gen each x;
    `weight`bias!(weight;bias)
 }

/ generate combinations of all indexes for the 3D weight list
indexGen:{
    structure:count each x`weight;
    weightCombination:{,[x;] each raze({,[x;] each til y}[;z] each til y)}[0;first structure;first structure];
    weightCombination,:raze{{,[x;] each raze({,[x;] each til y}[;z] each til y)}[x;y[x];y[x-1]]}[;structure] each 1 + til -1 + count structure;
    biasCombination:raze {{(x;y)}[x[0];] each 1_x} each (til count structure),'til each structure;
    `weight`bias!(weightCombination;biasCombination)
 }
