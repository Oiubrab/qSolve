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
    common:mGoras[(last results) - expected];

    pythagoreanFactors:common * (last results) - expected;
    resultInput:(enlist inputs) , -1_results;
    sigmoidFactors:mSig linear[resultInput;weightsBiases`weight;weightsBiases`bias];

    / for the result nodes, the weight and bias derivatives are just the straight result direction
    gradOutputWeight:((count last sigmoidFactors)#enlist last resultInput)*(last sigmoidFactors)*pythagoreanFactors;
    gradOutputBias:(last sigmoidFactors)*pythagoreanFactors;
    grad:{[weights;sigFactors;pyFactors;rezza]
        /will need to build all the weight grad scalars
        index:indexWeightGen[weights];
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
        }[weights;sigFactors;pyFactors;rezza;] each index;
        weightGrad:{x[y[0];y[1];y[2]]:y[3];x}/[weights;index,'weightGradList];
    }[weightsBiases`weight;sigmoidFactors;pythagoreanFactors;resultInput]
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
indexWeightGen:{
    structure:count each x;
    combination:{,[x;] each raze({,[x;] each til y}[;z] each til y)}[0;first structure;first structure];
    combination,:raze{{,[x;] each raze({,[x;] each til y}[;z] each til y)}[x;y[x];y[x-1]]}[;structure] each 1 + til -1 + count structure;
    combination
 }
