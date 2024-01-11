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

/uses the gradient function to build a new grad from inputs
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

/runs the backpropogation on a model and saves it to disk/outputs a new model
backPropogation:{[weightsBiases;trainingInput;trainingExpected;testInput;testExpected;trainGrouping;modelFileName;scale]

    / get the timer going
    `oldTimer set .z.Z;

    / set scales because it's 00:42 and I want some sleep
    `scales set scale;

    / base the number of groups used on the number of training examples
    noOfGroups:floor (count trainingExpected)%trainGrouping;

    /note: if the trainGrouping number is not an integer factor of the number of examples, the function will omit the remainder
    trainingInput:trainingInput[til noOfGroups*trainGrouping];
    trainingExpected:trainingExpected[til noOfGroups*trainGrouping];

    /note: the testGrouping number is calculated from the trainGrouping number and will also leave out a remainder number of cases
    testGrouping:floor (count testExpected)%noOfGroups;
    testInput:testInput[til noOfGroups*testGrouping];
    testExpected:testExpected[til noOfGroups*testGrouping];

    /separates two dimensional lists (i.e becomes 3D) into groups of grp
    grouper:{[twoDim;grp] {x[z+til y]}[twoDim;grp;] each grp*til "j"$(count twoDim)%grp};

    newWeightsBiasesWithMinDiff:{[modelAndMeta;trainingInput;trainingExpected;testInput;testExpected]

        gradient:gradBuild[modelAndMeta;trainingInput;trainingExpected];

        diff:{
            res:useModel[y[0] - x*y[1];] each z[0];
            avg (sum each abs res - z[1])
        }[;(modelAndMeta;gradient);(testInput;testExpected)];

        diffs:diff each scales;

        show " ";
        show "t"$ ("t"$.z.Z) - "t"$oldTimer;
        `oldTimer set .z.Z;
        show "Min Diff:";
        show min diffs;
        show "Scale:";
        show scales[first where (min diffs)=diffs];
        show "Gradient:";
        show gradient;
        show "model update:";
        show modelAndMeta - scales[first where (min diffs)=diffs] * gradient;

        modelAndMeta - scales[first where (min diffs)=diffs] * gradient


    }/[weightsBiases;grouper[trainingInput;trainGrouping];grouper[trainingExpected;trainGrouping];grouper[testInput;testGrouping];grouper[testExpected;testGrouping]];
    (hsym modelFileName) set newWeightsBiasesWithMinDiff;
    newWeightsBiasesWithMinDiff
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
