weightsBiases:weightBiasGen 2 3 2;

numOfEx:100
testInput:{(0.01*x?100),'(0.01*x?100)}[numOfEx]
testExpected:?[testInput[til count testInput - 1;0]>testInput[til count testInput - 1;1];numOfEx#enlist(1.0 0.0);numOfEx#enlist(0.0 1.0)]
