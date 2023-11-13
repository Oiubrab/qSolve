\l defineEquation.q

weightsBiases:weightBiasGen 2 5 3 2;

numOfEx:10000;
testInput:{(0.01*x?100),'(0.01*x?100)}[numOfEx];
testExpected:?[testInput[til count testInput - 1;0]>testInput[til count testInput - 1;1];numOfEx#enlist(1.0 0.0);numOfEx#enlist(0.0 1.0)];

newWeightsBiases:backPropogation/[weightsBiases;testInput;testExpected;0.2]
