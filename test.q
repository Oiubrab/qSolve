\l defineEquation.q

weightsBiases:weightBiasGen[2;2 2];

numOfEx:100000;
testInput:{(0.1*x?10),'(0.1*x?10)}[numOfEx];
testExpected:?[testInput[til count testInput - 1;0]>testInput[til count testInput - 1;1];numOfEx#enlist(1.0 0.0);numOfEx#enlist(0.0 1.0)];

newWeightsBiases:backPropogation/[weightsBiases;testInput;testExpected;0.0001]


normalised:{[model;input]
    points:0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
    top:max useModel[model;] each points cross points;
    bottom:min useModel[model;] each points cross points;
    range:top - bottom;
    above:useModel[model;input] - bottom;
    above*(1%range)
 }
