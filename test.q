\l defineEquation.q

weightsBiases: weightBiasGen[2;2 2];

backPropogation:{[wb]
    numOfEx:100;
    trainingInput:{(0.1*x?10),'(0.1*x?10)}[numOfEx];
    trainingExpected:?[trainingInput[til count trainingInput - 1;0]>trainingInput[til count trainingInput - 1;1];numOfEx#enlist(1.0 0.0);numOfEx#enlist(0.0 1.0)];

    testInput:{(0.1*x?10),'(0.1*x?10)}[numOfEx];
    testExpected:?[testInput[til count testInput - 1;0]>testInput[til count testInput - 1;1];numOfEx#enlist(1.0 0.0);numOfEx#enlist(0.0 1.0)];

    scales: 0.001 * til 100000;

    gradient:gradBuild[wb;trainingInput;trainingExpected];

    diff:{
        res:useModel[y[0] - x*y[1];] each z[0];
        avg (sum each abs res - z[1])
    }[;(wb;gradient);(testInput;testExpected)];

    diffs:diff each scales;
    (min diffs;wb - scales[first where (min diffs)=diffs] * gradient)
 }

/ normalised:{[model;input]
/     points:0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
/     top:max useModel[model;] each points cross points;
/     bottom:min useModel[model;] each points cross points;
/     range:top - bottom;
/     above:useModel[model;input] - bottom;
/     above*(1%range)
/  }
