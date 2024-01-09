\l defineEquation.q

/system"python mnist_data_pull.py3 noshow";

weightsBiases:weightBiasGen[784;30 20 10];

x_train:{(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":eternalnightmare/eternalnightmare",(string x),".txt")%255} each til -1 + count system"ls eternalnightmare";
y_train:{this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":eternalnightmare/eternalnightmareY.txt")[0];

x_test:{(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":byinheritance/byinheritance",(string x),".txt")%255} each til -1 + count system"ls byinheritance";
y_test:{this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":byinheritance/byinheritanceY.txt")[0];

backPropogation:{[weightsBiases;trainingInput;trainingExpected;testInput;testExpected;trainGrouping;modelFileName]

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
        scales: 0.1 * til 1000;

        gradient:gradBuild[modelAndMeta[0];trainingInput;trainingExpected];

        diff:{
            res:useModel[y[0] - x*y[1];] each z[0];
            avg (sum each abs res - z[1])
        }[;(modelAndMeta[0];gradient);(testInput;testExpected)];

        diffs:diff each scales;
        $[modelAndMeta[1] > min diffs;
            (modelAndMeta[0] - scales[first where (min diffs)=diffs] * gradient;min diffs);
            modelAndMeta
        ]
    }/[(weightsBiases;1f);grouper[trainingInput;trainGrouping];grouper[trainingExpected;trainGrouping];grouper[testInput;testGrouping];grouper[testExpected;testGrouping]]
    (hsym modelFileName) set newWeightsBiasesWithMinDiff[0];
    newWeightsBiasesWithMinDiff
 }
