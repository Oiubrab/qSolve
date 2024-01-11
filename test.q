\l defineEquation.q

/system"python mnist_data_pull.py3 noshow";

$[1b;
    [
    system"c 5000 5000";
    system"P 0";
    weightsBiases:weightBiasGen[784;784 10];

    useNo:600;
    ratioConversion:0b;

    train_total:-1 + count system"ls eternalnightmare";
    test_total:-1 + count system"ls byinheritance";
    ratioConvert:til "j"$(train_total*useNo)%test_total;

    x_train:({(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":eternalnightmare/eternalnightmare",(string x),".txt")%255} each til -1 + count system"ls eternalnightmare")[til useNo];
    y_train:({this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":eternalnightmare/eternalnightmareY.txt")[0])[til useNo];

    x_test:({(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":byinheritance/byinheritance",(string x),".txt")%255} each til -1 + count system"ls byinheritance")[$[ratioConversion;ratioConvert;til useNo]];
    y_test:({this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":byinheritance/byinheritanceY.txt")[0])[$[ratioConversion;ratioConvert;til useNo]];

    model:backPropogation[weightsBiases;x_train;y_train;x_test;y_test;20;`model;enlist 1000.1]
    ];[

    weightsBiases:weightBiasGen[4;4 2];

    numOfEx:500000;
    trainingInput:{"f"$x} each {(x?2),'(x?2),'(x?2),'(x?2)}[numOfEx];
    trainingInput[5 * til floor (numOfEx%5)]:("j"$numOfEx%5)?(1 1 0 0f;0 0 1 1f);
    trainingExpected:{$[x~1 1 0 0f;1 0f;x~0 0 1 1f;0 1f;0 0f]} each trainingInput;

    testInput:{"f"$x} each {(x?2),'(x?2),'(x?2),'(x?2)}[numOfEx];
    testInput[5 * til floor (numOfEx%5)]:("j"$numOfEx%5)?(1 1 0 0f;0 0 1 1f);
    testExpected:{$[x~1 1 0 0f;1 0f;x~0 0 1 1f;0 1f;0 0f]} each testInput;

    model:backPropogation[weightsBiases;trainingInput;trainingExpected;testInput;testExpected;20;`model;enlist 0.1]
    ]
 ]
