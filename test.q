\l defineEquation.q

weightsBiases:weightBiasGen[784;10 10];

useNo:60;

train_total:-1 + count system"ls eternalnightmare";
test_total:-1 + count system"ls byinheritance";

x_train:({(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":eternalnightmare/eternalnightmare",(string x),".txt")%255} each til -1 + count system"ls eternalnightmare")[til useNo];
y_train:({this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":eternalnightmare/eternalnightmareY.txt")[0])[til useNo];

x_test:({(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":byinheritance/byinheritance",(string x),".txt")%255} each til -1 + count system"ls byinheritance")[til "j"$(train_total*useNo)%test_total];
y_test:({this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":byinheritance/byinheritanceY.txt")[0])[til "j"$(train_total*useNo)%test_total];

newWeightsBiases:backPropogation[weightsBiases;x_train[0];y_train[0];10000.0];
useModel[newWeightsBiases;x_train[0]]

