\l defineEquation.q

system"python mnist_data_pull.py3 noshow";

weightsBiases:weightBiasGen[784;784 10];

useNo:60;

train_total:-1 + count system"ls eternalnightmare";
test_total:-1 + count system"ls byinheritance";

x_train:({(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":eternalnightmare/eternalnightmare",(string x),".txt")%255} each til -1 + count system"ls eternalnightmare")[til useNo];
y_train:({this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":eternalnightmare/eternalnightmareY.txt")[0])[til useNo];

x_test:({(raze flip ("JJJJJJJJJJJJJJJJJJJJJJJJJJJJ";4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 3) 0: `$":byinheritance/byinheritance",(string x),".txt")%255} each til -1 + count system"ls byinheritance")[til "j"$(train_total*useNo)%test_total];
y_test:({this:10#0f;this[x]:1f;this} each ((enlist "J";enlist 1) 0: `$":byinheritance/byinheritanceY.txt")[0])[til "j"$(train_total*useNo)%test_total];

model:backPropogation[weightsBiases;x_train;y_train;x_test;y_test;12;`model]
