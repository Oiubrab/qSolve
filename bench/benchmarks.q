/ train every benchmark dataset (plans/completed/improvemet_plan.md section 5, T3-T12) and check it reaches its target
/ q bench/benchmarks.q [-only name] [-skipLarge]
/ datasets that haven't been downloaded are skipped; exits 1 if any benchmark misses its target

.b.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
system "l ",.b.root,"/defineEquation.q";
system "l ",.b.root,"/data.q";
.qs.data.dir:.b.root,"/data";

/ one benchmark: dataset, model, training options and target
/ metric is a key of .qs.evaluate on the validation data: loss passes at or below target, acc and r2 at or above
.b.base:`name`data`n`noise`hidden`act`out`loss`opts`metric`target!(`;`;1000;0.1;enlist 16;`tanh;`softmax;`xent;()!();`acc;1f);
.b.case:{.b.base,x};
.b.cases:.b.case each (
    `name`data`hidden`out`loss`opts!(`and;`and;`long$();`sigmoid;`bce;`epochs`batch`lr!(300;4;1f));
    `name`data`hidden`out`loss`opts!(`or;`or;`long$();`sigmoid;`bce;`epochs`batch`lr!(300;4;1f));
    `name`data`hidden`out`loss`opts!(`nand;`nand;`long$();`sigmoid;`bce;`epochs`batch`lr!(300;4;1f));
    `name`data`hidden`out`loss`opts!(`xor;`xor;enlist 4;`sigmoid;`bce;`epochs`batch`lr!(300;4;0.5));
    `name`data`n`hidden`act`opts!(`fourBit;`fourBit;2000;enlist 8;`relu;`epochs`batch`lr`optimiser!(30;32;0.01;`adam));
    `name`data`n`out`loss`opts`metric`target!(`sine;`sine;1000;`linear;`mse;`epochs`batch`lr`optimiser!(300;32;0.01;`adam);`loss;0.001);
    `name`data`n`hidden`opts`target!(`circles;`circles;1000;enlist 8;`epochs`batch`lr`optimiser!(100;32;0.01;`adam);0.98);
    `name`data`n`opts`target!(`moons;`moons;1000;`epochs`batch`lr`optimiser!(100;32;0.01;`adam);0.97);
    `name`data`n`noise`hidden`opts`target!(`spirals;`spirals;2000;0.02;32 32;`epochs`batch`lr`optimiser!(300;32;0.005;`adam);0.95);
    `name`data`n`hidden`out`loss`opts`metric`target!(`houses;`houses;2000;enlist 32;`linear;`mse;`epochs`batch`lr`optimiser!(100;32;0.01;`adam);`r2;0.95);
    `name`data`n`hidden`out`loss`opts`metric`target!(`seasonal;`seasonal;3000;enlist 16;`linear;`mse;`epochs`batch`lr`optimiser!(100;32;0.001;`adam);`r2;0.85);
    `name`data`hidden`opts`target!(`iris;`iris;enlist 8;`epochs`batch`lr`optimiser!(100;16;0.01;`adam);0.93);
    `name`data`hidden`act`opts`target!(`mnist;`mnist;enlist 128;`relu;`epochs`batch`lr`optimiser!(5;64;0.001;`adam);0.96);
    `name`data`hidden`act`opts`target!(`fashion;`fashion;enlist 128;`relu;`epochs`batch`lr`optimiser!(5;64;0.001;`adam);0.84));

.b.large:`mnist`fashion;
.b.files:`iris`mnist`fashion!("iris/iris.data";"mnist/train-images-idx3-ubyte";"fashion/train-images-idx3-ubyte");

/ run one benchmark, returning a result row
.b.run:{[c]
    r:`name`status`metric`value`target`epochs`seconds!(c`name;`skip;c`metric;0n;c`target;0N;0n);
    if[c[`data] in key .b.files; if[()~key hsym `$.qs.data.dir,"/",.b.files c`data; :r]];
    system "S 42";
    d:.qs.data.load[c`data;c`n;c`noise];
    m:.qs.init[count first d`X;c[`hidden],count first d`Y;((count c`hidden)#c`act),c`out;42];
    t0:.z.p;
    t:.qs.train[m;d;(`loss`verbose!(c`loss;0)),c`opts];
    v:.qs.evaluate[t`model;d`Xval;d`Yval;c`loss] c`metric;
    ok:$[c[`metric]=`loss; v<=c`target; v>=c`target];
    r,`status`value`epochs`seconds!($[ok;`pass;`fail];v;count t`history;1e-9*.z.p-t0)
 };

.b.o:.Q.opt .z.x;
.b.todo:.b.cases;
if[`only in key .b.o; .b.todo:select from .b.todo where name in `$.b.o`only];
if[`skipLarge in key .b.o; .b.todo:select from .b.todo where not name in .b.large];
system "c 500 200";
.b.results:();
{-1 (string x`name),": ",-3!value 2_r:.b.run x; .b.results,:enlist r} each .b.todo;
-1 "";
show .b.results;
-1 ", " sv {(string sum x=.b.results`status)," ",string x} each `pass`fail`skip;
exit `int$`fail in .b.results`status
