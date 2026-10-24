/ decision boundary SVGs

.t.tmp:{[name] t:getenv`TMPDIR; `$":",$[""~t;"/tmp";t],"/",name}

.t.add[`plot_svg;{
    system "S 1"; d:.qs.data.split[.qs.data.moons[100;0.1];0.8];
    m:.qs.init[2;8 2;`tanh`softmax;1];
    f:.t.tmp "qsolve_plot.svg";
    .t.eq[.qs.plot2D[m;d;f;"moons"];f];
    s:read0 f;
    .t.ok[(first s) like "<svg*";"not an svg"];
    .t.eq[last s;"</svg>"];
    .t.eq["j"$sum s like "<rect*";1+.qs.plot.cells*.qs.plot.cells];
    .t.eq["j"$sum s like "<circle*";100];
    .t.eq["j"$sum s like "*fill='none'*";20];
    .t.ok[any s like "*moons*";"title missing"];
    hdel f;
 }];

.t.add[`plot_single_output;{
    system "S 1"; d:.qs.data.circles[50;0.1]; d[`Y]:enlist each d[`Y][;1];
    m:.qs.init[2;4 1;`tanh`sigmoid;1];
    f:.t.tmp "qsolve_plot1.svg";
    .qs.plot2D[m;d;f;"one output"];
    .t.eq["j"$sum (read0 f) like "<circle*";50];
    hdel f;
 }];

.t.add[`plot_needs_2_inputs;{
    .t.throws[{.qs.plot2D[.qs.init[3;2;`softmax;1];.qs.data.gate`xor;.t.tmp "x.svg";""]};::];
 }];
