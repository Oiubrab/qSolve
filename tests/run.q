/ test runner: q tests/run.q [-only substring]
/ loads the library, every tests/test_*.q, runs each registered test and exits 1 on any failure

.t.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
system "l ",.t.root,"/tests/assert.q";
system "l ",.t.root,"/defineEquation.q";
{if[not ()~key hsym `$.t.root,"/",x; system "l ",.t.root,"/",x]} each ("data.q";"plot.q";"fx.q");

.t.files:asc f where (f:key hsym `$.t.root,"/tests") like "test_*.q";
{system "l ",.t.root,"/tests/",string x} each .t.files;

.t.opt:.Q.opt .z.x;
.t.names:key .t.reg;
if[`only in key .t.opt; .t.names:.t.names where (string .t.names) like "*",(first .t.opt`only),"*"];

/ run one test in a trap, returning a result row; errors starting "skip: " mark a skipped test
.t.run1:{[name]
    t0:.z.p;
    r:@[{x[]; (`pass;"")};.t.reg name;{($["skip: "~6#x;`skip;`fail];x)}];
    `name`status`ms`message!(name;r 0;`long$(.z.p-t0)%1000000;r 1)
 }

system "c 500 200";
.t.results:.t.run1 each .t.names;
show .t.results;
-1 "";
.t.n:{sum x=.t.results`status};
-1 ", " sv {(string .t.n x)," ",string x} each `pass`fail`skip;
exit `int$`fail in .t.results`status
