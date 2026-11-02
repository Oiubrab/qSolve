/ test runner: q tests/run.q [-only substring]
/ loads the library and the projects' libraries, every tests/test_*.q and projects/*/tests/test_*.q, runs each
/ registered test and exits 1 on any failure

.t.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
system "l ",.t.root,"/tests/assert.q";
system "l ",.t.root,"/src/defineEquation.q";
{system "l ",.t.root,"/src/",x} each ("data.q";"plot.q";"signal.q");
.t.projects:asc key hsym `$.t.root,"/projects";
/ each project's library is .qs.<project> (projects/gw/src/gw.q is .qs.gw), pointed at its own folder
{[p] d:.t.root,"/projects/",string p;
    {[d;f] system "l ",d,"/src/",string f}[d] each asc f where (f:key hsym `$d,"/src") like "*.q";
    @[.Q.dd[`.qs;p];`home;:;d]} each .t.projects;

/ the base tests, then each project's
.t.dirs:enlist[.t.root,"/tests"],{.t.root,"/projects/",(string x),"/tests"} each .t.projects;
.t.files:raze {[d] d,/:"/",'string asc f where (f:key hsym `$d) like "test_*.q"} each .t.dirs;
{system "l ",x} each .t.files;

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
if[not count .t.names; -1 "no tests match"; exit 1];
.t.results:.t.run1 each .t.names;
show .t.results;
-1 "";
.t.n:{sum x=.t.results`status};
-1 ", " sv {(string .t.n x)," ",string x} each `pass`fail`skip;
exit `int$`fail in .t.results`status
