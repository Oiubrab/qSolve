/ l2 sensitivity of the linear model in bench/fx_trends.q: is its negative skill just too little shrinkage?
/ q bench/fx_trends_l2.q [-s 4]       linear model, l2 in 1 and 10, 6 pairs x horizons 20 60 120, same walk-forward

.fxl.root:{$[count x;x;"."]} "/" sv -2_ "/" vs string .z.f;
{system "l ",.fxl.root,"/",x} each ("defineEquation.q";"data.q";"fx.q");
system "c 100 250";
.fxl.t:.qs.fx.ecb `$":",.fxl.root,"/data/fx/eurofxref-hist.csv"; .fxl.dates:.fxl.t`date;
.fxl.years:2010+til 1+(`year$last .fxl.dates)-2010;
.fxl.one:{[pair;h;l2]
    r:.qs.fx.rate[.fxl.t;pair]; rows:.qs.fx.rows[.qs.fx.features[r;.fxl.dates];.qs.fx.returns r;h];
    spec:`optimiser`lr`epochs`batch`seed`verbose`out`hidden`loss`l2!(`adam;0.001;30;64;42;0;`linear;`long$();`mse;l2);
    system "S 42"; w:.qs.fx.walkRows[rows;.fxl.dates;.fxl.years;h;0N;::;spec];
    s:.qs.fx.summary[w`p;w`y;20|h;500];
    `pair`h`l2`skill`skillLo`skillHi`pNoSkill`ic!(pair;h;l2;s`skill;s`skillLo;s`skillHi;s`pNoSkill;s`ic)
 };
.fxl.args:raze {[ph] ph,/:1 10f} each `EURUSD`GBPUSD`USDJPY`USDCHF`AUDUSD`USDCAD cross 20 60 120;
.fxl.res:.fxl.one .' .fxl.args;
show .fxl.res;
show select skill:avg skill,ic:avg ic,positive:sum skill>0,sig:sum (skill>0) and pNoSkill<=0.05 by h,l2 from .fxl.res;
exit 0
