/ serve.q end to end: start a server process, query it over IPC, then stop it

.t.add[`serve_ipc;{
    d:getenv[`TMPDIR]; d:$[""~d;"/tmp";d]; f:d,"/qsolve_serve.qsm";
    system "S 4"; data:.qs.data.fromTable[.qs.data.houses 300;();`price;0.8];
    r:.qs.train[.qs.init[7;8 1;`tanh`linear;1];data;`loss`optimiser`lr`epochs`verbose!(`mse;`adam;0.01;20;0)];
    m:r`model; m[`enc]:data`enc; .qs.save[hsym `$f;m];
    p:30000+.z.i mod 20000;
    system "env ",(first .z.X)," ",.t.root,"/serve.q -q -model ",f," -p ",(string p)," </dev/null >/dev/null 2>&1 &";
    / wait up to about 5 s for the port to open
    h:{[p;x] if[not null x; :x]; system "env sleep 0.1"; @[hopen;(`$"::",string p;100);0Ni]}[p]/[50;0Ni];
    if[null h; '"server did not start"];
    res:.Q.trp[{[h;m;data;u]
        i:h (`info;::);
        .t.eq[i`target;`price];
        .t.eq[i[`layers]`params;64 9];
        t:([] area:120 80f; rooms:3 2; age:10 0n; city:`north`zzz);
        y:h (`predictTable;t);
        .t.eq[cols y;`area`rooms`age`city`prediction];
        .t.near[y`prediction;.qs.data.decodeY[m`enc;.qs.predict[m;.qs.data.encodeX[m`enc;t]]];1e-9];
        .t.near[h (`predict;data`Xval);.qs.predict[m;data`Xval];1e-12];
        .t.throws[h;"2+2"];
        .t.throws[h;(`system;"ls")];
        .t.throws[h;(`predict;1 2f)];
        .t.throws[h;(`predictTable;delete city from t)];
        i`pid}[h;m;data];::;{(`err;x,"\n",.Q.sbt y)}];
    pid:$[`err~first res; (h (`info;::))`pid; res];
    hclose h; system "kill ",string pid; hdel hsym `$f;
    if[`err~first res; e:res 1; 'e];
 }];
