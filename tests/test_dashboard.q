/ dashboard.q: the WebSocket API through its handler (replies captured by a stub sender), then HTTP and
/ IPC against a real dashboard process

.dash.root:.t.root;
system "l ",.t.root,"/dashboard.q";
.dash.sent:();
.dash.send:{[msg] .dash.sent,:enlist .j.j .dash.clean .dash.reqId,msg};
/ send one request, return every reply as parsed JSON
.t.ask:{[r] .dash.sent::(); .dash.handle .j.j r; .j.k each .dash.sent}
.t.types:{[m] `$m@\:`type}

.t.add[`dash_hello;{
    m:.t.ask `cmd`id!("hello";1);
    .t.eq[count m;1];
    .t.eq[m[0]`type;"hello"];
    .t.eq[m[0]`id;1f];
    .t.ok[all (string .qs.data.names) in m[0][`datasets]@\:`name;"datasets missing"];
 }];

.t.add[`dash_train_stream;{
    m:.t.ask `cmd`data`hidden`epochs`lr!("train";"moons";enlist 8;4;0.01);
    .t.eq[.t.types m;`start`epoch`epoch`epoch`epoch`done];
    d:last m;
    .t.eq[`int$(m 1)[`row]`epoch;1i];
    .t.eq[count d`history;4];
    .t.eq[(d`row)`arch;"2-8-2 tanh softmax"];
    .t.eq[count d`confusion;2];
    .t.eq[d[`plot]`n;60f];
    .t.eq[count d[`plot]`classes;3600];
    .t.ok[not null d[`val]`acc;"no validation accuracy"];
    .t.ok[(`long$d`run) in key .dash.model;"model not kept"];
 }];

.t.add[`dash_regression_and_predict;{
    m:.t.ask `cmd`data`rows`hidden`epochs`cv!("train";"houses";300;enlist 8;5;3);
    .t.eq[.t.types m;`start`cv,(5#`epoch),`done];
    d:last m; id:d`run;
    .t.eq[d[`row]`task;"regression"];
    .t.eq[count (m 1)`folds;3];
    .t.eq[`$(d`features)@\:`col;`area`rooms`age`city];
    .t.ok[d[`unitScale]>1;"no target scale"];
    p:.t.ask `cmd`run`rows!("predict";id;([] area:120 80f; rooms:3 2f; age:10 0n; city:("north";"zzz")));
    .t.eq[.t.types p;enlist `predicted];
    .t.eq[count p[0]`rows;2];
    .t.ok[all 100<p[0][`rows]@\:`prediction;"prices should be in their own units"];
 }];

.t.add[`dash_upload_series_and_table;{
    system "S 3";
    t:([] t:til 300; v:100+20*.qs.data.seasonal[300;24;0.1]; c:300?`a`b);
    m:.t.ask `cmd`name`csv!("upload";"my data.csv";"\n" sv csv 0: t);
    .t.eq[m[0]`name;"my_data_csv"];
    .t.eq[`$m[0][`columns]@\:`kind;`num`num`cat];
    s:.t.ask `cmd`data`series`lags`hidden`epochs!("train";"my_data_csv";"v";24;();3);
    .t.eq[last .t.types s;`done];
    .t.ok[not null last[s]`naiveMae;"no naive baseline"];
    c:.t.ask `cmd`data`target`features`epochs!("train";"my_data_csv";"c";("t";"v");3);
    .t.eq[last[c][`row]`task;"classes"];
    .t.eq[count last[c]`labels;2];
 }];

.t.add[`dash_search;{
    m:.t.ask `cmd`data`epochs`space!("search";"moons";3;`lr`hidden!(0.01 0.001;(enlist 4;8 8)));
    .t.eq[.t.types m;`searching`trial`trial`trial`trial`searched];
    r:last[m]`results;
    .t.eq[count r;4];
    .t.eq[r@\:`valLoss;asc r@\:`valLoss];
 }];

.t.add[`dash_save_load_forget;{
    id:last[.t.ask `cmd`data`hidden`epochs!("train";"xor";enlist 4;2)]`run;
    n:"qsolve_test_",string .z.i;
    s:.t.ask `cmd`run`name!("save";id;n,"/../x");
    .t.eq[s[0]`type;"saved"];
    .t.ok[not any "/." in s[0]`name;"unsafe name kept"];
    f:hsym `$.dash.modelDir,"/",s[0][`name],".qsm";
    .t.ok[not ()~key f;"model file missing"];
    l:.t.ask `cmd`name!("load";s[0]`name);
    .t.eq[l[0]`type;"loaded"];
    p:.t.ask `cmd`run`inputs!("predict";l[0]`run;enlist 1 0f);
    .t.eq[count first p[0]`outputs;1];
    .t.eq[.t.types .t.ask `cmd`run`inputs!("predict";l[0]`run;enlist 1 0 0f);enlist `error];
    g:.t.ask `cmd`run!("forget";l[0]`run);
    .t.ok[not (`long$l[0]`run) in key .dash.model;"run not forgotten"];
    .t.ok[(`long$id) in key .dash.model;"forget removed other runs"];
    hdel f;
 }];

/ bad requests come back as error messages and leave the server working
.t.add[`dash_errors;{
    .dash.sent::(); .dash.handle "not json"; e:.j.k each .dash.sent;
    .t.eq[e[0]`type;"error"];
    .t.eq[e[0]`message;"send a JSON object"];
    .t.eq[.t.types .t.ask enlist[`cmd]!enlist "bogus";enlist `error];
    .t.eq[.t.types .t.ask `cmd`data!("train";"nope");enlist `error];
    .t.eq[.t.types .t.ask `cmd`run!("predict";999);enlist `error];
    .t.eq[.t.types .t.ask enlist[`cmd]!enlist "hello";enlist `hello];
 }];

.t.add[`dash_clean_json;{
    .t.eq[.j.j .dash.clean (0w;-0w;1.5;`a`b!(0w;"x"));"[null,null,1.5,{\"a\":null,\"b\":\"x\"}]"];
 }];

/ a real dashboard process: static files, refused paths, and q IPC for KX Dashboards
.t.add[`dash_http_ipc;{
    p:30000+(.z.i+7) mod 20000;
    system "env ",(first .z.X)," ",.t.root,"/dashboard.q -q -port ",(string p)," </dev/null >/dev/null 2>&1 &";
    h:{[p;x] if[not null x; :x]; system "env sleep 0.1"; @[hopen;(`$"::",string p;100);0Ni]}[p]/[50;0Ni];
    if[null h; '"dashboard did not start"];
    fetch:{[p;path] system "env curl -s --path-as-is -w ' %{http_code}' http://127.0.0.1:",(string p),"/",path};
    res:.Q.trp[{[h;fetch;u]
        .t.ok[(first fetch "") like "<!doctype html>*";"index not served"];
        .t.ok[(last fetch "app.js") like "* 200";"app.js not served"];
        .t.ok[(last fetch "style.css?v=1") like "* 200";"query string not ignored"];
        {.t.ok[(last x) like "* 404";"served ",y]}'[fetch each ("../dashboard.q";"%2e%2e/PLAN.md";"nope.html";".hidden");("../dashboard.q";"%2e%2e/PLAN.md";"nope.html";".hidden")];
        .t.eq[0#h ".dash.runs";0#.dash.runs];
        .t.eq[h ".dash.history[]";()];
        h "\\p"}[h;fetch[p]];::;{(`err;x)}];
    pid:h ".z.i"; hclose h; system "kill ",string pid;
    if[`err~first res; e:res 1; 'e];
 }];
