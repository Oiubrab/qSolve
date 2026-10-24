/ GWOSC event catalogue and detector metadata

.t.add[`gw_catalogue_sample;{
    json:raze read0 hsym .t.path "tests/fixtures/gw_events.json";
    events:.qs.gw.parseEvents json;
    .t.eq[string events`name;("GW150914";"GW170817";"GWTEST")];
    .t.eq[events`version;3 3 1];
    .t.eq[events`catalogue;.qs.gw.symbols ("GWTC-1-confident";"GWTC-1-confident";"GWTC-1-confident")];
    .t.near[events[0 1;`gps];1126259462.4 1187008882.4;1e-6];
    .t.near[events[0 1;`snr];25.2 32.4;1e-12];
    .t.ok[null events[2;`mass1];"missing mass should remain null"];
    selected:.qs.gw.confidentBBH[events;10;80;0.9];
    .t.eq[count selected;1];
    .t.eq[first selected`name;`GW150914];
    details:`GW150914`GW170817`GWTEST!(
        "{\"events\":{\"GW150914-v3\":{\"strain\":[{\"detector\":\"H1\"},{\"detector\":\"L1\"}]}}}";
        "{\"events\":{\"GW170817-v3\":{\"strain\":[{\"detector\":\"H1\"}]}}}";
        "{\"events\":{\"GWTEST-v1\":{\"strain\":[]}}}");
    selected:.qs.gw.confidentWithDetectors[events;details;10;80;0.9;`H1`L1];
    .t.eq[count selected;1];
    .t.eq[first selected`name;`GW150914];
 }];

.t.add[`gw_detector_metadata;{
    json:"{\"events\":{\"GW150914-v3\":{\"strain\":[{\"detector\":\"H1\"},{\"detector\":\"L1\"}]}}}";
    .t.ok[.qs.gw.hasDetectors[json;`H1`L1];"H1 and L1 should be available"];
    .t.ok[not .qs.gw.hasDetectors[json;`H1`V1];"missing detector should not be available"];
 }];

.t.add[`gw_catalogue_cached;{
     file:.t.path "data/gw/allevents.json";
     if[()~key file; .t.skip "fetch the GWOSC catalogue with .qs.gw.events[]"];
     events:.qs.gw.parseEvents raze read0 file;
     .t.ok[300<count events;"unexpectedly few de-duplicated GWOSC events"];
     .t.ok[all `name`catalogue`gps`mass1`mass2`snr`pAstro in cols events;"catalogue fields missing"];
 }];