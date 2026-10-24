/ checks that the assertion helpers themselves behave

.t.add[`harness_ok;{.t.ok[1b;"unused"]; .t.ok[1 1b;"unused"]}];
.t.add[`harness_eq;{.t.eq[1 2 3;1 2 3]}];
.t.add[`harness_near;{.t.near[(1 2f;3f);(1 2.0000001;3f);1e-6]}];
.t.add[`harness_detects_failure;{
    .t.throws[.t.ok[;"x"];0b];
    .t.throws[.t.eq[1];2];
    .t.throws[.t.near[1f;;1e-9];1.1];
    .t.throws[.t.near[0n;;1];0f];
 }];
