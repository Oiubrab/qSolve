/ minimal assertion helpers and test registry
/ assertions signal an error string on failure; the runner traps it and records the message

/ registry of tests: name -> niladic function
.t.reg:(`symbol$())!()

/ register a test
.t.add:{[name;fn] .t.reg[name]:fn;}

/ fail unless c is true
.t.ok:{[c;msg] if[not all (),c; 'msg];}

/ fail unless a matches b exactly
.t.eq:{[a;b] if[not a~b; '"expected ",(-3!b)," got ",-3!a];}

/ fail unless every element of a is within tol of b (works on nested lists, nulls always fail)
.t.near:{[a;b;tol]
    d:abs (raze/) (),a-b;
    if[any null d; '"null in comparison: ",-3!a];
    if[tol<m:max d; '"max abs diff ",(string m)," > ",string tol];
 }

/ fail unless f[x] signals an error
.t.throws:{[f;x] if[not @[{x y;0b}[f];x;{1b}]; '"expected an error from ",-3!x];}

/ skip the current test, e.g. when downloaded data is missing
.t.skip:{[msg] '"skip: ",msg}

/ path of a file under the repo root
.t.path:{[rel] `$":",.t.root,"/",rel}
