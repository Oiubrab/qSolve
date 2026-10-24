/ qSolve plots: decision boundaries of 2-D models as SVG files

/ class colours, cycled for more classes
.qs.plot.colours:("#2563eb";"#ea580c";"#059669";"#9333ea";"#dc2626";"#0891b2")

/ grid cells per side, and image size in pixels
.qs.plot.cells:80
.qs.plot.size:480

/ svg element text: name, attribute names and values (anything; strings are used as-is)
.qs.plot.el:{[name;attrs;vals] "<",name,(raze {" ",(string x),"='",$[10h=type y;y;string y],"'"}'[attrs;vals]),"/>"}

/ the data behind a decision-boundary plot: the plane covering d's points (padded by a tenth) cut into
/ n x n cells, row-major from the top-left, with each cell's predicted class and the probability of that class
.qs.plot.grid:{[m;d;n]
    if[not 2=count first first m`weight; '"plots need a model with 2 inputs"];
    pts:d[`X],$[`Xval in key d;d`Xval;()];
    lo:min pts; hi:max pts; pad:0.1*hi-lo; lo-:pad; hi+:pad;
    i:til n*n; u:(0.5+i mod n)%n; v:(0.5+i div n)%n;
    P:.qs.predict[m;flip (lo[0]+u*hi[0]-lo 0;hi[1]-v*hi[1]-lo 1)];
    P:$[1=count first P; {(1-x;x)} each first each P; P];
    `lo`hi`n`classes`prob!(lo;hi;n;{x?max x} each P;max each P)
 }

/ write an SVG of model m's predicted class over the plane covering data d (`X`Y, optionally `Xval`Yval)
/ cells are coloured by predicted class, paler where the model is less sure; training points are filled
/ circles in their true class colour, validation points hollow ones
.qs.plot2D:{[m;d;file;title] (hsym file) 0: .qs.plot.svg[m;d;title]; hsym file}

/ the same SVG as a list of lines
.qs.plot.svg:{[m;d;title]
    if[not 2=count first first m`weight; '"plot2D needs a model with 2 inputs"];
    pts:d[`X],$[`Xval in key d;d`Xval;()];
    lo:min pts; hi:max pts; pad:0.1*hi-lo; lo-:pad; hi+:pad;
    n:.qs.plot.cells; px:.qs.plot.size; c:px%n;
    / cell centres, row-major from the top-left, y increasing upwards
    i:til n*n; u:(0.5+i mod n)%n; v:(0.5+i div n)%n;
    P:.qs.predict[m;flip (lo[0]+u*hi[0]-lo 0;hi[1]-v*hi[1]-lo 1)];
    P:$[1=count first P; {(1-x;x)} each first each P; P];
    k:count first P; cls:{x?max x} each P;
    conf:0.12+0.55*0f|((max each P)-1%k)%1-1%k;
    toPx:{[lo;hi;px;p] (px*(p[0]-lo 0)%hi[0]-lo 0;px*1-(p[1]-lo 1)%hi[1]-lo 1)}[lo;hi;px];
    col:{.qs.plot.colours x mod count .qs.plot.colours};
    rects:{[c;n;col;i;k;a] .qs.plot.el["rect";(`x`y`width`height`fill,`$"fill-opacity");(c*i mod n;c*i div n;c+0.01;c+0.01;col k;a)]}[c;n;col]'[til count cls;cls;conf];
    dots:{[toPx;col;hollow;p;k] q:toPx p;
        .qs.plot.el["circle";(`cx`cy`r`fill`stroke,`$"stroke-width");(q 0;q 1;3;$[hollow;"none";col k];$[hollow;col k;"#1f2937"];$[hollow;1.5;0.6])]}[toPx;col];
    tr:dots[0b]'[d`X;.qs.classes d`Y];
    va:$[`Xval in key d; dots[1b]'[d`Xval;.qs.classes d`Yval]; ()];
    head:"<svg xmlns='http://www.w3.org/2000/svg' width='",(string px),"' height='",(string px+28),"' viewBox='0 -28 ",(string px)," ",(string px+28),"' shape-rendering='crispEdges'>";
    bg:.qs.plot.el["rect";`x`y`width`height`fill;(0;-28;px;px+28;"#ffffff")];
    txt:"<text x='8' y='-9' font-family='sans-serif' font-size='14' fill='#111827'>",title,"</text>";
    (head;bg;txt),rects,(enlist "<g shape-rendering='auto'>"),tr,va,("</g>";"</svg>")
 }
