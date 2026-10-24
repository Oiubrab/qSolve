/ General-purpose radix-2 Fourier transforms. Complex values are (real; imaginary) float vectors.

.qs.sig.pi:acos -1

.qs.sig.pow2ceil:{[n]
    if[n<1; '"FFT input must not be empty"];
    size:1;
    while[size<n; size:2*size];
    size
 }

.qs.sig.transform:{[complex;direction]
    if[not 2=count complex; '"complex input must contain real and imaginary vectors"];
    real:"f"$complex 0;
    imag:"f"$complex 1;
    if[not count[real]=count imag; '"real and imaginary vectors must have equal lengths"];
    original:count real;
    n:.qs.sig.pow2ceil original;
    real:real,(n-original)#0f;
    imag:imag,(n-original)#0f;
    permutation:$[n=1;enlist 0;0 1];
    width:4;
    while[width<=n;
        permutation:raze (2*permutation;1+2*permutation);
        width:2*width];
    real:real permutation;
    imag:imag permutation;
    width:2;
    while[width<=n;
        half:width div 2;
        bases:width*til n div width;
        low:raze bases +\: til half;
        high:low+half;
        k:raze (count bases)#enlist til half;
        angle:direction*2*.qs.sig.pi*k%width;
        wr:cos angle;
        wi:sin angle;
        highReal:real high;
        highImag:imag high;
        mixedReal:(wr*highReal)-(wi*highImag);
        mixedImag:(wr*highImag)+(wi*highReal);
        lowReal:real low;
        lowImag:imag low;
        real[low]:lowReal+mixedReal;
        imag[low]:lowImag+mixedImag;
        real[high]:lowReal-mixedReal;
        imag[high]:lowImag-mixedImag;
        width:2*width];
    (real;imag)
 }

.qs.sig.fft:{[complex] .qs.sig.transform[complex;-1f]}

.qs.sig.ifft:{[complex]
    result:.qs.sig.transform[complex;1f];
    n:count result 0;
    ((result 0)%n;(result 1)%n)
 }

/ Real transforms use a one-sided spectrum; rfft zero-pads to a radix-2 length.
.qs.sig.rfft:{[real]
    real:"f"$real;
    n:.qs.sig.pow2ceil count real;
    result:.qs.sig.fft[(real;(count real)#0f)];
    bins:1+n div 2;
    (result[0] til bins;result[1] til bins)
 }

.qs.sig.irfft:{[spectrum]
    if[not 2=count spectrum; '"spectrum must contain real and imaginary vectors"];
    real:"f"$spectrum 0;
    imag:"f"$spectrum 1;
    if[not count[real]=count imag; '"spectrum vectors must have equal lengths"];
    n:$[1=count real;1;(2*count real)-2];
    if[not count[real]=1+n div 2; '"one-sided spectrum length does not match output length"];
    innerReal:reverse 1 _ -1 _ real;
    innerImag:reverse 1 _ -1 _ imag;
    full:.qs.sig.ifft[(real,innerReal;imag,neg innerImag)];
    full 0
 }

.qs.sig.hann:{[n]
    if[n<2; '"window length must be at least two"];
    0.5-0.5*cos (2*.qs.sig.pi*til n)%(n-1)
 }

.qs.sig.tukeyInner:{[n;alpha]
    edge:alpha*(n-1)%2;
    i:til n;
    left:0.5*(1+cos .qs.sig.pi*(2*i%(alpha*(n-1))-1));
    right:reverse left;
    ?[i<edge;left;?[i>(n-1)-edge;right;1f]]
 }

.qs.sig.tukey:{[n;alpha]
    if[n<2; '"window length must be at least two"];
    if[not alpha within 0 1; '"Tukey alpha must be between zero and one"];
    $[alpha=0;n#1f;alpha=1;.qs.sig.hann n;.qs.sig.tukeyInner[n;alpha]]
 }

.qs.sig.periodogram:{[segment;window;rate]
    n:count segment;
    if[not n=.qs.sig.pow2ceil n; '"Welch segment length must be a power of two"];
    if[not count window=n; '"window length must match segment length"];
    spectrum:.qs.sig.rfft segment*window;
    realSpectrum:spectrum 0;
    imagSpectrum:spectrum 1;
    realPower:realSpectrum*realSpectrum;
    imagPower:imagSpectrum*imagSpectrum;
    power:realPower+imagPower;
    windowPower:window*window;
    windowEnergy:sum windowPower;
    normalization:rate*windowEnergy;
    power:power%normalization;
    bins:count power;
    interior:1+til[bins-2];
    power[interior]:2*power[interior];
    `freq`psd!((rate*(til[bins])%n);power)
 }

/ Welch one-sided PSD in physical units per Hz. Overlap is a count of samples.
.qs.sig.welch:{[samples;rate;segment;overlap]
    samples:"f"$samples;
    if[not segment=.qs.sig.pow2ceil segment; '"Welch segment length must be a power of two"];
    if[overlap<0 or overlap>=segment; '"Welch overlap must be in [0, segment)"];
    if[segment>count samples; '"Welch segment is longer than the input"];
    stride:segment-overlap;
    starts:stride*til[1+(count[samples]-segment) div stride];
    window:.qs.sig.hann segment;
    chunks:{[series;offset;size] series[offset+til size]}[samples;;segment] each starts;
    spectra:.qs.sig.periodogram[;window;rate] each chunks;
    psdRows:{[spectrum] spectrum[`psd]} each spectra;
    `freq`psd!((first spectra)[`freq];avg psdRows)
 }