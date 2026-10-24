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

.qs.sig.interpolate1:{[sourceFreq;sourceValue;target]
    right:first where target<sourceFreq;
    $[0=count where target<sourceFreq;last sourceValue;right=0;first sourceValue;
        [left:right-1;
         fraction:(target-sourceFreq[left])%(sourceFreq[right]-sourceFreq[left]);
         sourceValue[left]+fraction*(sourceValue[right]-sourceValue[left])]]
 }

.qs.sig.interpolate:{[sourceFreq;sourceValue;targetFreq]
    if[not count[sourceFreq]=count sourceValue; '"interpolation inputs must have equal lengths"];
    if[not 1<count sourceFreq; '"interpolation needs at least two source points"];
    .qs.sig.interpolate1[sourceFreq;sourceValue] each targetFreq
 }

.qs.sig.asdOnGrid:{[asd;freq]
    if[not all `freq`asd in key asd; '"ASD needs freq and asd vectors"];
    .qs.sig.interpolate[asd[`freq];asd[`asd];freq]
 }

/ Whiten strain using a one-sided ASD; the flat-white-noise ASD maps to unit time-domain variance.
.qs.sig.whiten:{[samples;rate;asd]
    samples:"f"$samples;
    spectrum:.qs.sig.rfft samples;
    bins:count spectrum 0;
    frequencies:rate*(til[bins])%(.qs.sig.pow2ceil count samples);
    amplitude:.qs.sig.asdOnGrid[asd;frequencies];
    if[any amplitude<=0 or any null amplitude; '"ASD must be positive and non-null on the frequency grid"];
    scale:sqrt 2%rate;
    scale:scale%amplitude;
    .qs.sig.irfft[((spectrum 0)*scale;(spectrum 1)*scale)]
 }

.qs.sig.edgeMask1:{[freq;low;high;edge]
    lowPhase:.qs.sig.pi*(freq-(low-edge))%(2*edge);
    highPhase:.qs.sig.pi*((high+edge)-freq)%(2*edge);
    lowMask:$[freq<low-edge;0f;freq<low+edge;0.5-0.5*cos lowPhase;1f];
    highMask:$[freq>high+edge;0f;freq>high-edge;0.5-0.5*cos highPhase;1f];
    lowMask*highMask
 }

.qs.sig.edgeMask:{[frequencies;low;high;edge]
    if[low<0 or high<=low or edge<0; '"invalid band-pass limits"];
    $[edge=0; "f"$(frequencies>=low)&frequencies<=high;
        .qs.sig.edgeMask1[;low;high;edge] each frequencies]
 }

.qs.sig.bandpass:{[samples;rate;low;high;edge]
    samples:"f"$samples;
    spectrum:.qs.sig.rfft samples;
    bins:count spectrum 0;
    frequencies:rate*(til[bins])%(.qs.sig.pow2ceil count samples);
    mask:.qs.sig.edgeMask[frequencies;low;high;edge];
    .qs.sig.irfft[((spectrum 0)*mask;(spectrum 1)*mask)]
 }

/ Downsample by truncating frequencies above the new Nyquist and preserving time-domain amplitude.
.qs.sig.resample:{[samples;sourceRate;targetRate]
    samples:"f"$samples;
    if[targetRate<=0 or sourceRate<=targetRate; '"target rate must be positive and below source rate"];
    scaled:count[samples]*targetRate%sourceRate;
    targetCount:"j"$scaled;
    if[not scaled=targetCount; '"resampled length must be an integer"];
    if[not targetCount=.qs.sig.pow2ceil targetCount; '"resampled length must be a power of two"];
    spectrum:.qs.sig.rfft samples;
    targetBins:1+(targetCount div 2);
    scale:targetCount%count samples;
    real:scale*(targetBins#spectrum 0);
    imag:scale*(targetBins#spectrum 1);
    .qs.sig.irfft[(real;imag)]
 }