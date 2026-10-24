/ Radix-2 Fourier transforms

.t.sigDftBin:{[complex;index;n]
    angle:(2*.qs.sig.pi*index*til n)%n;
    real:complex 0;
    imag:complex 1;
    ((sum real*cos angle)+(sum imag*sin angle);(sum imag*cos angle)-(sum real*sin angle))
 }

.t.add[`signal_fft_matches_dft;{
    system "S 17";
    complex:(8?1f;8?1f);
    actual:.qs.sig.fft complex;
    n:count complex 0;
    expected:.t.sigDftBin[complex;;n] each til n;
    .t.near[actual 0;first each expected;1e-9];
    .t.near[actual 1;last each expected;1e-9];
 }];

.t.add[`signal_fft_roundtrip_parseval;{
    system "S 23";
    real:16?1f;
    imag:16?1f;
    spectrum:.qs.sig.fft[(real;imag)];
    restored:.qs.sig.ifft spectrum;
    .t.near[restored 0;real;1e-12];
    .t.near[restored 1;imag;1e-12];
    energy:(sum real*real)+(sum imag*imag);
    spectralEnergy:((sum spectrum[0]*spectrum[0])+(sum spectrum[1]*spectrum[1]))%count real;
    .t.near[spectralEnergy;energy;1e-12];
 }];

.t.add[`signal_fft_padding_tone;{
    .t.eq[.qs.sig.pow2ceil 13;16];
    real:cos (2*.qs.sig.pi*3*til[16])%16;
    spectrum:.qs.sig.rfft real;
    .t.eq[count spectrum 0;9];
    .t.eq[spectrum[0]?max spectrum 0;3];
    .t.near[.qs.sig.irfft spectrum;real;1e-12];
    short:1 2 3 4 5f;
    padded:.qs.sig.irfft .qs.sig.rfft short;
    .t.near[padded;1 2 3 4 5 0 0 0f;1e-12];
 }];

.t.add[`signal_windows;{
     hann:.qs.sig.hann 9;
     .t.near[hann;reverse hann;1e-12];
     .t.near[first hann;0f;1e-12];
     .t.near[last hann;0f;1e-12];
     .t.eq[.qs.sig.tukey[9;0];9#1f];
     .t.near[.qs.sig.tukey[9;1];hann;1e-12];
     .t.near[.qs.sig.tukey[9;0.5];reverse .qs.sig.tukey[9;0.5];1e-12];
 }];

.t.add[`signal_welch_white_noise;{
     system "S 31";
     rate:1024f;
   samples:0.5*.qs.data.normal 65536;
     spectrum:.qs.sig.welch[samples;rate;1024;512];
     expected:2*0.25%rate;
  psd:spectrum[`psd];
   interior:psd[1+til (count[psd]-2)];
   .t.near[avg interior;expected;0.03*expected];
    freq:spectrum[`freq];
    .t.eq[count freq;513];
    .t.near[freq[0 1 512];0 1 512f;1e-12];
 }];

.t.add[`signal_welch_tone;{
     rate:1024f;
   phase:(2*.qs.sig.pi*73*til 8192)%rate;
   samples:cos phase;
     spectrum:.qs.sig.welch[samples;rate;1024;512];
  psd:spectrum[`psd];
  freq:spectrum[`freq];
    peakValue:max psd;
   peak:first where peakValue=psd;
   .t.eq[freq[peak];73f];
 }];

.t.add[`signal_whiten_unit_variance;{
    system "S 37";
    rate:1024f;
   white:0.5*.qs.data.normal 65536;
   n:.qs.sig.pow2ceil count white;
   freq:rate*(til[1+(n div 2)])%n;
   baseline:sqrt 0.5%rate;
   asdValues:baseline*(1+4f*freq%rate);
   asd:`freq`asd!(freq;asdValues);
   sourceSpectrum:.qs.sig.rfft white;
   coloured:.qs.sig.irfft[((sourceSpectrum 0)*(asdValues%baseline);(sourceSpectrum 1)*(asdValues%baseline))];
   whitened:.qs.sig.whiten[coloured;rate;asd];
    variance:avg (whitened-avg whitened)*(whitened-avg whitened);
    .t.near[variance;1f;0.05];
    spectrum:.qs.sig.welch[whitened;rate;1024;512];
    psd:spectrum[`psd];
    .t.near[avg psd[1+til[count[psd]-2]];2%rate;0.05*(2%rate)];
 }];

.t.add[`signal_bandpass_tones;{
    rate:2048f;
    n:2048;
   time:(til[n])%rate;
   lowPhase:(2*.qs.sig.pi*100*time);
   highPhase:(2*.qs.sig.pi*700*time);
   lowTone:cos lowPhase;
   highTone:cos highPhase;
    filtered:.qs.sig.bandpass[lowTone+highTone;rate;50;300;20];
   filteredSpectrum:.qs.sig.rfft filtered;
   sourceSpectrum:.qs.sig.rfft lowTone;
   lowPeak:max sqrt (filteredSpectrum 0)*(filteredSpectrum 0)+(filteredSpectrum 1)*(filteredSpectrum 1);
   sourcePeak:max sqrt (sourceSpectrum 0)*(sourceSpectrum 0)+(sourceSpectrum 1)*(sourceSpectrum 1);
    .t.near[lowPeak;sourcePeak;0.05*sourcePeak];
    filteredHigh:.qs.sig.bandpass[highTone;rate;50;300;20];
    .t.ok[max abs filteredHigh<0.05;"out-of-band tone was not suppressed"];
 }];

.t.add[`signal_resample_tone;{
    sourceRate:4096f;
    targetRate:2048f;
    n:4096;
   time:(til[n])%sourceRate;
   phase:(2*.qs.sig.pi*200*time);
   tone:cos phase;
    reduced:.qs.sig.resample[tone;sourceRate;targetRate];
    .t.eq[count reduced;2048];
    .t.near[max reduced;1f;1e-10];
    spectrum:.qs.sig.rfft reduced;
   real:spectrum 0;
   peakValue:max real;
   peak:first where peakValue=real;
    .t.eq[peak;200];
 }];