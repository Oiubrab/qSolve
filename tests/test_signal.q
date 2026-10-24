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