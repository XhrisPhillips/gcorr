# Gcorr Generators
This code can be used to generate input data for gcorr correlator.  Depends on Intel IPP libraries

### Generate Spectrum
Create fake data with realistic Gaussian noise and a bandpass. 
```
Usage: generateSpectrum [options] [<outputFile>]
  -w/-bandwidth <BANDWIDTH>  Channel bandwidth in MHz (64)
  -b/-nbits <N>             Number of bits/sample (default 2)
  -C/-channels <N>          Number of IF channels (default 1)
  -c/-complex               Generate complex data
  -f/-float                 Save data as floats
  -l/-duration <DURATION>   Length of output, in seconds
  -T/-tone <TONE>           Frequency of tone (MHz)
  -2/-tone2 <TONE>          Frequency of second tone  (MHz)
  -a/-amp <amp>             Amplitude of tone
  -A/-amp2 <amp2>d          Amplitude of second tone
  -n/-noise                 Include (correlated) noise
  -ntap <TAPS>              Number of taps for FIR filter to create band shape
```

Typical usage:

```
generateSpectrum -bandwidth 16 -nbits 8 -channels 1 -complex -duration 10  -tone 4 -amp 0.01 test.vdf
```
Generate a spectrum with commplex voltage, 8 bits per complex component, 16 MHz bandwidth and a weak tone at 4 MHz from lower edge
```
generateSpectrum -bandwidth 16 -nbits 8 -channels 1 -complex -duration 10  -noise -tone2 4 -amp2 0.01 test.vdf
```
Generate the same spectrum as above, but include some correlatable noise - when run twice each file will include some random (uncorrelated) noise plus some correlated noise


### autoSpec

Generate autocorrelation Spectrum of voltage data
```
Usage: autoSpec [options]
  -C/-chan <n>        # Number of channels in input file
  -n/-npoint <n>      # spectral channels to generate
  -b/-bits <n>        # Input voltage bits per sample
  -c/-complex         # Input voltage complex
  -f/-float           # Input voltage IEEE float, not integer
  -noplot>            # Don't plot using PGPLOT
  -specfile <file>    # Output filename of spectrum
  -device <pgdev>     # Pgplot device to plot to
  -s/-skip <n>        Skip this many bytes at start of file
  -h/-help            This list
```

Typical usage:

```
./autoSpec -bits 8 -chan 1 -complex -bandwidth 16  test.vdf
```


## Compiling

Edit Makefile and change "base" to the location of IPP. Potentially need to edit IPPLIB paths
