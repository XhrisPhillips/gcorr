
base =  /data/BALDR_0/phi196/oneapi/ipp/latest/
IPPLIB = -I${base}/include -Wl,-rpath,${base}/lib/intel64 -Wl,-rpath,${base}/ipp/lib/intel64 -L${base}/lib/intel64 -L${base}/ipp/lib/intel64 -L${base}/compiler/lib/intel64 -lipps -lippvm -lippcore

PGPLOTINCS = 
PGPLOTLIBS = -lcpgplot -lpgplot

#CC = gcc -Wall -O3 -mavx2
CC = gcc -Wall -g

all: autoSpec generateSpectrum

generateSpectrum: generateSpectrum.c bitconversion.c  bitconversion.h
	$(CC)  -o  $@ $< bitconversion.c -lm $(IPPLIB)

autoSpec: autoSpec.c bitconversion.c  bitconversion.h
	$(CC)  -o  $@ $< bitconversion.c -lm $(IPPLIB) $(PGPLOTINCS) $(PGPLOTLIBS)

