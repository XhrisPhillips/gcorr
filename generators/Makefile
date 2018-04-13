
base =  /opt/intel
IPPLIB = -I${base}/ipp/include -Wl,-rpath,${base}/lib/intel64 -Wl,-rpath,${base}/ipp/lib/intel64 -L${base}/lib/intel64 -L${base}/ipp/lib/intel64 -L${base}/compiler/lib/intel64 -lipps -lippvm -lippcore

PGPLOTINCS = 
PGPLOTLIBS = -lcpgplot -lpgplot

#CC = gcc -Wall -O3 -mavx2
CC = gcc -Wall -g

all: autoSpec generateSpectrum

generateSpectrum: generateSpectrum.c bitconversion.c  bitconversion.h
	$(CC) $(IPPLIB) -o  $@ $< bitconversion.c -lm

autoSpec: autoSpec.c bitconversion.c  bitconversion.h
	$(CC) $(IPPLIB) $(PGPLOTINCS) $(PGPLOTLIBS) -o  $@ $< bitconversion.c -lm

