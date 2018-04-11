
IPPBASE =  /opt/intel
IPPLIB = -I$(IPPBASE)/ipp/include -Wl,-rpath,$(IPPBASE)/ipp/lib -L$(IPPBASE)/ipp/lib -lipps -lippvm -lippcore

DIFXROOT = /Users/phi196/code/difx
DIFXINC = -I$(DIFXROOT)/include
DIFXLIB = -L${DIFXROOT}/lib

PGPLOTINCS = 
PGPLOTLIBS = -lcpgplot -lpgplot


#CC = gcc -Wall -O3 -mavx2
CC = gcc -Wall -g

all: autoSpec generateSpectrum

generateSpectrum: generateSpectrum.c bitconversion.c  bitconversion.h
	$(CC) $(IPPLIB) -o  $@ $< bitconversion.c -lm

autoSpec: autoSpec.c bitconversion.c  bitconversion.h
	$(CC) $(IPPLIB) $(PGPLOTINCS) $(PGPLOTLIBS) -o  $@ $< bitconversion.c -lm

