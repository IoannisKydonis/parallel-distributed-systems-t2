CC=gcc
MPICC=/usr/local/bin/mpicc
CFLAGS=-O3
CBLAS=-I/usr/local/opt/openblas/include -lcblas

default: all

all: v0 v1

v0: v0.c utilities.c timer.c controller.c
	$(CC) $(CFLAGS) $(CBLAS) -o $@ $^

v1: v1.c utilities.c timer.c controller.c
	$(MPICC) $(CFLAGS) $(CBLAS) -o $@ $^

v2: v2.c utilities.c timer.c controller.c
	$(MPICC) $(CFLAGS) $(CBLAS) -o $@ $^

clean:
	rm -f v0
	rm -f v1
