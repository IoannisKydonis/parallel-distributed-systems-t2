CC=gcc
MPICC=/usr/local/bin/mpicc
CFLAGS=-O3
CBLAS=-I/usr/local/opt/openblas/include -lcblas

N = 3

default: all

all: v0 v1 v2

v0: v0.c utilities.c
	$(CC) $(CFLAGS) -o $@ $^ $(CBLAS)

v1: v1.c utilities.c timer.c controller.c read.c
	$(MPICC) $(CFLAGS) -o $@ $^ $(CBLAS)

v2: v2.c utilities.c timer.c controller.c read.c
	$(MPICC) $(CFLAGS) -o $@ $^ $(CBLAS)

run1:
	mpiexec -np $(N) ./v1 BBC.txt 10

run2:
	mpiexec -np $(N) ./v2 BBC.txt 10

clean:
	rm -f v0 v1 v2
