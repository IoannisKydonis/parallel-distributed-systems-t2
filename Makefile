CC=gcc
MPICC=mpicc
CFLAGS=-O3
CBLAS=-lopenblas -lpthread -lm
N=2

default: all

all: v0

v0: v0.c utilities.c timer.c controller.c
	$(CC) $(CFLAGS)  -o $@ $^ $(CBLAS)

v1: v1.c utilities.c timer.c controller.c read.c
	$(MPICC) $(CFLAGS) -o $@ $^ $(CBLAS)

v2: v2.c utilities.c timer.c controller.c read.c
	$(MPICC)  $^ -o $@ $(CBLAS) $(CFLAGS)

run1:
	mpiexec -np $(N) ./v1  CNN.txt 30

run2:
	mpiexec -np $(N) ./v2  CNN.txt 30 

clean:
	rm -f v0 v1 v2




