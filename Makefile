CC=gcc
CFLAGS=-O3
CBLAS=-I/usr/local/opt/openblas/include -lcblas

default: all

all: v0

v0: v0.c utilities.c
	$(CC) $(CFLAGS) $(CBLAS) -o $@ $^

clean:
	rm -f v0
