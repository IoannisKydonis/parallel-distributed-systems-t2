
#ifndef UTILITIES_H
#define UTILITIES_H

void hadamardProduct(double *x, double *y, double *res, int length);

double kNearest(double *dist, int *indexValues, int l, int r, int k, int *idx);

int partition(double *dist, int *indexValues, int l, int r);

void swap(double *n1, double *n2);

void swapInts(int *n1, int *n2);

#endif //UTILITIES_H
