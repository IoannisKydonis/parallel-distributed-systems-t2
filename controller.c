#include <stdio.h> // fprintf, printf
#include <stdlib.h> // EXIT_FAILURE, EXIT_SUCCESS
#include <stdint.h>
#include "timer.h" // measureTimeForRunnable
#include "types.h"

void saveStats(char *name, double time, int n, int d, int k, char *filename) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "Algorithm: %s\n", name);
    fprintf(f, "Time: %lf sec\n", time);
    fprintf(f, "n=%d, d=%d, k=%d\n", n, d, k);
    fclose(f);
}

knnresult runAndPresentResult(knnresult (*runnable)(double *x, int n, int d, int k), double *x, int n, int d, int k, char *name, char *outputFilename, char *resultsFilename) {
    struct knnresult result;
    double time = measureTimeForRunnable(runnable, x, n, d, k, &result);
    printf("-----------------------------------\n");
    printf("| Algorithm: %s\n", name);
    printf("| Time: %10.6lf\n", time);
    printf("| n=%d, d=%d, k=%d\n", n, d, k);
    printf("-----------------------------------\n");
    saveStats(name, time, n, d, k, resultsFilename);
    return result;
}
