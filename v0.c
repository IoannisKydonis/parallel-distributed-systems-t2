#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k);

// Definition of the kNN result struct
struct knnresult
{
    int *nidx;     //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist; //!< Distance of nearest neighbors          [m-by-k]
    int m;         //!< Number of query points                 [scalar]
    int k;         //!< Number of nearest neighbors            [scalar]
};

int MAX_Y_SIZE = 4;

int main(int argc, char *argv[])
{
    struct knnresult result;
    double x[20] = {
        1.0, 2.0,
        0.5, 1.2,
        7.9, 4.6,
        4.0, 0.0,
        8.4, 1.9,
        -1.0, 5.0,
        1.5, 7.2,
        7.1, 3.9,
        -45.0, 10.1,
        28.4, 31.329};

    double y[10] = {
        6.2, 1.2,
        7.0, 15.3,
        13.9, 1.2,
        17.22, 78.01,
        1.3, -23.9};

    result=kNN(x, y, 10, 5, 2, 5);
    printf("m=%d k=%d\n", result.m, result.k);

    printf("Nearest: ");
    int n = 10;
    for(int i=0; i<n * result.k; i++){
        if (i % result.k == 0) {
            printf("\n");
        }
        printf("%f ", result.ndist[i]);
    }
    printf("\n");

    printf("Indexes: ");
    for(int i=0; i<n * result.k; i++){
        if (i % result.k == 0) {
            printf("\n");
        }
        printf("%d ", result.nidx[i]);
    }
    printf("\n");

    free(result.nidx);
    free(result.ndist);
}

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k)
{
    struct knnresult *result = malloc(sizeof(struct knnresult));
    int * indexesRowMajor=(int*)malloc(n*k*sizeof(int));
    double * distRowMajor=(double*)malloc(n*k*sizeof(double));

    for (int ii = 0; ii < m / MAX_Y_SIZE; ii++) {
        int partitionSize = MAX_Y_SIZE;
        if (ii == m / MAX_Y_SIZE - 1) {
            partitionSize = m % MAX_Y_SIZE;
        }

        double *currentY = malloc(partitionSize * d * sizeof(double));
        for (int i = 0; i < partitionSize; i++) {
            for (int j = 0; j < d; j++) {
                currentY[i * d + j] = y[ii * MAX_Y_SIZE + i * d + j];
            }
        }

        double *xx = malloc(n * d * sizeof(double));
        hadamardProduct(x, x, xx, n * d);

        double *xxSum = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++)
        {
            xxSum[i] = cblas_dasum(d, xx + i * d, 1);
        }

        double *yy = malloc(partitionSize * d * sizeof(double));
        hadamardProduct(currentY, currentY, yy, partitionSize * d);

        double *yySum = malloc(partitionSize * sizeof(double));
        for (int i = 0; i < partitionSize; i++)
        {
            yySum[i] = cblas_dasum(d, yy + i * d, 1);
        }

        double *xy = malloc(n * partitionSize * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, partitionSize, d, -2, x, d, currentY, d, 0, xy, partitionSize);

        double *dist = malloc(n * partitionSize * sizeof(double));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < partitionSize; j++)
            {
                dist[i * partitionSize + j] = sqrt(xxSum[i] + xy[i * partitionSize + j] + yySum[j]);
            }
        }

        int *indexValues = (int *)malloc(n * partitionSize * sizeof(int));
        for (int i = 0; i < n * partitionSize; i++)
            indexValues[i] = i;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                int idx;
                distRowMajor[ii * MAX_Y_SIZE + i*k +j] = kNearest(dist, indexValues, i * partitionSize, (i + 1) * partitionSize - 1, j + 1, &idx);
                indexesRowMajor[ii * MAX_Y_SIZE + i*k +j] = idx - i * partitionSize;
            }
        }

        free(currentY);
        free(xx);
        free(yy);
        free(xy);
        free(xxSum);
        free(yySum);
        free(dist);
        free(indexValues);
    }

    result->ndist=distRowMajor;
    result->nidx=indexesRowMajor;
    result->m=m;
    result->k=k;



    return *result;
}
