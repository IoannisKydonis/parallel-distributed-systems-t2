#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k);

void hadamardProduct(double *x, double *y, double *res, int length);

double kNearest(double *dist, int *indexValues, int l, int r, int k, int *idx);

int partition(double *dist, int *indexValues, int l, int r);

void swap(double *n1, double *n2);

void swapInts(int *n1, int *n2);

// Definition of the kNN result struct
struct knnresult
{
    int *nidx;     //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist; //!< Distance of nearest neighbors          [m-by-k]
    int m;         //!< Number of query points                 [scalar]
    int k;         //!< Number of nearest neighbors            [scalar]
};

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

    double *xx = malloc(n * d * sizeof(double));
    hadamardProduct(x, x, xx, n * d);

    double *yy = malloc(m * d * sizeof(double));
    hadamardProduct(y, y, yy, m * d);

    double *xxSum = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        xxSum[i] = cblas_dasum(d, xx + i * d, 1);
    }

    double *yySum = malloc(m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        yySum[i] = cblas_dasum(d, yy + i * d, 1);
    }

    double *xy = malloc(n * m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, x, d, y, d, 0, xy, m);

    double *dist = malloc(n * m * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            dist[i * m + j] = sqrt(xxSum[i] + xy[i * m + j] + yySum[j]);
        }
    }

    int **indexes = (int **)malloc(n * sizeof(int *));
    double **nearest = (double **)malloc(n * sizeof(double *));

    for (int i = 0; i < n; i++)
    {
        nearest[i] = (double *)malloc(k * sizeof(double));
        indexes[i] = (int *)malloc(k * sizeof(int));
    }

    int *indexValues = (int *)malloc(n * m * sizeof(int));
    for (int i = 0; i < n * m; i++)
        indexValues[i] = i;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            int idx;
            nearest[i][j] = kNearest(dist, indexValues, i * m, (i + 1) * m - 1, j + 1, &idx);
            indexes[i][j] = idx - i * m;
        }
    }

    int * indexesRowMajor=(int*)malloc(n*k*sizeof(int));
    double * distRowMajor=(double*)malloc(n*k*sizeof(double));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            indexesRowMajor[i*k +j]=indexes[i][j];
            distRowMajor[i*k +j]=nearest[i][j];
        }
    }
    
    result->ndist=distRowMajor;
    result->nidx=indexesRowMajor;
    result->m=m;
    result->k=k;


    for (int i = 0; i < n; i++)
    {
        free(indexes[i]);
        free(nearest[i]);
    }
    free(xx);
    free(yy);
    free(xy);
    free(xxSum);
    free(yySum);
    free(dist);
    free(nearest);
    free(indexes);
    free(indexValues);

    return *result;
}

void hadamardProduct(double *x, double *y, double *res, int length)
{
    for (int i = 0; i < length; i++)
        res[i] = x[i] * y[i];
}

// k is one based
double kNearest(double *dist, int *indexValues, int left, int right, int k, int *idx)
{
    int pivot = partition(dist, indexValues, left, right);

    if (k < pivot - left + 1)
    {
        return kNearest(dist, indexValues, left, pivot - 1, k, idx);
    }
    else if (k > pivot - left + 1)
    {
        return kNearest(dist, indexValues, pivot + 1, right, k - pivot + left - 1, idx);
    }
    else
    {
        *idx = indexValues[pivot];
        return dist[pivot];
    }
}

int partition(double *dist, int *indexValues, int left, int right)
{
    int x = dist[right];
    int i = left;
    for (int j = left; j <= right - 1; j++)
    {
        if (dist[j] <= x)
        {
            swap(&dist[i], &dist[j]);
            swapInts(&indexValues[i], &indexValues[j]);
            i++;
        }
    }
    swap(&dist[i], &dist[right]);
    swapInts(&indexValues[i], &indexValues[right]);
    return i;
}

void swap(double *n1, double *n2)
{
    double temp = *n1;
    *n1 = *n2;
    *n2 = temp;
}

void swapInts(int *n1, int *n2)
{
    int temp = *n1;
    *n1 = *n2;
    *n2 = temp;
}