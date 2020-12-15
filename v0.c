#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt
#include <cblas.h> // cblas_dgemm

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k);

void hadamardProduct(double *x, double *y, double *res, int length);

// Definition of the kNN result struct
struct knnresult {
    int *nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist;   //!< Distance of nearest neighbors          [m-by-k]
    int m;       //!< Number of query points                 [scalar]
    int k;       //!< Number of nearest neighbors            [scalar]
};

int main(int argc, char *argv[]) {
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
            28.4, 31.329
    };

    double y[10] = {
            6.2, 1.2,
            7.0, 15.3,
            13.9, 1.2,
            17.22, 78.01,
            1.3, -23.9
    };

    kNN(x, y, 10, 5, 2, 0);
}

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k) {
    struct knnresult *result = malloc(sizeof(struct knnresult));

    double *xx = malloc(n * d * sizeof(double));
    hadamardProduct(x, x, xx, n * d);

    double *yy = malloc(m * d * sizeof(double));
    hadamardProduct(y, y, yy, m * d);

    double *xxSum = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        xxSum[i] = cblas_dasum(d, xx + i * d, 1);
    }

    double *yySum = malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) {
        yySum[i] = cblas_dasum(d, yy + i * d, 1);
    }

    double *xy = malloc(n * m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, x, d, y, d, 0, xy, m);

    double *dist = malloc(n * m * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            dist[i * m + j] = sqrt(xxSum[i] + xy[i * m + j] + yySum[j]);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%10.4lf ", dist[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");

    // TODO: kydonis - compute result

    free(xx);
    free(yy);
    free(xy);
    free(xxSum);
    free(yySum);
    free(dist);

    return *result;
}

void hadamardProduct(double *x, double *y, double *res, int length) {
    for (int i = 0; i < length; i++)
        res[i] = x[i] * y[i];
}
