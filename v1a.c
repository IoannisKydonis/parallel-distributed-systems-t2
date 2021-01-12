#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include <mpi.h>
#include <string.h>
#include "utilities.h"
#include "types.h"
#include "controller.h"
#include "read.h"

knnresult distrAllkNN(double *x, int n, int d, int k);

struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset);

int main(int argc, char *argv[]) {
    int SelfTID, NumTasks;
    MPI_Status mpistat;
    MPI_Request mpireq;  //initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);
    
    int n;
    int d;
    int k=atoi(argv[2]);

    //double * X=read_X(&n,&d,argv[1]);

    double X[30] = {
            1.0, 2.0,
            0.5, 1.2,
            7.9, 4.6,
            4.0, 0.0,
            8.4, 1.9,
            -1.0, 5.0,
            1.5, 7.2,
            7.1, 3.9,
            -45.0, 10.1,
            28.4, 31.329,
            0.7, 1.4,
            -8.4, -1.9,
            15, 7.2,
            -4.0, 1.1,
            8.4, -31.3};

    d=2;
    n=30/d;

    char *resFilename = (char *) malloc((40 + strlen(argv[1])) * sizeof(char));
    sprintf(resFilename, "v1_res_%s_%07d_%04d_%04d_%04d.txt", argv[1], n, d, k, NumTasks);
    knnresult mergedResult = runAndPresentResult(distrAllkNN, X, n, d, k, argv[1], "v1", resFilename);
    free(resFilename);

    if (SelfTID == 0) {    //send every result to the first process for printing
        char *outFilename = (char *) malloc((40 + strlen(argv[1])) * sizeof(char));
        sprintf(outFilename, "v1_out_%s_%07d_%04d_%04d_%04d.txt", argv[1], n, d, k, NumTasks);
        FILE *f = fopen(outFilename, "wb");
        for (int i = 0; i < mergedResult.m * mergedResult.k; i++) {
            if (i % mergedResult.k == 0)
                fprintf(f, "\n");
            fprintf(f, "%16.4f(%08d) ", mergedResult.ndist[i], mergedResult.nidx[i]);
        }
        for (int i = 1; i < NumTasks; i++) {
            double *incomingDistances = malloc(mergedResult.m * mergedResult.k * sizeof(double));
            int *incomingIndexes = malloc(mergedResult.m * mergedResult.k * sizeof(int));
            MPI_Recv(incomingDistances, mergedResult.m * mergedResult.k, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &mpistat);
            MPI_Recv(incomingIndexes, mergedResult.m * mergedResult.k, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &mpistat);
            int incomingLength;
            MPI_Get_count(&mpistat, MPI_INTEGER, &incomingLength);
            for (int i = 0; i < incomingLength; i++) {
                if (i % mergedResult.k == 0)
                    fprintf(f, "\n");
                fprintf(f, "%16.4f(%08d) ", incomingDistances[i], incomingIndexes[i]);
            }
            free(incomingDistances);
            free(incomingIndexes);
        }
        fprintf(f, "\n");
        fclose(f);
        free(outFilename);
    } else {
        MPI_Send(mergedResult.ndist, mergedResult.m * mergedResult.k, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Isend(mergedResult.nidx, mergedResult.m * mergedResult.k, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &mpireq);
    }

    MPI_Finalize();
    return 0;
}

knnresult distrAllkNN(double *x, int n, int d, int k) {

    int SelfTID, NumTasks;
    MPI_Status mpistat;
    MPI_Request mpireq;  //initialize MPI environment
    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    int *totalPoints = (int *) malloc(n * sizeof(int));
    dividePoints(n, NumTasks, totalPoints);
    int points = totalPoints[SelfTID];
    int offset = 0;

    double *X = (double *) malloc(points * d * sizeof(double));
    for (int i = 0; i < SelfTID; i++) {
        offset += totalPoints[i];
    }
    X = x + offset * d;

    int sentElements = totalPoints[SelfTID] * d;
    MPI_Isend(X, sentElements, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq);  //send self block

    int sender = findSender(SelfTID, NumTasks);
    int receivedArrayIndex, receivedElements; //initialize variables for use in loop
    int loopOffset;

    struct knnresult result, previousResult, newResult, mergedResult;
    result = smallKNN(X, X, points, points, d, k, offset);
    previousResult = result;
    mergedResult = result;

    for (int i = 1; i < NumTasks; i++) {  //ring communication

        receivedArrayIndex = findBlockArrayIndex(SelfTID, i, NumTasks);
        receivedElements = totalPoints[receivedArrayIndex] * d;
        double *Y = malloc(receivedElements * sizeof(double));

        MPI_Recv(Y, receivedElements, MPI_DOUBLE, sender, 55, MPI_COMM_WORLD, &mpistat); //receive from previous process
        MPI_Isend(Y, receivedElements, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq); //send the received array to next process

        loopOffset = findIndexOffset(SelfTID, i, NumTasks, totalPoints);
        newResult = smallKNN(X, Y, points, totalPoints[receivedArrayIndex], d, k, loopOffset);
        mergedResult = updateKNN(newResult, previousResult);
        previousResult = mergedResult;
    }

    return mergedResult;
}

struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset) {
    struct knnresult *result = malloc(sizeof(struct knnresult));
    int *indexesRowMajor = (int *) malloc(n * k * sizeof(int));
    double *distRowMajor = (double *) malloc(n * k * sizeof(double));

    for (int i = 0; i < n * k; i++) {
        distRowMajor[i] = INFINITY;
        indexesRowMajor[i] = -1;
    }

    double *xx = malloc(n * d * sizeof(double));
    hadamardProduct(x, x, xx, n * d);

    double *xxSum = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        xxSum[i] = cblas_dasum(d, xx + i * d, 1);
    }   

    int kMin = k > n ? n : k;
    kMin = kMin > m ? m : kMin;
    
    
    int MAX_Y_SIZE = 3;
    for (int ii = 0; ii < ceil(m / (double) MAX_Y_SIZE); ii++) {
        int partitionSize = MAX_Y_SIZE;
        if (ii * MAX_Y_SIZE + partitionSize > m) {
            partitionSize = m % MAX_Y_SIZE;
        }

        double *currentY = malloc(partitionSize * d * sizeof(double));
        for (int i = 0; i < partitionSize * d; i++) {
            currentY[i] = y[ii * MAX_Y_SIZE * d + i];
        }

        double *yy = malloc(partitionSize * d * sizeof(double));
        hadamardProduct(currentY, currentY, yy, partitionSize * d);

        double *yySum = malloc(partitionSize * sizeof(double));
        for (int i = 0; i < partitionSize; i++) {
            yySum[i] = cblas_dasum(d, yy + i * d, 1);
        }

        double *xy = malloc(n * partitionSize * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, partitionSize, d, -2, x, d, currentY, d, 0, xy, partitionSize);

        double *dist = malloc(n * partitionSize * sizeof(double));
        for (int j = 0; j < partitionSize; j++) {
            for (int i = 0; i < n; i++) {
                double distanceSquared = xxSum[i] + xy[i * partitionSize + j] + yySum[j];
                if (distanceSquared <= 0)
                    dist[j * n + i] = 0;
                else
                    dist[j * n + i] = sqrt(distanceSquared);
            }
        }

        int *indexValues = (int *) malloc(n * partitionSize * sizeof(int));
        for (int i = 0; i < n * partitionSize; i++)
            indexValues[i] = ii * MAX_Y_SIZE * k + i +indexOffset;

        for (int i = 0; i < partitionSize; i++) {
            int left=i*n;
            int right=(i+1)*n;
            int idx;
            int pivot;
            for (int j = kMin ; j > 0; j--) {
                distRowMajor[ii * MAX_Y_SIZE * k + i * k + j-1] = kNearestWithPivot(dist, indexValues, left, right - 1, j , &idx, &pivot);
                indexesRowMajor[ii * MAX_Y_SIZE * k + i * k + j-1] = idx - i * n;
                //printf("left is %d, right is %d, pivot is %d, idx is %d\n",left, right,pivot,idx);
                right=pivot;
            }
        }

        free(currentY);
        free(yy);
        free(xy);
        free(yySum);
        free(dist);
        free(indexValues);
    }

    result->ndist = distRowMajor;
    result->nidx = indexesRowMajor;
    result->m = n;
    result->k = k;

    free(xx);
    free(xxSum);
    return *result;
}
