#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"
#include <mpi.h>

struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset);

knnresult distrAllkNN(double * x,  int n, int d, int k);


int main(int argc, char *argv[]) {
    
    int n;
    int d; //= atoi(argv[1]);
    int k;
    double x[30] = {
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
    k=5;
    n=15;

    int SelfTID, NumTasks;
    MPI_Status mpistat;
    MPI_Request mpireq;  //initialize MPI environment
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
    MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );

    knnresult mergedResult;
    mergedResult=distrAllkNN(x,n,d,k);

    if (SelfTID == 0) {    //send every result to the first process for printing
        printResult(mergedResult);
        for (int i = 1; i < NumTasks; i++) {
            char *deserialized = malloc(n * k * 27 + n + 1);
            MPI_Recv(deserialized, n * k * 27 + n + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD, &mpistat);
            printf("%s", deserialized);
            free(deserialized);
        }
    } else {
        MPI_Isend(serializeKnnResult(mergedResult), n * k * 27 + n + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &mpireq);
    }

    MPI_Finalize();
    return(0);



}
struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset)
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
            double distanceSquared = xxSum[i] + xy[i * m + j] + yySum[j];
            if (distanceSquared <= 0)
                dist[i * m + j] = 0;
            else
                dist[i * m + j] = sqrt(distanceSquared);
        }
    }

    int *indexValues = (int *)malloc(n * m * sizeof(int));
    for (int i = 0; i < n * m; i++)
        indexValues[i] = i+indexOffset;

    int kMin = k > n ? n : k;
    kMin = kMin > m ? m : kMin;

    int * indexesRowMajor=(int*)malloc(n*k*sizeof(int));
    double * distRowMajor=(double*)malloc(n*k*sizeof(double));
    for (int i = 0; i < n * k; i++) {
        distRowMajor[i] = INFINITY;
        indexesRowMajor[i] = -1;
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < kMin; j++)
        {
            int idx;
            distRowMajor[i*k +j] = kNearest(dist, indexValues, i * m, (i + 1) * m - 1, j + 1, &idx);
            indexesRowMajor[i*k +j] = idx - i * m;
        }
    }

    result->ndist=distRowMajor;
    result->nidx=indexesRowMajor;
    result->m=n;
    result->k=k;

    free(xx);
    free(yy);
    free(xy);
    free(xxSum);
    free(yySum);
    return *result;
}

knnresult distrAllkNN(double * x , int n, int d, int k){

   int SelfTID, NumTasks;
   MPI_Status mpistat;
   MPI_Request mpireq;  //initialize MPI environment
   MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
   MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );

   int * totalPoints=(int *)malloc(n*sizeof(int));
   dividePoints(n , NumTasks, totalPoints);
   int points=totalPoints[SelfTID];
   int offset=0;

   double * X=(double *)malloc(points*d*sizeof(double));
   for(int i=0; i<SelfTID; i++){
   offset+=totalPoints[i];
   }
   X=x+offset*d;

   int sentElements=totalPoints[SelfTID]*d;
   MPI_Isend(X,sentElements,MPI_DOUBLE,findDestination(SelfTID, NumTasks),55,MPI_COMM_WORLD, &mpireq);  //send self block

   int sender=findSender(SelfTID, NumTasks);
   int receivedArrayIndex,receivedElements; //initialize variables for use in loop
   int loopOffset;

   struct knnresult result,previousResult,newResult,mergedResult;
   result=smallKNN(X,X,points,points,d,k,offset);
   previousResult=result;
   mergedResult=result;

   for (int i=1; i<NumTasks; i++){  //ring communication

   receivedArrayIndex=findBlockArrayIndex(SelfTID, i, NumTasks);
   receivedElements=totalPoints[receivedArrayIndex] * d;
   double * Y= malloc(receivedElements * sizeof(double));

   MPI_Recv(Y,receivedElements,MPI_DOUBLE,sender,55,MPI_COMM_WORLD,&mpistat); //receive from previous process
   MPI_Isend(Y,receivedElements,MPI_DOUBLE,findDestination(SelfTID, NumTasks),55,MPI_COMM_WORLD, &mpireq); //send the received array to next process

   loopOffset=findIndexOffset(SelfTID,i,NumTasks,totalPoints);
   newResult=smallKNN(X,Y,points,totalPoints[receivedArrayIndex],d,k,loopOffset);
   mergedResult=updateKNN(newResult,previousResult);
   previousResult=mergedResult;
   }

    return mergedResult;
}
