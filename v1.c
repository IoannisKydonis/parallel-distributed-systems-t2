#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"
#include <mpi.h>


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
   k=3;
   n=15;

   int SelfTID, NumTasks, t;
   MPI_Status mpistat;
   MPI_Request mpireq;
   MPI_Init( &argc, &argv );
   MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
   MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );

   int * totalPoints=(int *)malloc(n*sizeof(int));
   dividePoints(n , NumTasks, totalPoints);
   int points=totalPoints[SelfTID];
   int offset=0;

   double * X=(double *)malloc(points*d*sizeof(double));
   int min=totalPoints[0];
   for(int i=0; i<SelfTID; i++){
   offset+=totalPoints[i];   
   if(totalPoints[i]<min)
   min=totalPoints[i];
   }
   X=x+offset*d;

   int sentElements=totalPoints[SelfTID]*d;
   MPI_Isend(X,sentElements,MPI_DOUBLE,findDestination(SelfTID, NumTasks),55,MPI_COMM_WORLD, &mpireq);  //send self block

   int sender=findSender(SelfTID, NumTasks);
   int receivedArrayIndex;
   int receivedElements;
   struct knnresult newResult;
   int off;
   struct knnresult mergedResult;

   if(k>min)
   k=min-1;
   struct knnresult result,previousResult;
   result=smallKNN(X,X,points,points,d,k,offset);
   previousResult=result;
   mergedResult=result;

   for (int i=1; i<NumTasks; i++){  //loop

   receivedArrayIndex=findBlockArrayIndex(SelfTID, i, NumTasks);
   receivedElements=totalPoints[receivedArrayIndex] * d;
   double * Y= malloc(receivedElements * sizeof(double));
   MPI_Recv(Y,receivedElements,MPI_DOUBLE,sender,55,MPI_COMM_WORLD,&mpistat); //receive from previous process
   MPI_Isend(Y,receivedElements,MPI_DOUBLE,findDestination(SelfTID, NumTasks),55,MPI_COMM_WORLD, &mpireq); //send the received array to next process

   off=findIndexOffset(SelfTID,i,NumTasks,totalPoints);
   newResult=smallKNN(X,Y,points,points,d,k,off);
   mergedResult=updateKNN(newResult,previousResult);
   previousResult=mergedResult;
   }

    if (SelfTID == 0) {
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

