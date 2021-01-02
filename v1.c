#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"
#include <mpi.h>

struct knnresult
{
    int *nidx;     //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist; //!< Distance of nearest neighbors          [m-by-k]
    int m;         //!< Number of query points                 [scalar]
    int k;         //!< Number of nearest neighbors            [scalar]
};


void printResult(struct knnresult result, int id);

struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset);

struct knnresult updateKNN(struct knnresult oldResult, struct knnresult newResult );

void dividePoints(int n, int tasks, int * array);

int findDestination(int id, int NumTasks);

int findSender(int id, int NumTasks);

int findBlockArrayIndex(int id, int iteration, int NumTasks);

int findIndexOffset(int id, int iteration, int NumTasks, int * totalPoints);


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
        0.5, 1.2,
        8.4, 1.9,
        1.5, 7.2,
        -45.0, 10.1,
        28.4, 31.329};
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

   //printf("I am %d , I receive from %d and send to %d . \n", SelfTID,findSender(SelfTID, NumTasks),findDestination(SelfTID, NumTasks) );
   int sentElements=totalPoints[SelfTID]*d;
   MPI_Isend(X,sentElements,MPI_DOUBLE,findDestination(SelfTID, NumTasks),55,MPI_COMM_WORLD, &mpireq);  //send self block

   printf("Process %d sent to process %d array X: ",SelfTID,findDestination(SelfTID, NumTasks));
   for(int l=0; l<sentElements; l++)
   printf("%f ",X[l]);
   printf("\n");

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
   //previousResult=result;

   for (int i=1; i<NumTasks-1; i++){  //loop

   receivedArrayIndex=findBlockArrayIndex(SelfTID, i, NumTasks);
   receivedElements=totalPoints[receivedArrayIndex] * d;

   double * Y= malloc(receivedElements * sizeof(double));
   MPI_Recv(Y,receivedElements,MPI_DOUBLE,sender,55,MPI_COMM_WORLD,&mpistat); //receive from the last process

   printf("Process %d received from process %d and sends to process %d array Y: ",SelfTID,sender, findDestination(SelfTID, NumTasks));
   for(int l=0; l<receivedElements; l++)
   printf("%f ",Y[l]);
   printf("\n");

   MPI_Isend(Y,receivedElements,MPI_DOUBLE,findDestination(SelfTID, NumTasks),55,MPI_COMM_WORLD, &mpireq); //send the received array to next process
   
   off=findIndexOffset(SelfTID,i,NumTasks,totalPoints);
   //printf("offset is %d\n",off);
   newResult=smallKNN(X,Y,points,points,d,k,0);
   //newResult=smallKNN(X,Y,points,totalPoints[sender],d,k); don't know which is correct
   mergedResult=updateKNN(newResult,result);
   //previousResult=mergedResult;
   }


   //if(SelfTID==0);
   printResult(mergedResult, SelfTID);


   MPI_Finalize();
   //free(X)
   //free(totalPoints)

   
   
   return(0);
}






void printResult(struct knnresult result, int id){

   printf("Nearest %d: ",id);
    for(int i=0; i<result.m * result.k; i++){
        if (i % result.k == 0) {
            printf("\n");
        }
        printf("%f ", result.ndist[i]);
    }
    printf("\n");

    printf("Indexes %d: ",id);
    for(int i=0; i<result.m * result.k; i++){
        if (i % result.k == 0) {
            printf("\n");
        }
        printf("%d ", result.nidx[i]);
    }
    printf("\n");

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
            dist[i * m + j] = sqrt(xxSum[i] + xy[i * m + j] + yySum[j]);
        }
    }


    int *indexValues = (int *)malloc(n * m * sizeof(int));
    for (int i = 0; i < n * m; i++)   //this must change for Y block
        indexValues[i] = i+indexOffset;
    
    //printf("Indexes go from %d to %d with offset %d\n",indexValues[0],indexValues[n*m-1],indexOffset);

    int * indexesRowMajor=(int*)malloc(n*k*sizeof(int));
    double * distRowMajor=(double*)malloc(n*k*sizeof(double));    
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            int idx;
            distRowMajor[i*k +j] = kNearest(dist, indexValues, i * m, (i + 1) * m - 1, j + 1, &idx);
            indexesRowMajor[i*k +j] = idx - i * m;
            //printf("%f(%d) ",distRowMajor[i*k +j],indexesRowMajor[i*k +j]);
        }
       //printf("\n");
    }

    result->ndist=distRowMajor;
    result->nidx=indexesRowMajor;
    result->m=m;
    result->k=k;


    free(xx);
    free(yy);
    free(xy);
    free(xxSum);
    free(yySum);
    //free(dist);
    //free(indexValues);
    return *result;


}


void dividePoints(int n, int tasks, int * array){
   int points=n/tasks;
   for(int i=0; i<n; i++){
        array[i]=points;
   }

   int pointsLeft=n%tasks;
   for(int i=0; pointsLeft>0; i++){
        array[i]++;
        pointsLeft--;
   }

}

int findDestination(int id, int NumTasks){
    if (id==NumTasks-1)
    return 0;
    else
    return (id+1);
}

int findSender(int id, int NumTasks){
    if(id==0)
    return (NumTasks-1);
    else
    return (id-1);
}

struct knnresult updateKNN(struct knnresult oldResult, struct knnresult newResult ){
   struct knnresult *result = malloc(sizeof(struct knnresult));
   
   int k,kmin;
   int m=oldResult.m; //==newResult.m
   if(oldResult.k>newResult.k){
   k=oldResult.k;
   kmin=newResult.k;
   }
   else{
   k=newResult.k;
   kmin=oldResult.k;
   }

   double * newNearest = malloc(m * k* sizeof(double));
   int * newIndexes = malloc(m * k * sizeof(int));

   for(int i=0; i< m ; i++){
       int it1,it2; //iterator for old and new result.ndist
       it1=it2=i*k;
       for(int j =0; j<k ; j++){
           if(newResult.ndist[it1]<=oldResult.ndist[it2]){
               //printf("%f* is smaller than %f it1 %d it2 %d   -> %d\n",newResult.ndist[it1],oldResult.ndist[it2], it1, it2,i*k+j);
               newNearest[i*k+j]=newResult.ndist[it1];
               newIndexes[i*k+j]=newResult.nidx[it1];
               it1++;
           }
           else {
               //printf("%f is bigger than %f* it1 %d it2 %d   -> %d\n",newResult.ndist[it1],oldResult.ndist[it2], it1, it2,i*k+j);
               newNearest[i*k+j]=oldResult.ndist[it2];
               newIndexes[i*k+j]=oldResult.nidx[it2];
               it2++;
           }
           



     }
    }
    /*
    printf("New nearest\n");
    for(int i=0; i<m*k ; i++){
        if(i%k==0)
        printf("\n");
    printf("%f ",newNearest[i]);
    }
    printf("\nNew indexes \n");

    for(int i=0; i<m*k ; i++){
        if(i%k==0)
        printf("\n");
    printf("%d ",newIndexes[i]);
    }
    printf("\n");
    */
    result->ndist=newNearest; 
    result->nidx=newIndexes;
    result->m=oldResult.m;
    result->k=oldResult.k;
    return * result;

}

int findBlockArrayIndex(int id, int iteration, int NumTasks){ //iteration >=1
int Y=id-iteration;
if(Y<0)
Y+=NumTasks;
return Y;
}

int findIndexOffset(int id, int iteration, int NumTasks, int * totalPoints){ //total points is the number of the points before 
int Y=findBlockArrayIndex(id, iteration, NumTasks);
int result=0;
for(int i=0;i<Y;i++)
result+=totalPoints[i];

return result;
}

