#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include <float.h>
#include "utilities.h"

void *serializeKnnResult(knnresult res) {
    char *serialized = (char *)malloc(res.m * res.k * 27 + res.m + 1);
    int ind = 0;
    for (int i = 0; i < res.m * res.k; i++) {
        if (i % res.k == 0 && i != 0)
            ind += sprintf(serialized + ind, "\n");
        ind += sprintf(serialized + ind, "%16.8lf(%08d) ", res.ndist[i], res.nidx[i]);
    }
    ind += sprintf(serialized + ind, "\n");

    *(serialized + ind) = '\0';
    return (void *)serialized;
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
    double x = dist[right];
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

double kNearestWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right, int k, int *idx)
{
    int pivot = partitionWithOffsets(dist, indexValues, offsets, left, right);

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

int partitionWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right)
{
    double x = dist[right];
    int i = left;
    for (int j = left; j <= right - 1; j++)
    {
        if (dist[j] <= x)
        {
            swap(&dist[i], &dist[j]);
            swapInts(&indexValues[i], &indexValues[j]);
            swapInts(&offsets[i], &offsets[j]);
            i++;
        }
    }
    swap(&dist[i], &dist[right]);
    swapInts(&indexValues[i], &indexValues[right]);
    swapInts(&offsets[i], &offsets[right]);
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

void printResult(knnresult result) {
    for(int i=0; i<result.m * result.k; i++){
        if (i % result.k == 0) {
            printf("\n");
        }
        printf("%16.8f(%08d) ", result.ndist[i], result.nidx[i]);
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
               newNearest[i*k+j]=newResult.ndist[it1];
               newIndexes[i*k+j]=newResult.nidx[it1];
               if (newResult.nidx[it1] == oldResult.nidx[it2])
                   it2++;
               it1++;
           }
           else {
               newNearest[i*k+j]=oldResult.ndist[it2];
               newIndexes[i*k+j]=oldResult.nidx[it2];
               if (newResult.nidx[it1] == oldResult.nidx[it2])
                   it1++;
               it2++;
           }
        }
    }

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


double findDistance(double *point1, double *point2, int d) {
    double distance = 0;
    for (int i = 0; i < d; i++)
        distance += pow((point1[i] - point2[i]), 2);
    return sqrt(distance);
}
