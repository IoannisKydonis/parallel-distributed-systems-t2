#include <stdio.h>
#include <stdlib.h>
#include <math.h> // sqrt
#include <cblas.h> // cblas_dgemm

struct knnresult kNN(double *x, double *y, int n, int m, int d, int k);

void hadamardProduct(double *x, double *y, double *res, int length);

double  kNearest(double* D, int left, int right, int k, int * index, int m );

int partition(double* D, int left, int right, int pivotIndex); //partitioning function for kNearest

void swap(double* D , int left, int right); //swap function for kNearest

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

    kNN(x, y, 10, 5, 2, 0);   //k is the last value. If changed here , delete line 86
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

//// set value  of k ////
    k=3;
    int * index;
    int ** indexes=(int**)malloc(n*sizeof(int*));
    double** nearest=(double**)malloc(n*sizeof(double*));

    for (int i = 0; i < n; i++){
    nearest[i]=(double*)malloc(k*sizeof(double));
    indexes[i]=(int*)malloc(k*sizeof(int));
    }

    for (int i = 0; i < n; i++) {
           for (int j = 0; j < k; j++) {
               nearest[i][j] = kNearest(dist,i*m,(i+1)*m-1,j,&index,m);
               indexes[i][j]=index;

            }
        
    }
    

    for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
            printf("%10.4f(%d) ", nearest[i][j],indexes[i][j]);
            }
            printf("\n");
    }
    printf("\n");

    free(xx);
    free(yy);
    free(xy);
    free(xxSum);
    free(yySum);
    free(dist);
    free(nearest);
    free(indexes);

    return *result;
}

void hadamardProduct(double *x, double *y, double *res, int length) {
    for (int i = 0; i < length; i++)
        res[i] = x[i] * y[i];
}

double  kNearest(double* D, int left, int right, int k , int* index, int m){ // find the k-nearest neighbor for each element in D. The index of the nearest neighbor is D[index] which is returned.
                                                                             // *this function returns only 1 neighbor*
if (left==right){                                                            // m is used to find the correct index
    * index=left % m;
    //* index=left ;
    return D[left];
}
int pivotIndex=(right+left)/2 ;                                //set random value. Normally it should be ""left + floor(rand() % (right − left + 1))"" but this results in floating point error
pivotIndex=partition(D , left , right , pivotIndex );

if(k+left == pivotIndex){                        //normally it would be k == pivotIndex but we have to account for the cases that left>0
    * index= (k+left) % m;
    return D[k+left];                                  
}
else if ( k+left < pivotIndex )    
    return kNearest(D , left , pivotIndex - 1 , k, index, m);

else
    return kNearest(D , pivotIndex + 1 , right , k, index, m);
}


int partition(double* D, int left, int right, int pivotIndex){  //partitioning algorithm
int pivotValue=D[pivotIndex];
swap(D,pivotIndex, right);

int storeIndex=left;
for(int i=left; i<right; i++){
  if(D[i]<pivotValue){
     swap(D,storeIndex,i);
     storeIndex++;
  }
}

swap(D,right,storeIndex);
return storeIndex;
}


void swap(double* D , int left, int right){
   double temp=D[left];
   D[left]=D[right];
   D[right]=temp;

}



