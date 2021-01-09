#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"
#include <mpi.h>
#include <float.h>

vpNode *createVPTree(double *array, double *x, int n, int d, int *indexValues, vpNode *parent, int offset) {  //x represents a single block of elements

    vpNode *root = (vpNode *) malloc(sizeof(vpNode));
    root->parent = parent;

    double *distances = (double *) malloc(n * sizeof(double));
    int *localIndexValues = (int *) malloc(n * sizeof(int));

//int offset=0; //to split elements to the subtrees

    for (int i = 0; i < n; i++) {
        distances[i] = findDistance(x, x + i * d, d);
    }

    int idx;
    double median = findMedian(distances, indexValues, n, &idx);  //after this distances is partitioned from kNearest function

    root->mu = median;
    root->vpIdx = indexValues[0];
    root->vp = (double *) malloc(d * sizeof(double));
    for (int j = 0; j < d; j++) {
        root->vp[j] = x[j];
    }

//printNode(root,d);


    if (n > 1) { //to save time from the mallocs and initializations
        double *leftElements = (double *) malloc(sizeof(double));
        double *rightElements = (double *) malloc(sizeof(double));
        int *leftIndexes = (int *) malloc(sizeof(int));
        int *rightIndexes = (int *) malloc(sizeof(int));
        int leftSize = 1;
        int rightSize = 1;
        int flag = 0;

        for (int i = 1; i < n; i++) {
            if (distances[i] > median) {
                rightElements = (double *) realloc(rightElements, rightSize * d * sizeof(double));
                rightIndexes = (int *) realloc(rightIndexes, rightSize * sizeof(int));

                for (int j = 0; j < d; j++)
                    rightElements[(rightSize - 1) * d + j] = array[(indexValues[i] - offset) * d + j];
                rightIndexes[rightSize - 1] = indexValues[i];
                rightSize++;
            } else {

                leftElements = (double *) realloc(leftElements, leftSize * d * sizeof(double));
                leftIndexes = (int *) realloc(leftIndexes, leftSize * sizeof(int));

                for (int j = 0; j < d; j++)
                    leftElements[(leftSize - 1) * d + j] = array[(indexValues[i] - offset) * d + j];
                leftIndexes[leftSize - 1] = indexValues[i];
                leftSize++;
            }
        }

        leftSize--;
        rightSize--;
/*
printf("Left Subtree: \n");
for(int i=0; i<leftSize; i++){
   for(int j=0; j<d; j++)
      printf("%f ",leftElements[i*d+j]);
   printf(" (%d) - %f\n", leftIndexes[i],distances[i+1] );
}
printf("\n");

printf("Right Subtree: \n");
for(int i=0; i<rightSize; i++){
   for(int j=0; j<d; j++)
      printf("%f ",rightElements[i*d+j]);
   printf(" (%d) - %f\n", rightIndexes[i],distances[i+1+leftSize] );
}
printf("\n");
*/


        if (leftSize > 0) {
            root->left = createVPTree(array, leftElements, leftSize, d, leftIndexes, root, offset);
        }

        if (rightSize > 0) {
            root->right = createVPTree(array, rightElements, rightSize, d, rightIndexes, root, offset);
        }

    } else {
        root->left = NULL;
        root->right = NULL;
    }

    return root;

}

double findDistance(double *point1, double *point2, int d) {
    double distance = 0;
    for (int i = 0; i < d; i++) {
        distance += pow((point1[i] - point2[i]), 2);
    }
    return sqrt(distance);
}

double findMedian(double *distances, int *indexValues, int n, int *idx) {
    double median;
    if (n % 2 == 0) {
        int idx2;
        median = kNearest(distances, indexValues, 0, n - 1, n / 2, idx) +
                 kNearest(distances, indexValues, 0, n - 1, n / 2 + 1, &idx2);
        median /= 2;
    } else {
        median = kNearest(distances, indexValues, 0, n - 1, n / 2 + 1, idx);
    }

    return median;
}

void printNode(vpNode *node, int d) {
    printf("**NODE**\n");
    printf("mu is %f\n", node->mu);
    printf("Index is %d\n", node->vpIdx);
    printf("Cords: ");
    for (int i = 0; i < d; i++)
        printf("%f ", node->vp[i]);
    printf("\n\n");

}

int isPresent(int *nidx, int target, int k) {
    for (int i = 0; i < k; i++) {
        if (nidx[i] == target)
            return 1;
    }
    return 0;
}

void searchVpt(vpNode *node, double *ndist, int *nidx, int d, int k, double *x) {
    if (node == NULL)
        return;

    double dist = findDistance(node->vp, x, d);

    for (int i = 0; i < k; i++) {
        if (dist < ndist[i] && !isPresent(nidx, node->vpIdx, k)) {
            for (int j = k - 1; j > i; j--) {
                ndist[j] = ndist[j - 1];
                nidx[j] = nidx[j - 1];
            }
            ndist[i] = dist;
            nidx[i] = node->vpIdx;
        }
    }
    double tau = ndist[k - 1];
    if (dist <= node->mu + tau)
        searchVpt(node->left, ndist, nidx, d, k, x);
    if (dist > node->mu - tau)
        searchVpt(node->right, ndist, nidx, d, k, x);
}

void printTree(vpNode *root, int d) {
    printNode(root, d);


    printf("Left Point of %d: \n", root->vpIdx);
    if (root->left != NULL)
        printTree(root->left, d);


    printf("Right Point of %d: \n", root->vpIdx);
    if (root->right != NULL)
        printTree(root->right, d);


}

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

    double y[14] = {
            -45.0, 10.1,
            28.4, 31.329,
            0.7, 1.4,
            -8.4, -1.9,
            15, 7.2,
            -4.0, 1.1,
            8.4, -31.3};
    d = 2;
    k = 3;
    n = 15;


    int SelfTID, NumTasks, t;
    MPI_Status mpistat;
    MPI_Request mpireq;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    int *totalPoints = (int *) malloc(n * sizeof(int));
    dividePoints(n, NumTasks, totalPoints);
    int points = totalPoints[SelfTID];
    int offset = 0;

    double *X = (double *) malloc(points * d * sizeof(double));
    int min = totalPoints[0];
    for (int i = 0; i < SelfTID; i++) {
        offset += totalPoints[i];
        if (totalPoints[i] < min)
            min = totalPoints[i];
    }
    X = x + offset * d;
    vpNode *root = (vpNode *) malloc(sizeof(vpNode));


    int elements = totalPoints[SelfTID];
    int *indexValues = (int *) malloc(elements * sizeof(int));
    for (int i = 0; i < elements; i++) {
        indexValues[i] = i + offset;
        for (int j = 0; j < d; j++)
            //printf("%f ", X[i*d+j]);
            printf("(%d)", indexValues[i]);
    }

    int *indexes = (int *) malloc(8 * sizeof(int));
    for (int i = 0; i < 8; i++) {
        indexes[i] = i;
    }

    printf("Elements: %d\n", elements);
    root = createVPTree(X, X, elements, d, indexValues, NULL, offset);
    //root=createVPTree(y,y,8,2,indexes , NULL);
    if (SelfTID == 1)
        printTree(root, d);
    /*
    for(int i =0 ; i < n ; i++){
       printf("Dist from %d: ",i);
       for(int j=0; j < n; j++)
          printf("%f(%d) ",findDistance(x + i*d,x+ j*d,d),j);
       printf("\n");
    }

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
    if(SelfTID==0)
    printResult(mergedResult, SelfTID);
    */

    double *xx = (double *) malloc(d * sizeof(double));
    xx[0] = 1.5;
    xx[1] = 7.2;
    double *ndist = (double *) malloc(k * sizeof(double));
    int *nidx = (int *) malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        ndist[i] = DBL_MAX;
        nidx[i] = -1;
    }

    searchVpt(root, ndist, nidx, d, k, xx);
    for (int i = 0; i < k; i++) {
        printf("%16.8lf(%d)\n", ndist[i], nidx[i]);
    }

    MPI_Finalize();
    return (0);
}