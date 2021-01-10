#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"
#include <mpi.h>
#include <float.h>

vpNode *createVPTree(double *array, double *x, int n, int d, int *indexValues, vpNode *parent, int *offsets) {  //x represents a single block of elements
    vpNode *root = (vpNode *) malloc(sizeof(vpNode));
    root->parent = parent;

    double *distances = (double *) malloc(n * sizeof(double));

    for (int i = 0; i < n; i++)
        distances[i] = findDistance(x, x + i * d, d);

    int idx;
    double median = findMedian(distances, indexValues, offsets, n, &idx);  //after this distances is partitioned from kNearest function

    root->mu = median;
    root->vpIdx = indexValues[0];
    root->vp = (double *) malloc(d * sizeof(double));
//    printf("First element: %2d, off=%2d, final=%2d\n", indexValues[0], offsets[0], indexValues[0] - offsets[0]);
    for (int j = 0; j < d; j++)
        root->vp[j] = x[j]; // TODO: verify correctness

    double *leftElements = (double *) malloc(0);
    double *rightElements = (double *) malloc(0);
    int *leftIndexes = (int *) malloc(0);
    int *rightIndexes = (int *) malloc(0);
    int leftSize = 0;
    int rightSize = 0;

    for (int i = 1; i < n; i++) {
        if (distances[i] > median) {
            rightSize++;
            rightElements = (double *) realloc(rightElements, rightSize * d * sizeof(double));
            rightIndexes = (int *) realloc(rightIndexes, rightSize * sizeof(int));

            for (int j = 0; j < d; j++)
                rightElements[(rightSize - 1) * d + j] = array[(indexValues[i] - offsets[i]) * d + j];
            rightIndexes[rightSize - 1] = indexValues[i];
        } else {
            leftSize++;
            leftElements = (double *) realloc(leftElements, leftSize * d * sizeof(double));
            leftIndexes = (int *) realloc(leftIndexes, leftSize * sizeof(int));

            for (int j = 0; j < d; j++)
                leftElements[(leftSize - 1) * d + j] = array[(indexValues[i] - offsets[i]) * d + j];
            leftIndexes[leftSize - 1] = indexValues[i];
        }
    }

    if (leftSize > 0)
        root->left = createVPTree(array, leftElements, leftSize, d, leftIndexes, root, offsets);
    else
        root->left = NULL;
    if (rightSize > 0)
        root->right = createVPTree(array, rightElements, rightSize, d, rightIndexes, root, offsets);
    else
        root->right = NULL;

    return root;
}

double findMedian(double *distances, int *indexValues, int *offsets, int n, int *idx) {
    if (n % 2 == 0) {
        int idx2;
        return (kNearestWithOffsets(distances, indexValues, offsets, 0, n - 1, n / 2, idx) +
                kNearestWithOffsets(distances, indexValues, offsets, 0, n - 1, n / 2 + 1, &idx2)) / 2;
    } else {
        return kNearestWithOffsets(distances, indexValues, offsets, 0, n - 1, n / 2 + 1, idx);
    }
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

void insertValueToResult(knnresult *result, double value, int idx, int position, int offset) {
    for (int j = offset + result->k - 1; j > position; j--) {
        result->ndist[j] = result->ndist[j - 1];
        result->nidx[j] = result->nidx[j - 1];
    }
    result->ndist[position] = value;
    result->nidx[position] = idx;
}

void searchVpt(vpNode *node, knnresult *result, int d, double *x, int offset) {
    if (node == NULL)
        return;

    double dist = findDistance(node->vp, x, d);

    for (int i = 0; i < result->k; i++) {
        if (dist < result->ndist[offset + i]) {
            insertValueToResult(result, dist, node->vpIdx, offset + i, offset);
            break;
        }
    }
    double tau = result->ndist[result->k + offset - 1];
    if (dist <= node->mu + tau)
        searchVpt(node->left, result, d, x, offset);
    if (dist >= node->mu - tau)
        searchVpt(node->right, result, d, x, offset);
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

double *mergeArrays(double *arr1, double *arr2, int len1, int len2) {
    double *merged = (double *)malloc((len1 + len2) * sizeof(double));
    for (int i = 0; i < len1; i++)
        merged[i] = arr1[i];
    for (int i = 0; i < len2; i++)
        merged[len1 + i] = arr2[i];
    return merged;
}

int *mergeIntArrays(int *arr1, int *arr2, int len1, int len2) {
    int *merged = (int *)malloc((len1 + len2) * sizeof(int));
    for (int i = 0; i < len1; i++)
        merged[i] = arr1[i];
    for (int i = 0; i < len2; i++)
        merged[len1 + i] = arr2[i];
    return merged;
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

    d = 2;
    k = 6;
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
    for (int i = 0; i < SelfTID; i++) {
        offset += totalPoints[i];
    }
    X = x + offset * d;
    vpNode *root = (vpNode *) malloc(sizeof(vpNode));

    int elements = totalPoints[SelfTID];
    int *indexValues = (int *) malloc(elements * sizeof(int));
    int *offsets = (int *) malloc(elements * sizeof(int));
    for (int i = 0; i < elements; i++) {
        indexValues[i] = i + offset;
        offsets[i] = offset;
    }

    root = createVPTree(X, X, elements, d, indexValues, NULL, offsets);

    int sentElements = totalPoints[SelfTID] * d;
    MPI_Isend(X, sentElements, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq);  //send self block

    int sender = findSender(SelfTID, NumTasks);
    int receivedArrayIndex;
    int receivedElements;
    struct knnresult newResult;
    int off;
    struct knnresult mergedResult;

    struct knnresult *result = (struct knnresult *)malloc(sizeof(struct knnresult));
    result->ndist = (double *)malloc(elements * k * sizeof(double));
    result->nidx = (int *)malloc(elements * k * sizeof(int));
    result->k = k;
    result->m = elements;
    for (int ii = 0; ii < result->m * k; ii++) {
        result->ndist[ii] = INFINITY;
        result->nidx[ii] = -1;
    }

    struct knnresult previousResult;

    for (int i = 0; i < elements; i++) {
        searchVpt(root, result, d, X + (i) * d, (i) * k);
    }
    previousResult = *result;
    mergedResult = previousResult;

    for (int i = 1; i < NumTasks; i++) {  //loop

        receivedArrayIndex = findBlockArrayIndex(SelfTID, i, NumTasks);
        receivedElements = totalPoints[receivedArrayIndex] * d;
        double *Y = malloc(receivedElements * sizeof(double));
        MPI_Recv(Y, receivedElements, MPI_DOUBLE, sender, 55, MPI_COMM_WORLD, &mpistat); //receive from previous process
        MPI_Isend(Y, receivedElements, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq); //send the received array to next process

        off = findIndexOffset(SelfTID, i, NumTasks, totalPoints);
        int elements2 = totalPoints[receivedArrayIndex];
        int *indexValues2 = (int *) malloc((elements + elements2) * sizeof(int));
        int *offsets2 = (int *) malloc((elements + elements2) * sizeof(int));
        for (int ii = 0; ii < elements; ii++) {
            indexValues2[ii] = ii + offset;
            offsets2[ii] = offset;
        }
        for (int ii = 0; ii < elements2; ii++) {
            indexValues2[elements + ii] = ii + off;
            offsets2[elements + ii] = off - elements;
        }

//        if (SelfTID == 0) {
//            for (int ii = 0; ii < elements + elements2; ii++) {
//                printf("index=%d, offset=%d, final=%d\n", indexValues2[ii], offsets2[ii], indexValues2[ii] - offsets2[ii]);
//            }
//            printf("\n");
//        }

        double *merged = mergeArrays(X, Y, sentElements, receivedElements);
        root = createVPTree(merged, merged, elements + elements2, d, indexValues2, NULL, offsets2);

        struct knnresult *res = (struct knnresult *)malloc(sizeof(struct knnresult));
        res->ndist = (double *)malloc(elements * k * sizeof(double));
        res->nidx = (int *)malloc(elements * k * sizeof(int));
        res->k = k;
        res->m = elements;
        for (int ii = 0; ii < elements * k; ii++) {
            res->ndist[ii] = INFINITY;
            res->nidx[ii] = -1;
        }

        for (int ii = 0; ii < elements; ii++) {
            searchVpt(root, res, d, X + (ii) * d, (ii) * k);
        }

        mergedResult = updateKNN(*res, previousResult);
        previousResult = mergedResult;
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
    return 0;
}
