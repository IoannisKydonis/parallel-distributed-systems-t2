#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt, pow
#include <cblas.h> // cblas_dgemm
#include <mpi.h>
#include <float.h> //DBL_MAX
#include <time.h> //rand
#include "utilities.h"
#include "types.h"
#include "controller.h"

knnresult distrAllkNN(double *x, int n, int d, int k);

vpNode *createVPTree(double *array, double *x, int n, int d, int *indexValues, vpNode *parent, int *offsets);

void searchVpt(vpNode *node, knnresult *result, int d, double *x, int offset);

int main(int argc, char *argv[]) {
    int SelfTID, NumTasks;
    MPI_Status mpistat;
    MPI_Request mpireq;  //initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    int n = atoi(argv[2]);
    int d = atoi(argv[3]);
    int k = atoi(argv[4]);
    double * X=(double *)malloc(n * d * sizeof(double));
    int numbered=1; //1 for asc, 2 for 1:123 2:22324, 3 else
        printf("n is %d, d is %d, k is %d\n",n,d,k);


FILE *matFile=fopen(argv[1],"r");
		if (matFile == NULL)
		{
			printf("Couldn't open file\n");
			exit(1);
		}
		double num,temp;
		int i=0,j=0;
		if(numbered==1){
			while ( fscanf(matFile,"%lf",&num) !=EOF )
			{
				if(i%(d+1)!=0){
					X[j]=num;
					j++;
				}
				i++;
			}
		}else if(numbered==2){
			for(i=0; i<n; i++){
				fscanf(matFile,"%lf",&num);
				for(j=0; j<d; j++){
					if(fscanf(matFile," %lf:%lf",&temp,&num)==EOF) break;
					X[i*d+j]=num;
        		}
				fscanf(matFile,"%*[^\n]\n");
			}
		}else{
			for(int skip=0;skip<4;skip++){
				fscanf(matFile,"%*[^\n]\n");
			}
			for(i=0; i<n; i++){
				fscanf(matFile,"%lf",&num);
				for(j=0; j<d; j++){
					if(fscanf(matFile,",%lf",&num)==EOF) break;
					X[i*d+j]=num;
        		}
				fscanf(matFile,"%*[^\n]\n");
			}
		}
	fclose(matFile);

    printf("Exited read\n");
    // if(SelfTID==0)
    // for(int i=0; i < 32 ; i++){
    //     printf("%f ",X[i]);
    // }
    // printf("Exited print\n");
     n=10000; //change size for faster execution
     d=2; //change dimensions for faster execution

    char *filename = (char *) malloc(16 * sizeof(char));
    sprintf(filename, "v2_res_%04d.txt", SelfTID);
    knnresult mergedResult = runAndPresentResult(distrAllkNN, X, n, d, k, "v2", "v2_out.txt", filename);
    free(filename);

    if (SelfTID == 0) {    //send every result to the first process for printing
        FILE *f = fopen("v2_out.txt", "wb");
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
    int elements = totalPoints[SelfTID]; //find elements for self block
    int offset = 0;

    double *X = (double *) malloc(elements * d * sizeof(double));  //each process takes a block from X
    for (int i = 0; i < SelfTID; i++) {
        offset += totalPoints[i];
    }
    X = x + offset * d;
    vpNode *root = (vpNode *) malloc(sizeof(vpNode));

    int *indexValues = (int *) malloc(elements * sizeof(int));  //initialize offsets and indexes
    int *offsets = (int *) malloc(elements * sizeof(int));
    for (int i = 0; i < elements; i++) {
        indexValues[i] = i + offset;
        offsets[i] = offset;
    }

    root = createVPTree(X, X, elements, d, indexValues, NULL, offsets);  //create local vptree

    free(indexValues);
    free(offsets);

    int sentElements = elements * d;
    MPI_Isend(X, sentElements, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq);  //send self block

    int sender = findSender(SelfTID, NumTasks);
    int receivedArrayIndex, receivedElements;  //variables for later use in ring communication
    struct knnresult newResult, mergedResult, previousResult;

    struct knnresult *result = (struct knnresult *) malloc(sizeof(struct knnresult));  //initialize result for local vptree
    initializeResult(result, elements, k);

    for (int i = 0; i < elements; i++) {
        searchVpt(root, result, d, X + (i) * d, (i) * k);  //search knearest for each point
    }
    previousResult = *result;
    mergedResult = previousResult;

    for (int i = 1; i < NumTasks; i++) {  //ring communication

        receivedArrayIndex = findBlockArrayIndex(SelfTID, i, NumTasks);
        receivedElements = totalPoints[receivedArrayIndex];
        double *Y = malloc(receivedElements * d * sizeof(double));
        MPI_Recv(Y, receivedElements * d, MPI_DOUBLE, sender, 55, MPI_COMM_WORLD, &mpistat); //receive from previous process
        MPI_Isend(Y, receivedElements * d, MPI_DOUBLE, findDestination(SelfTID, NumTasks), 55, MPI_COMM_WORLD, &mpireq); //send the received array to next process

        int loopOffset = findIndexOffset(SelfTID, i, NumTasks, totalPoints);

        int *mergedIndexes = (int *) malloc(
                (elements + receivedElements) * sizeof(int));  //initialize merged indexes and offsets
        int *mergedOffsets = (int *) malloc((elements + receivedElements) * sizeof(int));

        for (int ii = 0; ii < elements; ii++) {
            mergedIndexes[ii] = ii + offset;
            mergedOffsets[ii] = offset;
        }
        for (int ii = 0; ii < receivedElements; ii++) {
            mergedIndexes[elements + ii] = ii + loopOffset;
            mergedOffsets[elements + ii] = loopOffset - elements;
        }

        double *mergedArray = mergeArrays(X, Y, sentElements, receivedElements * d);
        root = createVPTree(mergedArray, mergedArray, elements + receivedElements, d, mergedIndexes, NULL, mergedOffsets); //create vptree for merged array

        struct knnresult *newResult = (struct knnresult *) malloc(sizeof(struct knnresult));  //initialize struct
        initializeResult(newResult, elements, k);

        for (int ii = 0; ii < elements; ii++) {
            searchVpt(root, newResult, d, X + (ii) * d, (ii) * k); //search knearest for each point
        }

        mergedResult = updateKNN(*newResult, previousResult);  //update neighbors
        previousResult = mergedResult;

        free(Y);
        free(mergedIndexes);
        free(mergedOffsets);
    }

    return mergedResult;
}

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
    for (int j = 0; j < d; j++)
        root->vp[j] = array[(indexValues[0] - offsets[0]) * d + j];

    double *leftElements = (double *) malloc(0);
    double *rightElements = (double *) malloc(0);
    int *leftIndexes = (int *) malloc(0);
    int *rightIndexes = (int *) malloc(0);
    int *leftOffsets = (int *) malloc(0);
    int *rightOffsets = (int *) malloc(0);
    int leftSize = 0;
    int rightSize = 0;

    for (int i = 1; i < n; i++) {
        if (distances[i] > median) {
            rightSize++;
            rightElements = (double *) realloc(rightElements, rightSize * d * sizeof(double));
            rightIndexes = (int *) realloc(rightIndexes, rightSize * sizeof(int));
            rightOffsets = (int *) realloc(rightOffsets, rightSize * sizeof(int));

            for (int j = 0; j < d; j++)
                rightElements[(rightSize - 1) * d + j] = array[(indexValues[i] - offsets[i]) * d + j];
            rightIndexes[rightSize - 1] = indexValues[i];
            rightOffsets[rightSize - 1] = offsets[i];
        } else {
            leftSize++;
            leftElements = (double *) realloc(leftElements, leftSize * d * sizeof(double));
            leftIndexes = (int *) realloc(leftIndexes, leftSize * sizeof(int));
            leftOffsets = (int *) realloc(leftOffsets, leftSize * sizeof(int));

            for (int j = 0; j < d; j++)
                leftElements[(leftSize - 1) * d + j] = array[(indexValues[i] - offsets[i]) * d + j];
            leftIndexes[leftSize - 1] = indexValues[i];
            leftOffsets[leftSize - 1] = offsets[i];
        }
    }
    root->left = NULL;
    root->right = NULL;

    if (leftSize > 0)
        root->left = createVPTree(array, leftElements, leftSize, d, leftIndexes, root, leftOffsets);
    if (rightSize > 0)
        root->right = createVPTree(array, rightElements, rightSize, d, rightIndexes, root, rightOffsets);

    free(leftElements);
    free(rightElements);
    free(leftIndexes);
    free(rightIndexes);
    free(leftOffsets);
    free(rightOffsets);
    free(distances);

    return root;
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
