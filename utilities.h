
#ifndef UTILITIES_H
#define UTILITIES_H

typedef struct vpNode {
    struct vpNode *parent;
    double *vp;
    int vpIdx;
    double mu;
    struct vpNode *left;
    struct vpNode *right;
} vpNode;

typedef struct knnresult
{
    int *nidx;     //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist; //!< Distance of nearest neighbors          [m-by-k]
    int m;         //!< Number of query points                 [scalar]
    int k;         //!< Number of nearest neighbors            [scalar]
}knnresult;

void *serializeKnnResult(knnresult res);

void hadamardProduct(double *x, double *y, double *res, int length);

double kNearest(double *dist, int *indexValues, int l, int r, int k, int *idx);

int partition(double *dist, int *indexValues, int l, int r);

double kNearestWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right, int k, int *idx);

int partitionWithOffsets(double *dist, int *indexValues, int *offsets, int left, int right);

void swap(double *n1, double *n2);

void swapInts(int *n1, int *n2);

void printResult(knnresult result);

struct knnresult smallKNN(double *x, double *y, int n, int m, int d, int k, int indexOffset);

struct knnresult updateKNN(struct knnresult oldResult, struct knnresult newResult );

void dividePoints(int n, int tasks, int * array);

int findDestination(int id, int NumTasks);

int findSender(int id, int NumTasks);

int findBlockArrayIndex(int id, int iteration, int NumTasks);

int findIndexOffset(int id, int iteration, int NumTasks, int * totalPoints);

double findDistance (double * point1, double * point2, int d);

double findMedian(double *distances, int *indexValues, int *offsets, int n, int *idx);

void printNode(vpNode * node,int d);

vpNode *createVPTree(double *array, double *x, int n, int d, int *indexValues, vpNode *parent, int *offsets);

void printTree (vpNode * root,int d);


#endif //UTILITIES_H
