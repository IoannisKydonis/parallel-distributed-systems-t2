
#ifndef UTILITIES_H
#define UTILITIES_H

typedef struct knnresult
{
    int *nidx;     //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist; //!< Distance of nearest neighbors          [m-by-k]
    int m;         //!< Number of query points                 [scalar]
    int k;         //!< Number of nearest neighbors            [scalar]
}knnresult;

void *serializeKnnResult(knnresult res);

char *deserializeKnnResult(void *serialized);

void hadamardProduct(double *x, double *y, double *res, int length);

double kNearest(double *dist, int *indexValues, int l, int r, int k, int *idx);

int partition(double *dist, int *indexValues, int l, int r);

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


#endif //UTILITIES_H
