#include "utilities.h"

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
    int x = dist[right];
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
