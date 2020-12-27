#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // sqrt
#include <cblas.h> // cblas_dgemm
#include "utilities.h"
#include <mpi.h>

int * customize (int * array, int id){
for(int i=0; i<2; i++){
   array[i]+=  pow(10,id);
}

}

int main( int argc, char** argv ) {
 //struct knnresult result;
    double X;
    int n;
    int d; //= atoi(argv[1]);
    int k;
    int ** array = (int **)malloc( 4 * sizeof(int*));
    for (int i=0; i<4;i++){
    array[i] = (int*)malloc(2*sizeof(int));
    array[i][0]=i+1;
    array[i][1]=i+1;
    }
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
        28.4, 31.329};

   int SelfTID, NumTasks, t;
   int * data = (int*)malloc(2*sizeof(int));
   MPI_Status mpistat;
   MPI_Request mpireq;
   MPI_Init( &argc, &argv );
   MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
   MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );

  customize(array[SelfTID],SelfTID);


   if(SelfTID==0){
   //printf("First, %d\n", SelfTID);
   MPI_Isend(array[SelfTID],2,MPI_INT,SelfTID+1,55,MPI_COMM_WORLD, &mpireq); 
   MPI_Recv(data,2,MPI_INT,NumTasks-1,55,MPI_COMM_WORLD,&mpistat); //receive from the last process
   }

   else if(SelfTID==NumTasks-1){
      //printf("Second, %d\n", SelfTID);
   MPI_Isend(array[SelfTID],2,MPI_INT,0,55,MPI_COMM_WORLD, &mpireq); //send to the first process
   MPI_Recv(data,2,MPI_INT,SelfTID-1,55,MPI_COMM_WORLD,&mpistat);
   }

   else{
     // printf("Third, %d\n", SelfTID);
   MPI_Isend(array[SelfTID],2,MPI_INT,SelfTID+1,55,MPI_COMM_WORLD, &mpireq);
   MPI_Recv(data,2,MPI_INT,SelfTID-1,55,MPI_COMM_WORLD,&mpistat);   
   }
   
   customize(data,SelfTID);
   printf("TID%i: customized data:",SelfTID);
   printf("[%d , %d]\n", data[0],data[1]);
   
   MPI_Finalize();
   return( 0 );
}