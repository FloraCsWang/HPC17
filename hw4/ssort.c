/* Parallel sample sort
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>


static int compare(const void *a, const void *b)
{
    int *da = (int *)a;
    int *db = (int *)b;
    
    if (*da > *db)
        return 1;
    else if (*da < *db)
        return -1;
    else
        return 0;
}

int main( int argc, char *argv[])
{
    int rank;
    int i, N;
    int *vec;
    int p;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status* status;
    
    // Number of random numbers per processor (this should be increased for actual tests or could be passed in through the command line
    N = 100;
    
    vec = calloc(N, sizeof(int));
    //seed random number generator differently on every core
    srand((unsigned int) (rank + 393919));
    
    // fill vector with random integers
    for (i = 0; i < N; ++i) {
        vec[i] = rand();
    }
    //printf("rank: %d, first entry: %d\n", rank, vec[0]);
    
    // sort locally
    qsort(vec, N, sizeof(int), compare);
    
    // randomly sample s entries from vector or select local splitters,i.e., every N/P-th entry of the sorted vector
    int s = 10;
    int* randomEntries = calloc(s, sizeof(int));
    for (i= 0; i < s ; i++){
        
        int index = rand()%N;
        randomEntries[i] = vec[index];
    }
    
    // every processor communicates the selected entries to the root processor; use for instance an MPI_Gather
    int root = 0;
    int *rbuf;
    rbuf = calloc(p * s, sizeof(int) );
    MPI_Gather( randomEntries, s, MPI_INT, rbuf, s, MPI_INT, root, MPI_COMM_WORLD);
    
    
    // root processor does a sort, determinates splitters that split the data into P buckets of approximately the same size root process broadcasts splitters
    int* splitters = calloc(p - 1, sizeof(int) );
    if (rank == root){
         qsort(rbuf, p * s, sizeof(int), compare);
        for(i = 0; i < p - 1; i++ ){
            splitters[i] = rbuf[s * (i + 1) - 1];
            //printf("this splitter is %d\n", splitters[i] );
        }
    }
    
    
    MPI_Bcast(splitters, p - 1, MPI_INT, root, MPI_COMM_WORLD);
    
    
    // every processor uses the obtained splitters to decidewhich integers need to be sent to which other processor (local bins)
    
    int * count = calloc(p, sizeof(int));
    int *buckets = calloc (N, sizeof(int));
    for (i = 0; i < N; i++ ){
        int r = 0;
        while (compare(vec+i,splitters+r) == 1 && r < (p - 1)){
            r++;
        }
        count[r] = count[r] + 1;
        buckets[i] = r;
        //MPI_Send(vec+i, 1, MPI_INT , r, 0 , MPI_COMM_WORLD);
    }
    
    //send and receive:
    
    int *sendBuffer = calloc(p * p, sizeof(int) );
    int * recBuffer = calloc(p * p, sizeof(int) );
    for (i = 0; i < p; i++){
        for (int j = 0; j < p; j++ ){
            sendBuffer[i*p+j] = count[j];
        }
    }
    MPI_Alltoall(sendBuffer, p, MPI_INT, recBuffer, p, MPI_INT, MPI_COMM_WORLD  );
    int totalNum = 0;
    for (i = 0; i < p; i++){
        totalNum +=  recBuffer[i*p + rank];
    }
    int * resVector = calloc(totalNum, sizeof(int));
    
    for ( i= 0; i < N; i++){
        MPI_Send(vec+i, 1, MPI_INT , buckets[i], 0 , MPI_COMM_WORLD);
    }
    
    for (i = 0; i < totalNum; i++ ){
         MPI_Recv(resVector+i, 1, MPI_INT , MPI_ANY_SOURCE, 0 , MPI_COMM_WORLD, status);
    }
     
     
    
    //do a local sort
    qsort(resVector, totalNum, sizeof(int), compare);
    
    //every processor writes its result to a file
    {
        FILE* fd = NULL;
        char filename[256];
        snprintf(filename, 256, "output%02d.txt", rank);
        fd = fopen(filename,"w+");
        
        if(NULL == fd)
        {
            printf("Error opening file \n");
            return 1;
        }
        
        //fprintf(fd, "rank %d received from %d the message:\n", rank, origin);
        for(i = 0; i < totalNum; i++)
            fprintf(fd, "  %d\n", resVector[i]);
        
        fclose(fd);
    }

    
    free(vec);
    free(randomEntries);
    free(resVector);
    free(sendBuffer);
    free(recBuffer);
    free(count);
    free(buckets);
    free(splitters);
    free(rbuf);
    MPI_Finalize();
    return 0;
}