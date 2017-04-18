

#include <stdio.h>
#include  <stdlib.h>
#include "util.h"
#include <math.h>
#include <mpi.h>
#include <assert.h>



int lN;

int convertToOneDimension(int x, int y){
    return (lN + 2) * x + y;
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double  *lu, int lN, double invhsq){
    int p, q;
    double tmp, gres = 0.0, lres = 0.0;
    for (p = 1; p <= lN; p++){
        for ( q = 1; q <= lN; q++){
            tmp = ((2.0* lu[convertToOneDimension(p,q)]-
		    lu[convertToOneDimension(p,q-1)]-
		    lu[convertToOneDimension(p,q+1)] )*invhsq -1);
            lres += tmp * tmp;
        }
    }

    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}

int main(int argc, char * argv[]) {
    int rank, i, j, iter, p, N, max_iters;
    if (argc!= 3){
        printf ("you need to provide N and max iterations");
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status status1, status2, status3, status4;
    int sqrp = sqrt(p);
    assert(p == sqrp*sqrp);
    
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &max_iters);
 
    lN = N / sqrp;
    

    if ((N % sqrp != 0) && rank == 0 ) {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of sqrt(p)\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    
    //timing
    MPI_Barrier(MPI_COMM_WORLD);
    timestamp_type time1, time2;
    get_timestamp(&time1);
    
 
    
    double *lu = (double *) calloc(sizeof(double), (lN+2)*(lN+2));
    double *lunew = (double *) calloc(sizeof(double), (lN+2)*(lN+2));
    double *lutemp;

    double h = 1.0 / (N + 1);
    double hsq = h * h;
    double invhsq = 1./hsq;
    double gres, gres0, tol = 1e-5;

    /* initial residual */
    gres0 = compute_residual(lu, lN, invhsq);
    gres = gres0;

    double* topSend = calloc(sizeof(double), lN);
    double* bottomSend = calloc(sizeof(double), lN);
    double* leftSend = calloc(sizeof(double), lN);
    double* rightSend = calloc(sizeof(double), lN);
    
    double *topRec = calloc(sizeof(double), lN);
    double *bottomRec = calloc(sizeof(double), lN);
    double *leftRec = calloc(sizeof(double), lN);
    double *rightRec = calloc(sizeof(double), lN);
    
    for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
        /* Jacobi step for local points */
        for (i = 1; i <= lN; i++){
            for (j = 1; j <= lN; j++){
                lunew[convertToOneDimension(i,j)] = 0.25 *
		  (hsq  + lu[convertToOneDimension(i-1,j)] +
		   lu[convertToOneDimension(i,j-1)] +
		   lu[convertToOneDimension(i+1,j)] +
		   lu[convertToOneDimension(i,j+1)] );
            }
        }
        
        //printf( "send top values to its top block and receive from them\n" );
        if (rank < p - sqrp){
            for (i = 1; i <= lN; i++ ){
                topSend[i - 1] = lunew[convertToOneDimension(lN, i)];
            }
	        MPI_Send(topSend, lN, MPI_DOUBLE, rank+sqrp, 124, MPI_COMM_WORLD);
	        MPI_Recv(topRec, lN , MPI_DOUBLE, rank+sqrp, 123, MPI_COMM_WORLD, &status1);
           
        }
	
	    //printf( "i am rank %d , first step finished\n", rank);

        //send its lower values to its bottom block and receive from them
        if (rank > sqrp - 1){

            for (i = 1; i <= lN; i++ ){
                bottomSend[i - 1] = lunew[convertToOneDimension(1,i)];
            }
            MPI_Send(bottomSend, lN, MPI_DOUBLE, rank - sqrp, 123, MPI_COMM_WORLD);
            MPI_Recv(bottomRec, lN , MPI_DOUBLE, rank - sqrp, 124, MPI_COMM_WORLD, &status2);
            
        }
	
        //printf( "i am rank %d ,second step finished\n", rank);
        
        // send it left values to its left block and receive from them
        if ((rank % sqrp )!= 0){
           
            for (i = 1; i <= lN; i++ ){
                leftSend[i - 1] = lunew[convertToOneDimension(i,1)];
                
            }
            MPI_Send(leftSend, lN, MPI_DOUBLE, rank - 1, 126, MPI_COMM_WORLD);
            //printf ("i am rank %d , sending finished\n", rank);
            MPI_Recv(leftRec, lN , MPI_DOUBLE, rank - 1, 125, MPI_COMM_WORLD, &status3);
            //printf ("i am rank %d , receiving finished\n", rank);
           
        }

        //printf( "i am rank %d , third step finished\n", rank);
        //send its right values to its right block and receive from them
        if ((rank + 1 ) % sqrp != 0 ){
            
            for (i = 1; i <= lN; i++ ){
                rightSend[i - 1] = lunew[convertToOneDimension(i, lN)];
                //printf("right send is %f, i is %d\n", lunew[convertToOneDimension(i,lN)], i);
            }
            MPI_Send(rightSend, lN, MPI_DOUBLE, rank + 1, 125, MPI_COMM_WORLD);
            //printf ("i am rank %d , sneding finished\n", rank);
            MPI_Recv(rightRec, lN , MPI_DOUBLE, rank + 1 , 126, MPI_COMM_WORLD, &status4);
            //printf ("i am rank %d , receiving finished\n", rank);
            
        }

        //printf( "i am rank %d , final step finished\n", rank);
        
        //update ghost values
        for (i = 1; i <= lN; i++ ){
            lunew[convertToOneDimension(lN + 1, i)] = topRec[i - 1];
            lunew[convertToOneDimension(0,i)] = bottomRec[i - 1];
            lunew[convertToOneDimension(i,0)] = leftRec[i - 1];
            lunew[convertToOneDimension(i,lN + 1)] = rightRec[i - 1];
        }
  
        //exchange lu and lunew
        lutemp = lu;
        lu = lunew;
        lunew = lutemp;
        
        if (0 == (iter % 10)) {
            gres = compute_residual(lu, lN, invhsq);
             if (0 == rank) {
                 printf("Iter : %d, Residual: %f\n", iter, gres);
             }
        }
       
    }
   
    // timing
    MPI_Barrier(MPI_COMM_WORLD);
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1,time2);
    if (0 == rank)
    printf("Time elapsed is %f seconds.\n", elapsed);
    
    free(topSend);
    free(bottomSend);
    free(leftSend);
    free(rightSend);
    free (topRec );
    free (bottomRec );
    free (leftRec );
    free (rightRec );
    free (lu );
    free (lunew );
    MPI_Finalize();
}
