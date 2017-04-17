//
//   j.cpp
//
//
//  Created by Mengran Wang on 3/30/17.
//
//

#include <stdio.h>
#include <iostream>
//#include "util.h"
#include <cmath>
#include <mpi.h>
#include <assert.h>


using namespace std;

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

int main(int argc, char** argv) {
    int rank, i, p, N, max_iters;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status* status1, *status2, * status3,* status4;
    int sqrp = sqrt(p);
    assert(p == sqrp*sqrp);
    if (rank == 0 ){
        cout << "input number of discretization points N \n";
        cin >> N;
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    lN = N / sqrp;

    if (rank == 0 ){
        cout << "input max iteration\n";
        cin >> max_iters;
    }
    MPI_Bcast(&max_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if ((N % sqrp != 0) && rank == 0 ) {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of sqrt(p)\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    
    /* timing
    MPI_Barrier(MPI_COMM_WORLD);
    timestamp_type time1, time2;
    get_timestamp(&time1);
     */
    
    double *lu = new double[(lN+2)*(lN+2)]();
    double *lunew = new double[(lN+2)*(lN+2)]();
    double *lutemp;

    double h = 1.0 / (N + 1);
    double hsq = h * h;
    double invhsq = 1./hsq;
    double gres, gres0, tol = 1e-5;

    /* initial residual */
    gres0 = compute_residual(lu, lN, invhsq);
    gres = gres0;
    int j, iter;

    if (rank == 0) {
      //cout << "i am rank " << rank << endl;
      //      cout << "gres is " << gres << "gres0 is " << gres0 <<
      //	"tol is " << tol << "max iter is" << max_iters
      //   << "ln is "<< lN << endl;
    }
    for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
        /*
        if (rank == 1){
        cout << "lu :" <<endl;
        for (i = 0; i <= lN + 1; i++){
            for (j = 0; j <= lN + 1; j++){
                cout << lu[convertToOneDimension(i,j)] << " ";
	       }
            cout << endl;
	    }
        }
              cout << endl;
         */
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
	if (rank == 0)
	  cout << " communicate ghost values" << endl;

        double *topRec = new double[lN]();
        double *bottomRec = new double[lN]();
        double *leftRec = new double[lN]();
        double *rightRec = new double[lN]();

    	if (rank == 0)
	  cout << "send top values to its top block and receive from them" << endl;

        if (rank < p - sqrp){
            double *topSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                topSend[i - 1] = lunew[convertToOneDimension(lN, i)];
            }
	    MPI_Send(topSend, lN, MPI_DOUBLE, rank+sqrp, 124, MPI_COMM_WORLD);
	    if (rank == 0)
	      cout << "Done 1" << endl;
	    MPI_Recv(topRec, lN , MPI_DOUBLE, rank+sqrp, 123, MPI_COMM_WORLD, status1);


            delete [] topSend;
        }

	if (rank == 0)
	  cout << "i am rank " << rank << " first step finished"<<endl;

        //send its lower values to its bottom block and receive from them
        if (rank > sqrp - 1){
            double* bottomSend;
            bottomSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                bottomSend[i - 1] = lunew[convertToOneDimension(1,i)];
            }
	    if (rank == 2)
	      cout << "other side of communication" << endl;
            MPI_Recv(bottomRec, lN , MPI_DOUBLE, rank - sqrp, 124, MPI_COMM_WORLD, status2);
	    if (rank == 2)
	      cout << "Done 2" << endl;
	    MPI_Send(bottomSend, lN, MPI_DOUBLE, rank - sqrp, 123, MPI_COMM_WORLD);


            delete[] bottomSend;
        }
	if (rank == 0)
	  cout<< "i am rank " <<rank<< " second step finished"<<endl;

        // send it left values to its left block and receive from them
        if ((rank % sqrp )!= 0){
            double* leftSend;
            leftSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                leftSend[i - 1] = lunew[convertToOneDimension(i,1)];
            }

            MPI_Send(leftSend, lN, MPI_DOUBLE, rank - 1, 126, MPI_COMM_WORLD);
            MPI_Recv(leftRec, lN , MPI_DOUBLE, rank - 1, 125, MPI_COMM_WORLD, status3);

            cout << "leftrec is " <<endl;
            for (i = 1; i <= lN; i++ ){
                cout << leftRec[i - 1] << " ";
            }
            cout << " "<< endl;
            delete[] leftSend;
        }

	if (rank == 0)
	  cout<< "i am rank " <<rank<< " third step finished"<<endl;

        //send its right values to its right block and receive from them
        if ((rank + 1 ) % sqrp != 0 ){
            double* rightSend;
            rightSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                rightSend[i - 1] = lunew[convertToOneDimension(i, lN)];
            }
            MPI_Send(rightSend, lN, MPI_DOUBLE, rank + 1, 125, MPI_COMM_WORLD);
            MPI_Recv(rightRec, lN , MPI_DOUBLE, rank + 1 , 126, MPI_COMM_WORLD, status4);
            cout << "rightrec is " <<endl;
            for (i = 1; i <= lN; i++ ){
                cout << rightRec[i - 1] << " ";
            }
            cout << " "<< endl;
            delete []  rightSend;
        }

          cout<< "i am rank " <<rank<< " third step finished"<<endl;

        for (i = 1; i <= lN; i++ ){
            lunew[convertToOneDimension(lN + 1, i)] = topRec[i - 1];
            lunew[convertToOneDimension(0,i)] = bottomRec[i - 1];
            lunew[convertToOneDimension(i,0)] = leftRec[i - 1];
            lunew[convertToOneDimension(i,lN + 1)] = rightRec[i - 1];
        }

        delete [] topRec;
        delete[]  bottomRec;
        delete[]  leftRec;
        delete[] rightRec;

        lutemp = lu;
        lu = lunew;
        lunew = lutemp;
        
        if (0 == (iter % 10)) {
            gres = compute_residual(lu, lN, invhsq);
             if (0 == rank) {
            cout << "Iter : "<<iter<< ", Residual: "<< gres<<endl;
             }
        }
        //cout << "At the end"<<endl;
    }
    if (rank == 0)
      cout<< "Done. Cleaning up."<<endl;
    
    /* timing
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1,time2);
    printf("Time elapsed is %f seconds.\n", elapsed);
    */
    delete []lu;
    delete []lunew;
    MPI_Finalize();
}
