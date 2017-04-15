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


using namespace std;

int lN;


int convertToOneDimension(int x, int y){
    return (lN + 2) * x + y;
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double  *lu, int lN, double invhsq){
    int p;
    int q;
    double tmp, gres = 0.0, lres = 0.0;
    for (p = 1; p <= lN; p++){
        for ( q = 1; q <= lN; q++){
            tmp = ((2.0* lu[convertToOneDimension(p,q)]-lu[convertToOneDimension(p,q-1)]-lu[convertToOneDimension(p,q+1)]- lu[convertToOneDimension(p-1,q)]-lu[convertToOneDimension(p + 1,q)] )*invhsq -1);
            //tmp = ((2.0*u[p][q] - u[p][q-1] - u[p][q+1]) * invhsq - 1);
            lres += tmp * tmp;
            //cout<< "tmp is " <<tmp <<endl;
        }
    }
    
    ///cout  << "lres is " <<lres<<endl;
 
    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}

int main(int argc, char** argv) {
    int rank;
    int i, N;
    int p;
    int max_iters;
   
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status* status;
    
   
    
    if (rank == 0 ){
        cout << "input number of discretization points N \n";
        cin >> N;
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }else {
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    lN = N / p;
    
    if (rank == 0 ){
        cout << "input max iteration\n";
        cin >> max_iters;
        MPI_Bcast(&max_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }else {
        MPI_Bcast(&max_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    
    
    if ((N % p != 0) && rank == 0 ) {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of p\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    
    
    
    
    /* timing
    MPI_Barrier(MPI_COMM_WORLD);
    timestamp_type time1, time2;
    get_timestamp(&time1);
     */
    
    double *lu = new double[(lN+2)*(lN+2)]();
    double *lunew = new double[(lN+2)*(lN+2)]();
   
    
    double h = 1.0 / (lN + 1);
    double hsq = h * h;
    double invhsq = 1./hsq;
    double gres, gres0, tol = 1e-5;
    
    /* initial residual */
    gres0 = compute_residual(lu, lN, invhsq);
    gres = gres0;
    //u[0] = u[N+1] = 0.0;
  
    int j, iter;
    
    
    /*
    cout<< "i am rank " <<rank<<endl;
    cout<< "gres is " << gres << "gres0 is " << gres0 << "tol is " <<tol<<"max iter is"<< max_iters
    << "ln is "<<lN<<endl;
     */
    
    
     
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
        
        /* Jacobi step for local points */
        for (i = 1; i <= lN; i++){
            for (j = 1; j <= lN; j++){
                lunew[convertToOneDimension(i,j)] = 0.25 *(hsq  + lu[convertToOneDimension(i-1,j)] + lu[convertToOneDimension(i,j-1)] + lu[convertToOneDimension(i+1,j)] + lu[convertToOneDimension(i,j+1)] );
                //unew[i][j] = 0.25 *(hsq * hsq + u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1] );
                //std::cout << "unew[ "<< i<< "][" <<j <<"] is " << u[convertToOneDimension(i,j)] <<endl;
            }
            
        }
        
        
        
        
        //cout<<" communicate ghost values"<<endl;
        
        double* topRec;
        double* bottomRec;
        double* leftRec;
        double* rightRec;
        
        topRec = new double[lN]();
        bottomRec = new double[lN]();
        leftRec = new double[lN]();
        rightRec = new double[lN]();
        
        
         //MPI_Barrier(MPI_COMM_WORLD);
        //cout <<"send top values to its top block and receive from them"<<endl;
        
        if (rank < p - sqrt(p)){
            double* topSend;
            topSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                topSend[i - 1] = lunew[convertToOneDimension(lN, i)];
            }
            
            MPI_Send(topSend, lN, MPI_DOUBLE, rank+sqrt(p), 124, MPI_COMM_WORLD);
            MPI_Recv(topRec, lN , MPI_DOUBLE, rank+sqrt(p), 123, MPI_COMM_WORLD, status);
            delete [] topSend;
        }
        
        //cout<< "i am rank " <<rank<< " first step finished"<<endl;
        
        //send its lower values to its bottom block and receive from them
        if (rank > sqrt(p) - 1){
            double* bottomSend;
            bottomSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                bottomSend[i - 1] = lunew[convertToOneDimension(1,i)];
            }
            
            
            
            MPI_Send(bottomSend, lN, MPI_DOUBLE, rank - sqrt(p), 123, MPI_COMM_WORLD);
            MPI_Recv(bottomRec, lN , MPI_DOUBLE, rank - sqrt(p), 124, MPI_COMM_WORLD, status);
            delete[] bottomSend;
          
        }
        
         //cout<< "i am rank " <<rank<< " second step finished"<<endl;
        
        
        // send it left values to its left block and receive from them
        if ((rank % (int)sqrt(p) )!= 0){
            double* leftSend;
            leftSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                leftSend[i - 1] = lunew[convertToOneDimension(i,1)];
            }

            MPI_Send(leftSend, lN, MPI_DOUBLE, rank - 1, 126, MPI_COMM_WORLD);
            MPI_Recv(leftRec, lN , MPI_DOUBLE, rank - 1, 125, MPI_COMM_WORLD, status);
            
            delete[] leftSend;
        }
        
         //cout<< "i am rank " <<rank<< " third step finished"<<endl;
       
        
        //send its right values to its right block and receive from them
        if ((rank + 1 ) % (int)sqrt(p) != 0 ){
            double* rightSend;
            rightSend = new double[lN]();
            for (i = 1; i <= lN; i++ ){
                rightSend[i - 1] = lunew[convertToOneDimension(i, lN)];
            }
            
            MPI_Send(rightSend, lN, MPI_DOUBLE, rank + 1, 125, MPI_COMM_WORLD);
            MPI_Recv(rightRec, lN , MPI_DOUBLE, rank + 1 , 126, MPI_COMM_WORLD, status);
            
            delete []  rightSend;
        }
        
          //cout<< "i am rank " <<rank<< " third step finished"<<endl;
        
         MPI_Barrier(MPI_COMM_WORLD);
        
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
        
        
      
       

        /* copy new_u onto u */
        for ( i= 1 ; i <= lN; i++){
            for ( j = 1; j <= lN; j++){
                //cout << "lunew is " << lunew[convertToOneDimension(i,j)]<<endl;
                lu[convertToOneDimension(i,j)] = lunew[convertToOneDimension(i,j)];
            }
        }
        
        //cout<< "I am rank " <<rank <<endl;
        
        
        if (0 == (iter % 10)) {
            gres = compute_residual(lu, lN, invhsq);
             if (0 == rank) {
            cout << "Iter : "<<iter<< ", Residual: "<< gres<<endl;
             }
        }
    }
    
    
   
    
    /* timing
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1,time2);
    printf("Time elapsed is %f seconds.\n", elapsed);
    */
    delete []lu;
    delete []lunew;
    MPI_Finalize();
    return 0;
    
    
    
}
