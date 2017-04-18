linker: mpi_solved1.c  mpi_solved2.c mpi_solved3.c mpi_solved4.c mpi_solved6.c mpi_solved7.c jacobi-mpi2D.cpp ssort.c
	mpicc  mpi_solved1.c -o mpi_solved1
	mpicc  mpi_solved2.c -o mpi_solved2
	mpicc  mpi_solved3.c -o mpi_solved3
	mpicc  mpi_solved4.c -o mpi_solved4
	mpicc  mpi_solved6.c -o mpi_solved6
	mpicc  mpi_solved7.c -o mpi_solved7
	mpicc  ssort.c  -o ssort
	mpicc jacobi-mpi2D.c -o jacobi-mpi2D -lm


clean:
	rm -f  mpi_solved*~
	rm -f  jacobi-mpi2D *~
	rm -f  ssort *~