#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <stdio.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
	using std::vector;
	MPI_Init(&argc, &argv);
	double t1, t2;
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = MPI_Wtime();

	int n = atoi(argv[1]); // input size of vector
	int c = atoi(argv[2]); // input seed

	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int myC = c + rank;
	srand48(myC);
	int numberEveryProcessor = n / size;
	vector <double> x(numberEveryProcessor); // part of vector x in this processor
	int i;
	for (i = 0; i < numberEveryProcessor; i++) {
		x[i] = drand48();
	}

	double localSum = 0.0;
	for (i = 0; i < numberEveryProcessor; i++) {
		localSum += x[i] * x[i];
	}
	double receivedSum;
	double sentSum = localSum; // for the 0th rank, it is the final sum

	int d = 0; int npRoot2 = size;
	while (npRoot2 > 1) { // find log_2(size)
		npRoot2 = npRoot2 >> 1;
		d++;
	}

	int j; int jpower2 = 1;
	for (j = 0; j < d; j++) {
		if ((rank & jpower2) != 0) {
			int receiver = rank ^ jpower2;
			// printf("I am processor %d, in %d iter, my receiver is %d\n", rank, j, receiver);
			MPI_Send(&sentSum, 1, MPI_DOUBLE, receiver, 11, MPI_COMM_WORLD);
			break;
		}
		else {
			int sender = rank ^ jpower2;
			// printf("I am processor %d, in %d iter, my sender is %d\n", rank, j, sender);
			MPI_Status stat;
			MPI_Recv(&receivedSum, 1, MPI_DOUBLE, sender, 11, MPI_COMM_WORLD, &stat);
			sentSum += receivedSum;
		}
		jpower2 = jpower2 << 1;
	}

	double result;
	result = sqrt(sentSum);
		
	MPI_Barrier(MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	if (rank == 0) printf("%d %d %d %9.3f %9.6f\n", n, size, c, result, (t2 - t1));

	return MPI_Finalize();
}
