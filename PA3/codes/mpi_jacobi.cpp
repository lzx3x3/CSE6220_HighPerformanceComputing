/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <string.h>

/*
 * TODO: Implement your solutions here
 */
#define NDIM 2


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    // firstly, find my coord in the cart comm
    int crank, size; // sequential rank
    int coords[NDIM]; // 2-d coord
    MPI_Comm_rank(comm, &crank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, crank, NDIM, coords);
    int i, j, k;
    int sqrtSize = (int)sqrt(size);
    int *rowColCounts = (int*)malloc(sizeof(int)*sqrtSize);
    for (i = 0; i < sqrtSize; i++) { // initialize
        rowColCounts[i] = 0;
    }
    int rem = n % sqrtSize;
    for (i = 0; i < sqrtSize; i++) { // how to divide n entries into sqrt(size) processors? It can be either row or col
        rowColCounts[i] = n / sqrtSize;
        if (rem-- > 0)
            rowColCounts[i]++;
    }

    // secondly, for (0,0) thread, scatter the input vector to first column (x, 0)
    MPI_Comm colComm; // processors in the same column in this comm
    MPI_Comm_split(comm, coords[1], coords[0], &colComm);
    int localSizeRow = rowColCounts[coords[0]];
    double *receive0col;
    int *sendcounts = (int*)malloc(sizeof(int)*sqrtSize);
    int *displs = (int*)malloc(sizeof(int)*sqrtSize);
    for (i = 0; i < sqrtSize; i++) { // initialize
        sendcounts[i] = 0;
        displs[i] = 0;
    }
    int sum = 0;
    if (coords[1] == 0) { // reminder: (dim0, dim1). dim0 is row number, dim1 is the column number
        //// verify the correctness of the code
        // if (coords[0] == 0) {
        //     printf("(0,0) processor print the transpose of input vector\n");
        //     for (int y = 0; y < n; y++) {
        //         printf("%9.6f ", input_vector[y]);
        //     }
        //     printf("\n");
        // }
        receive0col = (double*)malloc(sizeof(double)*localSizeRow); // save the ith row matrix
        *local_vector = receive0col;
        for (i = 0; i < sqrtSize; i++) { // i: row of threads
            sendcounts[i] = rowColCounts[i]; // how many entries are sent?
            displs[i] = sum;
            sum += sendcounts[i];
        }
        MPI_Scatterv(input_vector, sendcounts, displs, MPI_DOUBLE, receive0col, localSizeRow, MPI_DOUBLE, 0, colComm);
        //// verify the correctness of the code
        // printf("(%d, %d) processor print the transpose of input local vector\n", coords[0], coords[1]);
        // for (int y = 0; y < localSizeRow; y++) {
        //     printf("%9.6f ", receive0col[y]);
        // }
        // printf("\n");
    }

    // thirdly, remove the created comm and space
    MPI_Comm_free(&colComm);
    free(rowColCounts);
    free(sendcounts);
    free(displs);
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    // firstly, find my coord in the cart comm
    int crank, size; // sequential rank
    int coords[NDIM]; // 2-d coord
    MPI_Comm_rank(comm, &crank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, crank, NDIM, coords);
    int i, j, k;
    int sqrtSize = (int)sqrt(size);
    int *rowColCounts = (int*)malloc(sizeof(int)*sqrtSize);
    for (i = 0; i < sqrtSize; i++) { // initialize
        rowColCounts[i] = 0;
    }
    int rem = n % sqrtSize;
    for (i = 0; i < sqrtSize; i++) { // how to divide n entries into sqrt(size) processors? It can be either row or col
        rowColCounts[i] = n / sqrtSize;
        if (rem-- > 0)
            rowColCounts[i]++;
    }

    // secondly, for (i,0) thread, gather the output vector to first row (0, 0)
    MPI_Comm colComm; // processors in the same column in this comm
    MPI_Comm_split(comm, coords[1], coords[0], &colComm);
    int localSizeRow = rowColCounts[coords[0]];
    int *receivecounts = (int*)malloc(sizeof(int)*sqrtSize);
    int *displs = (int*)malloc(sizeof(int)*sqrtSize);
    int sum = 0;
    if (coords[1] == 0) {
        //// verify the correctness of the code
        // printf("(%d, %d) processor print the transpose of local vector to be gathered\n", coords[0], coords[1]);
        // for (int x = 0; x < localSizeRow; x++) {
        //     printf("%9.6f ", local_vector[x]);
        // }
        // printf("\n");
        for (j = 0; j < sqrtSize; j++) { // j: col of threads
            receivecounts[j] = rowColCounts[j]; // how many entries are sent?
            displs[j] = sum;
            sum += receivecounts[j];
        }
        // MPI_Gather(local_vector, localSizeRow, MPI_DOUBLE, output_vector, localSizeRow, MPI_DOUBLE, 0, colComm); // mistake. use MPI_Gatherv
        MPI_Gatherv(local_vector, rowColCounts[coords[0]], MPI_DOUBLE, output_vector, receivecounts, displs, MPI_DOUBLE, 0, colComm);
        //// verify the correctness of the code
        // if (coords[0] == 0) {
        //     printf("(0, 0) processor print the transpose of gathered vector\n");
        //     for (int x = 0; x < n; x++) {
        //         printf("%9.6f ", output_vector[x]);
        //     }
        //     printf("\n");
        // }
    }

    free(rowColCounts);
    free(receivecounts);
    free(displs);
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // firstly, find my coord in the cart comm
    int crank, size; // sequential rank
    int coords[NDIM]; // 2-d coord
    MPI_Comm_rank(comm, &crank);
    MPI_Comm_size(comm, &size);
    MPI_Cart_coords(comm, crank, NDIM, coords);
    int i, j, k;
    int sqrtSize = (int)sqrt(size);
    int *rowColCounts = (int*)malloc(sizeof(int)*sqrtSize);
    for (i = 0; i < sqrtSize; i++) { // initialize
        rowColCounts[i] = 0;
    }
    int rem = n % sqrtSize;
    for (i = 0; i < sqrtSize; i++) {
    	rowColCounts[i] = n / sqrtSize;
        if (rem-- > 0)
            rowColCounts[i]++;
    }

    // secondly, for (0,0) thread, scatter the input matrix to first column (x, 0)
    MPI_Comm colComm; // processors in the same column in this comm
    MPI_Comm_split(comm, coords[1], coords[0], &colComm);
    int localSizeRow = rowColCounts[coords[0]];
    double *receive0col;
    int *sendcounts = (int*)malloc(sizeof(int)*sqrtSize);
    int *displs = (int*)malloc(sizeof(int)*sqrtSize);
    for (i = 0; i < sqrtSize; i++) { // initialize
        sendcounts[i] = 0;
        displs[i] = 0;
    }
    int sum = 0;
    if (coords[1] == 0) { // reminder: (dim0, dim1). dim0 is row number, dim1 is the column number
        receive0col = (double*)malloc(sizeof(double)*(localSizeRow*n)); // save the ith row matrix
        for (i = 0; i < sqrtSize; i++) { // i: row of threads
            sendcounts[i] = rowColCounts[i]*n; // how many entries are sent?
            displs[i] = sum;
            sum += sendcounts[i];
        }
        MPI_Scatterv(input_matrix, sendcounts, displs, MPI_DOUBLE, receive0col, localSizeRow*n, MPI_DOUBLE, 0, colComm);
        for (i = 0; i < sqrtSize; i++) { // restore
            sendcounts[i] = 0;
            displs[i] = 0;
        }


        //// verify the correctness of the code
        // if ((coords[0] == 0) && (coords[1] == 0)) {
        //     printf("(0,0) processor print the input matrix\n");
        //     for (int x = 0; x < n; x++) {
        //         for (int y = 0; y < n; y++) {
        //             printf("%9.6f ", input_matrix[x*n+y]);
        //         }
        //         printf(" end of input matrix row %d\n", x);
        //     }
        // }
    }

    // thirdly, for (i,0) thread, scatter the received rows to other threads in the same row
    MPI_Comm rowComm; // processors in the same row in this comm
    MPI_Comm_split(comm, coords[0], coords[1], &rowComm);
    int localSizeCol = rowColCounts[coords[1]];
    double *local_matrix_space = (double*)malloc(sizeof(double)*localSizeRow*localSizeCol);
    *local_matrix = local_matrix_space;
    sum = 0;
    for (j = 0; j < sqrtSize; j++) { // j: col of threads
        sendcounts[j] = rowColCounts[j]; // how many entries are sent?
        displs[j] = sum;
        sum += sendcounts[j];
    }
    double *send0col;
    if (coords[1] == 0) {
        send0col = (double*)malloc(sizeof(double)*n);
    }
    for (k = 0; k < localSizeRow; k++) { // k: row of matrix. The (i, 0) thread scatter the matrix row by row
    	if (coords[1] == 0) {
        	memcpy(send0col, receive0col + n*k, sizeof(double)*n);// composing send0col array for scatterv
    	}
    	MPI_Scatterv(send0col, sendcounts, displs, MPI_DOUBLE, local_matrix_space + k*localSizeCol, localSizeCol, MPI_DOUBLE, 0, rowComm);
    }

    //// verify the correctness of the code
    // printf("I am thread (%d, %d). My local matrix is\n", coords[0], coords[1]);
    // for (int x = 0; x < localSizeRow; x++) {
    //     for (int y = 0; y < localSizeCol; y++) {
    //         printf("%9.6f ", local_matrix_space[x*localSizeCol+y]);
    //     }
    //     printf(" end of local matrix row %d\n", x);
    // }

    // at last, free distributed space
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);
    free(rowColCounts);
    free(sendcounts);
    free(displs);
    if (coords[1] == 0) {// after local matrices are distributed
        free(receive0col);
        free(send0col);
    }
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    int rank, coords[2];
    int bRow, bCol;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];
    //
    MPI_Status stat;
    // printf("in transpose_bcast_vector\n");
    // printf("row %d, col %d, rank %d\n", row, col, rank);
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(comm, row, col, &rowComm);
    MPI_Comm_split(comm, col, row, &colComm);
    bRow = block_decompose(n,rowComm);
    bCol = block_decompose(n,colComm);
    // send data from the first column to the diagonal
    if (col == 0 && row != 0) {
        int receive, receiveCoords[2];
        receiveCoords[0] = row;
        receiveCoords[1] = row;
        MPI_Cart_rank(comm, receiveCoords, &receive);
        MPI_Send(&col_vector[0], bCol, MPI_DOUBLE, receive, 1, comm);
    }
    // diagonal receive
    if (row == col && row != 0) {
        int send, sendCoords[2];
        sendCoords[0] = row;
        sendCoords[1] = 0;
        MPI_Cart_rank(comm, sendCoords, &send);
        MPI_Recv(&row_vector[0], bCol, MPI_DOUBLE, send, 1, comm, &stat);
    } else if (row == col && row == 0) {
        for (int i = 0; i < bCol; i++) {
            row_vector[i] = *(col_vector + i);
        }
    }
    // broadcast from diagonal
    MPI_Bcast(&row_vector[0], bRow, MPI_DOUBLE, col, colComm);
    //// verify the correctness of the code
    // if (row == 0) {
    //     printf("row 0, processor %d, rank %d print the data\n", col, rank);
    //     for (int i = 0; i < bCol; i++) {
    //         printf("%9.6f ", row_vector[i]);
    //     }
    //     printf("\n");
    // }
    // printf("(%d, %d) processor print the transpose of input local vector\n", coords[0], coords[1]);
    // for (int y = 0; y < bRow; y++) {
    //     printf("%9.6f ", row_vector[y]);
    // }
    // printf("\n");

    // free
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    int rank, coords[2];
    int bRow, bCol;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];
    //
    // printf("in distributed_matrix_vector_mult\n");
    // printf("row %d, col %d, rank %d\n", row, col, rank);
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(comm, row, col, &rowComm);
    MPI_Comm_split(comm, col, row, &colComm);
    bRow = block_decompose(n,rowComm);
    bCol = block_decompose(n,colComm);
    // distribute local_x
    double *distrX = new double[bRow];
    transpose_bcast_vector(n, local_x, distrX, comm);
    // compute locally
    double *subY = new double[bCol];
    for (int i = 0; i < bCol; i++) {
        subY[i] = 0.0;
        // for (int j = 0; j < bCol; j++) { // mistake
        for (int j = 0; j < bRow; j++) {
            subY[i] += (*(local_A + i * bRow + j)) * (*(distrX + j));
        }
    }
    // reduction

    // printf("processor (%d, %d) print the computed local y data\n", row, col);
    // for (int i = 0; i < bCol; i++) {
    //     printf("%9.3f ", subY[i]);
    // }
    // printf("\n");

    MPI_Reduce(&subY[0], &local_y[0], bCol, MPI_DOUBLE, MPI_SUM, 0, rowComm);
    //// verify the correctness of the code
    // if (col == 0) {
    //     printf("processor (%d, %d) print the reduced Ax data\n", row, col, rank);
    //     for (int i = 0; i < bCol; i++) {
    //         printf("%9.3f ", local_y[i]);
    //     }
    //     printf("\n");
    // }
    // release
    delete[] distrX;
    delete[] subY;
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);    
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    int rank, coords[2];
    int bRow, bCol;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];
    //
    MPI_Status stat; 
    // printf("in distributed_jacobi ");
    // printf("row %d, col %d, rank %d\n", row, col, rank);
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(comm, row, col, &rowComm);
    MPI_Comm_split(comm, col, row, &colComm);
    bRow = block_decompose(n,rowComm);
    bCol = block_decompose(n,colComm);
    // D
    double *D = new double[bCol];
    if (row == col) {
        for (int i = 0; i < bCol; i++) {
            D[i] = local_A[i * bCol + i];
        }
        int receive, receiveCoords[2];
        receiveCoords[0] = row;
        receiveCoords[1] = 0;
        //
        // printf("D, row %d, col %d, rank %d\n", row, col, rank);
        // for (int i = 0; i < bCol; i++) {
        //     printf("%9.3f", D[i]);
        // }
        // printf("\n#############\n");
        MPI_Cart_rank(comm, receiveCoords, &receive);
        // do not sent to 0 from 0
        if (receive != 0) {
            // printf("send D from row %d, col %d to %d\n", row, col, receive);
            MPI_Send(&D[0], bCol, MPI_DOUBLE, receive, 1, comm);
        }
    }
    // D inverse
    double *invD = new double[bCol];
    if (col == 0) {
        int send, sendCoords[2];
        sendCoords[0] = row;
        sendCoords[1] = row;
        MPI_Cart_rank(comm, sendCoords, &send);
        //
        if (row != 0) {
            // printf("receive D from %d by row %d, col %d\n", send, row, col);
            MPI_Recv(&D[0], bCol, MPI_DOUBLE, send, 1, comm, &stat);
        }
        for (int i = 0; i < bCol; i++) {
            invD[i] = 1 / D[i];
        }
        //
        // printf("invD, row %d, col %d, rank %d\n", row, col, rank);
        // for (int i = 0; i < bCol; i++) {
        //     printf("%9.3f",invD[i]);
        // }
        // printf("\n#############\n");
    }
    // R = A - D
    double *R = new double[bCol*bRow];
    if (row == col) {
        for (int i = 0; i < bCol; i++) {
            for (int j = 0; j < bCol; j++) {
                if (i == j) {
                    R[i * bCol + j] = 0;
                } else {
                    R[i * bCol + j] = local_A[i * bCol + j];
                }
            }
        }
    } else {
        for (int i = 0; i < bRow * bCol; i++) {
            R[i] = *(local_A + i);
        }
    }
    //
    // printf("R, row %d, col %d, rank %d\n", row, col, rank);
    // for (int i = 0; i < bCol * bRow; i++) {
    //     printf("%9.3f", R[i]);
    // }
    // printf("\n#############\n");
    // initialize local x
    for (int i = 0; i < bCol; i++) {
        local_x[i] = 0;
    }
    double l2;
    double *Ax = new double[bCol];
    double localL2; 
    int coordsZero[2] = {0, 0};
    int rankZero;
    MPI_Cart_rank(comm, coordsZero, &rankZero);
    double *bSubRx = new double[bCol];
    // jacobi
    for (int iter = 0; iter < max_iter; iter++) {
        // A * x -> Ax
        distributed_matrix_vector_mult(n, &local_A[0], &local_x[0], &Ax[0], comm);
        // A * x - b
        localL2 = 0.0;
        if (col == 0) {
            for (int i = 0; i < bCol; i++) {
                double temp = Ax[i] - local_b[i];
                localL2 += temp * temp;
            }
            MPI_Reduce(&localL2, &l2, 1, MPI_DOUBLE, MPI_SUM, 0, colComm);
        }
        if (row == 0 && col == 0) {
            l2 = sqrt(l2);
            // printf("l2: %9.4f\n", l2);
        }
        MPI_Bcast(&l2, 1, MPI_DOUBLE, rankZero, comm);
        if (l2 <= l2_termination) {
            delete[] D;
            delete[] invD;
            delete[] R;
            delete[] Ax;
            delete[] bSubRx;
            MPI_Comm_free(&colComm);
            MPI_Comm_free(&rowComm);
            return;
        }
        // R * x -> Ax
        distributed_matrix_vector_mult(n, &R[0], &local_x[0], &Ax[0], comm);
        // b - R * x -> bSubRx
        if (col == 0) {
            for (int i = 0; i < bCol; i++) {
                bSubRx[i] = local_b[i] - Ax[i];
            }
            for (int i = 0; i < bCol; i++) {
                local_x[i] = invD[i] * bSubRx[i];
            }
        }
    }
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination) // comm at here is a grid_comm
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
