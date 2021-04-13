/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    // TODO
    for(int i=0; i<n; i++) {
        y[i] = 0.0;
        for(int j=0; j<n; j++) {
            y[i] += A[i*n + j] * x[j];
        } 
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    // TODO
    for(int i=0; i<n; i++) {
        y[i] = 0.0;
        for(int j=0; j<m; j++) {
            y[i] += A[i*n + j] * x[j];
        }
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    // TODO

    // arrays init
    double* D = new double[n];  // diagonal matrix D
    double* R = new double[n*n];    // remaining elements matrix R
    double* y = new double[n];  // store Ax

    double l2 = 0.0;

    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            R[i*n + j] = A[i*n + j];
        }
        D[i] = A[n * i + i];
        R[n * i + i] = 0.0;
    }

    // repeating steps
    for (int iter = 0; iter < max_iter; iter++) {
        l2 = calculate_l2(n, &A[0], &b[0], &x[0], &y[0]);
        if (l2 < l2_termination) {
            break;
        }
        matrix_vector_mult(n, &R[0], &x[0], &y[0]);        // calculate y = Rx
        for(int i=0; i<n; i++) {
            x[i] = (b[i] - y[i]) / D[i];
        }
    }
}

double calculate_l2(const int n, double* A, double* b, double* x, double* y) {
    double l2 = 0.0;
    matrix_vector_mult(n, &A[0], &x[0], &y[0]);        // calculate y = Ax
    for(int i=0; i<n; i++) {               // calculate ||Ax-b||
        l2 += pow(y[i] - b[i], 2.0);
    }  
    l2 = sqrt(l2);
    return l2;                                    
}
