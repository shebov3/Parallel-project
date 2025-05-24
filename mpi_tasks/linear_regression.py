#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import time
import csv

def compute_gradient(X, y, w, b):
    """
    Compute the gradient of the linear regression cost function.
    X: Features matrix
    y: Target vector
    w: Weight vector
    b: Bias term
    """
    m = X.shape[0]
    f_wb = X @ w + b  # predictions
    error = f_wb - y  # prediction error
    
    # Compute gradients
    dj_dw = (1/m) * (X.T @ error)
    dj_db = (1/m) * np.sum(error)
    
    return dj_dw, dj_db

def compute_cost(X, y, w, b):
    """
    Compute the cost function of linear regression.
    """
    m = X.shape[0]
    f_wb = X @ w + b  # predictions
    cost = (1/(2*m)) * np.sum((f_wb - y) ** 2)  # mean squared error
    return cost

def fit_gradient_descent(X, y, w_init, b_init, alpha, num_iters, comm):
    """
    Fit a linear regression model using gradient descent.
    Parallelized using MPI to distribute computation of gradients.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Initialize parameters
    w = w_init
    b = b_init
    
    # Distribute data among processes
    m = X.shape[0]
    local_m = m // size
    start_idx = rank * local_m
    end_idx = (rank + 1) * local_m if rank < size - 1 else m
    
    local_X = X[start_idx:end_idx]
    local_y = y[start_idx:end_idx]
    
    # Perform gradient descent iterations
    for i in range(num_iters):
        # Compute local gradients
        local_dj_dw, local_dj_db = compute_gradient(local_X, local_y, w, b)
        
        # Reduce (sum) gradients from all processes
        dj_dw = np.zeros_like(local_dj_dw)
        dj_db = np.zeros_like(local_dj_db)
        
        comm.Allreduce(local_dj_dw, dj_dw, op=MPI.SUM)
        comm.Allreduce(local_dj_db, dj_db, op=MPI.SUM)
        
        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Process 0 reads the input file
    if rank == 0:
        if len(sys.argv) != 2:
            print("Usage: mpiexec -n <num_processes> python linear_regression.py <input_file>")
            comm.Abort(1)
        
        input_file = sys.argv[1]
        try:
            # Read data from CSV file
            data = []
            with open(input_file, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    data.append([float(x) for x in row])
            
            # Convert to numpy array
            data = np.array(data)
            
            # Separate features (X) and target (y)
            X = data[:, :-1]  # all columns except the last
            y = data[:, -1]   # last column
            
            # Record the start time
            start_time = time.time()
            
            # Initialize parameters
            w_init = np.zeros(X.shape[1])
            b_init = 0.0
            
            # Hyperparameters
            alpha = 0.01     # learning rate
            num_iters = 1000  # number of iterations
        except Exception as e:
            print(f"Error processing input file: {e}")
            comm.Abort(1)
    else:
        X = None
        y = None
        w_init = None
        b_init = None
        alpha = None
        num_iters = None
        start_time = None
    
    # Broadcast data and parameters to all processes
    X = comm.bcast(X, root=0)
    y = comm.bcast(y, root=0)
    w_init = comm.bcast(w_init, root=0)
    b_init = comm.bcast(b_init, root=0)
    alpha = comm.bcast(alpha, root=0)
    num_iters = comm.bcast(num_iters, root=0)
    
    # Train the model with gradient descent
    w_final, b_final = fit_gradient_descent(X, y, w_init, b_init, alpha, num_iters, comm)
    
    # Process 0 outputs the results
    if rank == 0:
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Output results
        print(','.join(map(str, w_final)))
        print(f"{b_final}")
        print(f"{elapsed_time}")

if __name__ == "__main__":
    main() 