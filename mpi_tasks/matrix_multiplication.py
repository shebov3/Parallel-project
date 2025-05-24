#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import time
import os

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Process 0 reads the input files
    if rank == 0:
        if len(sys.argv) != 3:
            print("Usage: mpiexec -n <num_processes> python matrix_multiplication.py <matrix_a_file> <matrix_b_file>")
            comm.Abort(1)
        
        matrix_a_file = sys.argv[1]
        matrix_b_file = sys.argv[2]
        
        try:
            # Load matrices from CSV files
            matrix_a = np.loadtxt(matrix_a_file, delimiter=',')
            matrix_b = np.loadtxt(matrix_b_file, delimiter=',')
            
            # Check if matrices can be multiplied
            if matrix_a.shape[1] != matrix_b.shape[0]:
                print(f"Error: Matrix dimensions don't match for multiplication. A: {matrix_a.shape}, B: {matrix_b.shape}")
                comm.Abort(1)
            
            # Record the start time
            start_time = time.time()
            
            # Matrix dimensions
            m = matrix_a.shape[0]  # rows of A
            k = matrix_a.shape[1]  # cols of A = rows of B
            n = matrix_b.shape[1]  # cols of B
            
            # Split matrix A into chunks for each process (row-wise distribution)
            rows_per_process = m // size
            remainder = m % size
            
            # Calculate send counts and displacements for Scatterv
            send_counts = np.array([rows_per_process + (1 if i < remainder else 0) for i in range(size)]) * k
            displacements = np.zeros(size, dtype=int)
            for i in range(1, size):
                displacements[i] = displacements[i-1] + send_counts[i-1]
            
            # Reshape matrix_a to 1D for Scatterv
            matrix_a_1d = matrix_a.flatten()
            
            # Create result matrix
            result_matrix = np.zeros((m, n))
        except Exception as e:
            print(f"Error processing input files: {e}")
            comm.Abort(1)
    else:
        matrix_a = None
        matrix_b = None
        m = None
        k = None
        n = None
        send_counts = None
        displacements = None
        matrix_a_1d = None
        result_matrix = None
        start_time = None
    
    # Broadcast matrix dimensions
    m, k, n = comm.bcast((m, k, n) if rank == 0 else (None, None, None), root=0)
    
    # Calculate local matrix dimensions
    rows_per_process = m // size
    local_rows = rows_per_process + (1 if rank < m % size else 0)
    
    # Create local buffer for rows of A
    local_a = np.empty(local_rows * k, dtype=float)
    
    # Scatter rows of A to all processes
    comm.Scatterv([matrix_a_1d, send_counts, displacements, MPI.DOUBLE] if rank == 0 else None, 
                  local_a, root=0)
    
    # Reshape local_a back to 2D
    local_a = local_a.reshape(local_rows, k)
    
    # Broadcast matrix B to all processes
    matrix_b = comm.bcast(matrix_b, root=0)
    
    # Perform local matrix multiplication
    local_result = np.dot(local_a, matrix_b)
    
    # Gather local results to form the complete result matrix
    # First, calculate gather counts and displacements
    if rank == 0:
        result_counts = np.array([rows_per_process + (1 if i < m % size else 0) for i in range(size)]) * n
        result_displacements = np.zeros(size, dtype=int)
        for i in range(1, size):
            result_displacements[i] = result_displacements[i-1] + result_counts[i-1]
        
        # Flatten result_matrix for Gatherv
        result_matrix_1d = result_matrix.flatten()
    else:
        result_counts = None
        result_displacements = None
        result_matrix_1d = None
    
    # Flatten local_result for Gatherv
    local_result_1d = local_result.flatten()
    
    # Gather local results
    comm.Gatherv(local_result_1d, 
                [result_matrix_1d, result_counts, result_displacements, MPI.DOUBLE] if rank == 0 else None, 
                root=0)
    
    # Process 0 saves the result and outputs
    if rank == 0:
        # Reshape result back to 2D
        result_matrix = result_matrix_1d.reshape(m, n)
        
        # Create output filename
        output_file = os.path.join('uploads', f"result_matrix_{time.time()}.csv")
        
        # Save the result matrix
        np.savetxt(output_file, result_matrix, delimiter=',')
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Output results
        print(output_file)
        print(f"{elapsed_time}")

if __name__ == "__main__":
    main() 