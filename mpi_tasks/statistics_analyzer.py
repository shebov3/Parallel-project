#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import time
import csv
from collections import Counter

def calculate_local_statistics(data):
    """Calculate basic statistics for each column in the local data chunk"""
    stats = {}
    
    # Loop through each column
    for col_idx in range(data.shape[1]):
        column = data[:, col_idx]
        
        # Calculate statistics
        stats[col_idx] = {
            'count': len(column),
            'sum': np.sum(column),
            'min': np.min(column),
            'max': np.max(column),
            'sum_squares': np.sum(column ** 2),
            'values': column.tolist()  # For calculating median and mode later
        }
    
    return stats

def finalize_statistics(all_stats):
    """Combine local statistics to get global statistics"""
    final_stats = {}
    
    # Get all column indices
    all_columns = set()
    for proc_stats in all_stats:
        all_columns.update(proc_stats.keys())
    
    # Calculate global statistics for each column
    for col_idx in all_columns:
        # Combine statistics from all processes for this column
        count = sum(proc_stats[col_idx]['count'] for proc_stats in all_stats if col_idx in proc_stats)
        total_sum = sum(proc_stats[col_idx]['sum'] for proc_stats in all_stats if col_idx in proc_stats)
        min_val = min(proc_stats[col_idx]['min'] for proc_stats in all_stats if col_idx in proc_stats)
        max_val = max(proc_stats[col_idx]['max'] for proc_stats in all_stats if col_idx in proc_stats)
        sum_squares = sum(proc_stats[col_idx]['sum_squares'] for proc_stats in all_stats if col_idx in proc_stats)
        
        # Combine all values for median and mode calculation
        all_values = []
        for proc_stats in all_stats:
            if col_idx in proc_stats:
                all_values.extend(proc_stats[col_idx]['values'])
        
        # Calculate mean
        mean = total_sum / count if count > 0 else 0
        
        # Calculate median
        all_values.sort()
        if len(all_values) % 2 == 0:
            median = (all_values[len(all_values)//2 - 1] + all_values[len(all_values)//2]) / 2
        else:
            median = all_values[len(all_values)//2]
        
        # Calculate mode
        counter = Counter(all_values)
        mode = counter.most_common(1)[0][0] if counter else None
        
        # Calculate standard deviation
        variance = (sum_squares / count) - (mean ** 2) if count > 0 else 0
        std_dev = np.sqrt(variance) if variance > 0 else 0
        
        # Store final statistics
        final_stats[col_idx] = {
            'mean': mean,
            'median': median,
            'mode': mode,
            'min': min_val,
            'max': max_val,
            'std_dev': std_dev
        }
    
    return final_stats

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Process 0 reads the input file
    if rank == 0:
        if len(sys.argv) != 2:
            print("Usage: mpiexec -n <num_processes> python statistics_analyzer.py <input_file>")
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
            
            # Record the start time
            start_time = time.time()
            
            # Get column names (if any) or generate default names
            num_columns = data.shape[1]
            column_names = [f"Column {i+1}" for i in range(num_columns)]
            
            # Split data rows among processes
            rows_per_process = data.shape[0] // size
            data_chunks = []
            
            for i in range(size):
                start_row = i * rows_per_process
                end_row = data.shape[0] if i == size - 1 else (i + 1) * rows_per_process
                data_chunks.append(data[start_row:end_row])
        except Exception as e:
            print(f"Error processing input file: {e}")
            comm.Abort(1)
    else:
        data_chunks = None
        column_names = None
        start_time = None
    
    # Scatter data chunks to all processes
    local_data = comm.scatter(data_chunks, root=0)
    
    # Broadcast column names to all processes
    column_names = comm.bcast(column_names, root=0)
    
    # Calculate local statistics
    local_stats = calculate_local_statistics(local_data)
    
    # Gather statistics from all processes
    all_stats = comm.gather(local_stats, root=0)
    
    # Process 0 finalizes statistics and outputs results
    if rank == 0:
        # Finalize statistics
        final_stats = finalize_statistics(all_stats)
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Output results
        for col_idx, stats in final_stats.items():
            print(f"Column: {column_names[col_idx]}")
            print(f"Mean: {stats['mean']}")
            print(f"Median: {stats['median']}")
            print(f"Mode: {stats['mode']}")
            print(f"Min: {stats['min']}")
            print(f"Max: {stats['max']}")
            print(f"StdDev: {stats['std_dev']}")
            print()
        
        print(f"{elapsed_time}")

if __name__ == "__main__":
    main() 