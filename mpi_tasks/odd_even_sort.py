#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import sys
import time

def odd_even_sort(comm, local_data):
    """
    Implement odd-even transposition sort using MPI.
    Each process sorts its local data and then participates in the global sort.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Sort local data first
    local_data.sort()
    
    # Perform odd-even transposition sort across processes
    for phase in range(size):
        # Determine communication partner based on odd/even phase
        if phase % 2 == 0:  # Even phase
            if rank % 2 == 0:  # Even rank
                if rank < size - 1:
                    # Send to higher rank and receive
                    partner = rank + 1
                    comm.send(local_data, dest=partner)
                    received_data = comm.recv(source=partner)
                    # Merge and keep smaller half
                    merged = np.sort(np.concatenate((local_data, received_data)))
                    local_data = merged[:len(local_data)]
            else:  # Odd rank
                # Receive from lower rank and send
                partner = rank - 1
                received_data = comm.recv(source=partner)
                comm.send(local_data, dest=partner)
                # Merge and keep larger half
                merged = np.sort(np.concatenate((received_data, local_data)))
                local_data = merged[len(received_data):]
        else:  # Odd phase
            if rank % 2 == 1:  # Odd rank
                if rank < size - 1:
                    # Send to higher rank and receive
                    partner = rank + 1
                    comm.send(local_data, dest=partner)
                    received_data = comm.recv(source=partner)
                    # Merge and keep smaller half
                    merged = np.sort(np.concatenate((local_data, received_data)))
                    local_data = merged[:len(local_data)]
            elif rank > 0:  # Even rank > 0
                # Receive from lower rank and send
                partner = rank - 1
                received_data = comm.recv(source=partner)
                comm.send(local_data, dest=partner)
                # Merge and keep larger half
                merged = np.sort(np.concatenate((received_data, local_data)))
                local_data = merged[len(received_data):]
    
    return local_data

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Read input file (only rank 0)
    if rank == 0:
        if len(sys.argv) != 2:
            print("Usage: mpiexec -n <num_processes> python odd_even_sort.py <input_file>")
            comm.Abort(1)
        
        input_file = sys.argv[1]
        try:
            with open(input_file, 'r') as f:
                data = f.read().strip()
                numbers = [int(x.strip()) for x in data.split(',') if x.strip()]
        except Exception as e:
            print(f"Error reading input file: {e}")
            comm.Abort(1)
        
        # Record the start time
        start_time = time.time()
    else:
        numbers = None
        start_time = None
    
    # Broadcast the total number of elements to all processes
    if rank == 0:
        total_elements = len(numbers)
    else:
        total_elements = None
    total_elements = comm.bcast(total_elements, root=0)
    
    # Calculate local array size and distribute data
    local_size = total_elements // size
    remainder = total_elements % size
    
    # Calculate displacement and local size for each process
    if rank < remainder:
        local_size += 1
        displacement = rank * local_size
    else:
        displacement = remainder * (local_size + 1) + (rank - remainder) * local_size
    
    # Create array to receive scattered data
    local_data = np.empty(local_size, dtype=int)
    
    # Scatter the data from process 0 to all processes
    if rank == 0:
        # Convert list to numpy array for scattering
        data_array = np.array(numbers, dtype=int)
        
        # Create sendbuf and displacement arrays for Scatterv
        send_counts = np.array([total_elements // size + (1 if r < remainder else 0) for r in range(size)])
        displacements = np.zeros(size, dtype=int)
        for i in range(1, size):
            displacements[i] = displacements[i-1] + send_counts[i-1]
        
        # Scatterv the data
        comm.Scatterv([data_array, send_counts, displacements, MPI.INT], local_data, root=0)
    else:
        comm.Scatterv(None, local_data, root=0)
    
    # Perform odd-even transposition sort
    sorted_local_data = odd_even_sort(comm, local_data)
    
    # Gather results back to process 0
    if rank == 0:
        # Create receive buffers for gathering
        recv_counts = np.array([total_elements // size + (1 if r < remainder else 0) for r in range(size)])
        recv_displacements = np.zeros(size, dtype=int)
        for i in range(1, size):
            recv_displacements[i] = recv_displacements[i-1] + recv_counts[i-1]
        
        # Create buffer for sorted data
        sorted_data = np.empty(total_elements, dtype=int)
        
        # Gatherv the sorted data
        comm.Gatherv(sorted_local_data, [sorted_data, recv_counts, recv_displacements, MPI.INT], root=0)
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Output sorted data and timing
        print(','.join(map(str, sorted_data)))
        print(f"{elapsed_time}")
    else:
        comm.Gatherv(sorted_local_data, None, root=0)

if __name__ == "__main__":
    main() 