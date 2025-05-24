#!/usr/bin/env python3
from mpi4py import MPI
import sys
import time
import re

def find_occurrences(text, keyword):
    """Find all occurrences of a keyword in text and return positions"""
    positions = []
    start = 0
    
    # Using regex to find all occurrences (case insensitive)
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    
    for match in pattern.finditer(text):
        positions.append(match.start())
    
    return positions

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Process 0 reads the input file
    if rank == 0:
        if len(sys.argv) != 3:
            print("Usage: mpiexec -n <num_processes> python text_search.py <input_file> <keyword>")
            comm.Abort(1)
        
        input_file = sys.argv[1]
        keyword = sys.argv[2]
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}")
            comm.Abort(1)
        
        # Record the start time
        start_time = time.time()
        
        # Split the text into chunks for each process
        text_length = len(text)
        chunk_size = text_length // size
        
        # Add overlap to ensure keywords that cross chunk boundaries are found
        overlap = len(keyword) - 1
        
        chunks = []
        offsets = []
        
        for i in range(size):
            start_pos = max(0, i * chunk_size - (0 if i == 0 else overlap))
            end_pos = text_length if i == size - 1 else (i + 1) * chunk_size
            
            chunks.append(text[start_pos:end_pos])
            offsets.append(start_pos)
    else:
        chunks = None
        offsets = None
        keyword = None
        start_time = None
    
    # Broadcast the keyword to all processes
    keyword = comm.bcast(keyword, root=0)
    
    # Scatter the text chunks and offsets to all processes
    local_chunk = comm.scatter(chunks, root=0)
    local_offset = comm.scatter(offsets, root=0)
    
    # Find occurrences in the local chunk
    local_positions = find_occurrences(local_chunk, keyword)
    
    # Adjust positions to be relative to the entire text
    adjusted_positions = [pos + local_offset for pos in local_positions]
    
    # Gather positions from all processes
    all_positions = comm.gather(adjusted_positions, root=0)
    
    # Process 0 combines results and outputs
    if rank == 0:
        # Flatten the list of positions
        all_occurrences = []
        for positions in all_positions:
            all_occurrences.extend(positions)
        
        # Sort the positions
        all_occurrences.sort()
        
        # Remove duplicates (for overlapping chunks)
        if all_occurrences:
            unique_occurrences = [all_occurrences[0]]
            for pos in all_occurrences[1:]:
                if pos != unique_occurrences[-1]:
                    unique_occurrences.append(pos)
        else:
            unique_occurrences = []
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Output results
        print(len(unique_occurrences))
        print(','.join(map(str, unique_occurrences)))
        print(f"{elapsed_time}")

if __name__ == "__main__":
    main() 