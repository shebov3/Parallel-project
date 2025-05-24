#!/usr/bin/env python3
from mpi4py import MPI
import sys
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    words = None
    start_time = None

    if rank == 0:
        if len(sys.argv) != 2:
            print("Usage: mpiexec -n <num_processes> python file_processor.py <input_file>")
            comm.Abort(1)

        input_file = sys.argv[1]
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}")
            comm.Abort(1)

        start_time = time.time()

        # Split the text into words first
        all_words = text.split()
        total_words = len(all_words)

        # Divide the words evenly among processes
        chunk_size = total_words // size
        chunks = [all_words[i * chunk_size : (i + 1) * chunk_size] for i in range(size)]

        # Add remaining words to the last chunk
        chunks[-1].extend(all_words[size * chunk_size:])
    else:
        chunks = None

    # Scatter chunks of words
    local_words = comm.scatter(chunks, root=0)

    # Count local words
    local_word_count = len(local_words)
    local_unique_words = set(word.lower().strip(",.?!;:") for word in local_words)

    # Gather results
    all_word_counts = comm.gather(local_word_count, root=0)
    all_unique_sets = comm.gather(local_unique_words, root=0)

    if rank == 0:
        total_word_count = sum(all_word_counts)
        unique_words = set().union(*all_unique_sets)
        unique_word_count = len(unique_words)
        elapsed_time = time.time() - start_time

        # Output
        print(total_word_count)
        print(unique_word_count)
        print(elapsed_time)

if __name__ == "__main__":
    main()
