#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
from PIL import Image
import sys
import time
import os

def apply_grayscale(image_chunk):
    """Apply grayscale filter to an image chunk"""
    # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
    return np.dot(image_chunk[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def apply_blur(image_chunk, kernel_size=5):
    """Apply blur filter to an image chunk"""
    # Create a simple blur kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    height, width = image_chunk.shape[:2]
    
    # Apply convolution for each channel
    result = np.zeros_like(image_chunk)
    
    # Apply padding to the chunk to handle boundaries
    pad = kernel_size // 2
    padded = np.pad(image_chunk, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    # Apply convolution
    for y in range(height):
        for x in range(width):
            for c in range(3):  # channels
                result[y, x, c] = np.sum(padded[y:y+kernel_size, x:x+kernel_size, c] * kernel)
    
    return result.astype(np.uint8)

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Process 0 reads the input image
    if rank == 0:
        if len(sys.argv) != 3:
            print("Usage: mpiexec -n <num_processes> python image_processor.py <input_image> <filter_type>")
            comm.Abort(1)
        
        input_file = sys.argv[1]
        filter_type = sys.argv[2]  # 'grayscale' or 'blur'
        
        if filter_type not in ['grayscale', 'blur']:
            print("Filter type must be 'grayscale' or 'blur'")
            comm.Abort(1)
        
        try:
            image = Image.open(input_file)
            img_array = np.array(image)
        except Exception as e:
            print(f"Error reading input image: {e}")
            comm.Abort(1)
        
        # Record the start time
        start_time = time.time()
        
        # Get image dimensions
        height, width = img_array.shape[:2]
        
        # Split the image into chunks for each process
        rows_per_process = height // size
        image_chunks = []
        
        for i in range(size):
            start_row = i * rows_per_process
            end_row = height if i == size - 1 else (i + 1) * rows_per_process
            image_chunks.append(img_array[start_row:end_row])
    else:
        image_chunks = None
        filter_type = None
        img_array = None
        start_time = None
    
    # Broadcast the filter type to all processes
    filter_type = comm.bcast(filter_type, root=0)
    
    # Scatter the image chunks to all processes
    local_chunk = comm.scatter(image_chunks, root=0)
    
    # Apply the selected filter
    if filter_type == 'grayscale':
        processed_chunk = apply_grayscale(local_chunk)
        # Convert grayscale to 3D array for consistency
        if len(processed_chunk.shape) == 2:
            processed_chunk = np.stack((processed_chunk,) * 3, axis=-1)
    else:  # blur
        processed_chunk = apply_blur(local_chunk)
    
    # Gather processed chunks
    processed_chunks = comm.gather(processed_chunk, root=0)
    
    # Process 0 combines and saves the result
    if rank == 0:
        # Combine processed chunks
        processed_image = np.vstack(processed_chunks)
        
        # Create output filename
        file_dir, file_name = os.path.split(input_file)
        name, ext = os.path.splitext(file_name)
        output_file = os.path.join('uploads', f"{name}_{filter_type}{ext}")
        
        # Convert to PIL Image and save
        result_image = Image.fromarray(processed_image)
        result_image.save(output_file)
        
        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Output results
        print(output_file)
        print(f"{elapsed_time}")

if __name__ == "__main__":
    main() 