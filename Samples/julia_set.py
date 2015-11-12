#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is created as a test to use pyCUDA to generate a julia fractal.
"""
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import matplotlib.pyplot as plt

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

# -----------------------------------------------------------------------------
# ¡Important note! This can be done manually using:
#
#  import pycuda.driver as cuda
#  device_nr=0 # Get the device 0 as the device we want to use
#  cuda.init() # Init pycuda driver
#  current_dev = cuda.Device(device_nr) # Device we are working on
#  ctx = current_dev.make_context() # Make a working context --> New concept: context
#  ctx.push() # Let context make the lead
# -----------------------------------------------------------------------------

# First, we start defining the Kernel. The kernel must accept
# linearized arrays, so, when we construct the indexes, we
# must take this into account. This sample kernel only
# add the two arrays.
#
#This Kernel must be written in C/C++.
#
# The Kernel is a string with the C/C++ code in python so beware of using
# % and \n in the code because python will interpret these as scape
# characters. Use %% and \\n instead

kernel_code_template = """
#include <stdio.h>

/**
 * Data type: complex number.
 * All of its functions -and constructor- are thought to be executed in
 * the GPU. As they are declared as __device__, even the constructor, only
 * functions executed in the GPU can use this type of data.
 *
 * The definition of the struct does not need to be declared with __device__:
 * this cuComplex declaration just declares a new type of data. As the compiler
 * only needs to know that the memory allocation -the construction of the
 * struct- and the operations regarding complex numbers -product and sum-
 * need to be done in the GPU, only this functions need to have the qualifier.
 **/
struct cuComplex{
    float r;
    float i;

    // Constructs a new complex number given its real and imaginary parts
    __device__ cuComplex( float a, float b ):
        r(a), i(b)
        {}

    // Returns the norm, squared, of the complex number.
    __device__ float magnitude2(){
        return r*r + i*i;
    }

    // Returns the product of two complex numbers.
    __device__ cuComplex operator*( const cuComplex& a ){
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    // Returns the sum of two complex numbers.
    __device__ cuComplex operator+( const cuComplex& a ){
        return cuComplex(r+a.r, i+a.i);
    }
};

/**
 * Returns whether an (x,y) point in the image belongs to Julia set.
 *
 * The function maps the (x,y) image point to a point in the complex
 * unit disk (if scale=1.0). Then, it computes a fixed number of
 * iterations of the Julia sequence: Z_{n+1} = Z_n^2 + C, where C is
 * a constant complex number.
 * If this iteration diverges; i.e., the norm of the result of the
 * iterations is greater than a fixed threshold, then the point is not
 * in the Julia set and the function returns 0.
 * In any other case, the point belongs to the Julia set and the function
 * returns 1.
 *
 * That was the first behaviour. Now, the function returns the number of
 * iterations used to know whether a point belongs to the set, always
 * normalized to 1. If the whole loop is done and the point has not
 * overpassed the threshold, then the point is considered to be in th set
 * and 1.0 is returned.
 *
 * This function is executed in the GPU and not callable from the host.
 * Obviously, all the variables used in it -including the complex numbers-
 * are stored and computed in the GPU.
 **/
__device__ unsigned int julia( int x, int y, int DIM, float scale ){
    // Interval homomorphism: [0,DIM] |-> [-scale,scale]
    float jx = scale * (float) (DIM/2 - x)/(DIM/2);
    float jy = scale * (float) (DIM/2 - y)/(DIM/2);

    int num_iter = 500;

    //cuComplex c(-0.8, 0.156);
    //cuComplex c(0.285, -0.01);
    cuComplex c(-0.4, 0.6);
    cuComplex a(jx, jy);

    unsigned int i = 0;
    for (i = 0; i < num_iter; i++) {
        a = a*a + c;
        if(a.magnitude2() > 1000){
            return i;
        }
    }

    return (unsigned int) 0;
}

/**
 * Function executed in the GPU as many times as pixels have the final image.
 *
 * The GPU block coordinates are used as the image point coordinates. The
 * function calls julia() with its block coordinates to know how close is the
 * point to belong to the set. This number in [0,1] is used to colour the pixel.
 *
 * The first argument, ptr, is a pointer to the GPU-stored image data.
 **/
__global__ void kernel( unsigned int *ptr){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int DIM = %(IMAGE_SIZE)s;
    float scale = %(SCALE)s;

    if(x < DIM && y < DIM){
        // Offset in the ptr data. Usual serialization of a 2D array.
        int offset = x + y * DIM ;

        float juliaValue = julia( x,y,DIM,scale );
        ptr[offset] = juliaValue;
    }
}
"""

# Define the (square) matrix size and the TILE size.


Image_size = 2**12

threads_per_block = 32
blocks_per_grid = int( (Image_size + (threads_per_block-1)) / threads_per_block )


# Create a cpu easily recognized array using numpy.
## -----------------------------------------------------------------------------
# ¡Important note!
#
# When this array will be transferred to gpy (next code line) it will arrive
# at the VRAM as a 1D array flattened by the usual linealisation technique
# BUT when this array will come back to the CPU it will be re-structured as a 2D
# array. Yay!
#
# Now this gif is really usefull:  http://www.reactiongifs.com/r/mgc.gif
# -----------------------------------------------------------------------------


image_cpu = np.zeros((Image_size, Image_size)).astype(np.uint32)

# Transfer host (CPU) memory to device (GPU) memory
image_gpu = gpuarray.to_gpu(image_cpu)

# get the kernel code from the template
# by specifying the constant MATRIX_SIZE
# This is only a dictionary substitution of the string.

kernel_code = kernel_code_template % {
    'IMAGE_SIZE': Image_size,
    'SCALE': 1.0
    }

# Compile the kernel code using pycuda.compiler
mod = compiler.SourceModule(kernel_code)

# Get the kernel function from the compiled module
julia_kernel = mod.get_function("kernel")


# create two timers so we measure time
start = driver.Event()
end = driver.Event()

start.record() # start timing

# call the kernel on the card
julia_kernel(
    # inputs
    image_gpu,
    # Grid definition -> number of blocks x number of blocks.
    grid = (blocks_per_grid,blocks_per_grid),
    # block definition -> number of threads x number of threads
    block = (threads_per_block, threads_per_block, 1),
    )

end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print "%f seconds" % (secs)

# Copy back the results
image_cpu = image_gpu.get()

# Print results

imgplot = plt.imshow(image_cpu)
plt.show()
