#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N-body simulation running in PyCUDA
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
from PIL import Image

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
#  ctx = current_dev.make_context() # Make a working context -->
#  New concept: context
#  ctx.push() # Let context make the lead
# -----------------------------------------------------------------------------

# First, we start defining the Kernel. The kernel must accept
# linearized arrays, so, when we construct the indexes, we
# must take this into account.
#
# This Kernel must be written in C/C++.
#
# The Kernel is a string with the C/C++ code in python so beware of using
# % and \n in the code because python will interpret these as scape
# characters. Use %% and \\n instead

kernel_code_template = """
#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.

__global__ void galaxyKernel(float * pdata, float step, int bodies)
{

    // index for vertex (pos)
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockDim.x * gridDim.x + x;

    // index for global memory
    unsigned int posLoc = x * 4;
    unsigned int velLoc = y * 4;

    // position (last frame)
    float px = pdata[posLoc + 0];
    float py = pdata[posLoc + 1];
    float pz = pdata[posLoc + 2]; // velocity (last frame)
    float vx = pdata[velLoc + 0];
    float vy = pdata[velLoc + 1];
    float vz = pdata[velLoc + 2];

    // update gravity (accumulation): naive big loop
    float3 acc = {0.0f, 0.0f, 0.0f};
    float3 r;
    float distSqr, distCube, s;

    unsigned int ni = 0;

    for (int i = 0; i < bodies; i++)
    {

        ni = i * 4;

        r.x = pdata[ni + 0] - px;
        r.y = pdata[ni + 1] - py;
        r.z = pdata[ni + 2] - pz;

        distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
        distSqr += softeningSquared;

        float dist = sqrtf(distSqr);
        distCube = dist * dist * dist;

        s = pdata[ni + 3] / distCube;

        acc.x += r.x * s;
        acc.y += r.y * s;
        acc.z += r.z * s;

    }

    // update velocity with above acc
    vx += acc.x * step;
    vy += acc.y * step;
    vz += acc.z * step;

    // update position
    px += vx * step;
    py += vy * step;
    pz += vz * step;

    // thread synch
    __syncthreads();

    // update global memory with update value (position, velocity)
    pdata[posLoc + 0] = px;
    pdata[posLoc + 1] = py;
    pdata[posLoc + 2] = pz;
    pdata[velLoc + 0] = vx;
    pdata[velLoc + 1] = vy;
    pdata[velLoc + 2] = vz;

}
"""

# File loading
data = np.loadtxt("../Data/dubinski.tab.gz")

# Positions and masses of all the objects in the files
d_pos = np.column_stack((data[:, 4:], data[:, 0]))

# Velocities (and a fancy 1.0) of all the objects in the files
d_vel = np.column_stack((data[:, 1:4], np.ones(data.shape[0])))

# Array
data_cpu = np.concatenate((d_pos, d_vel)).astype(np.float32)

# Transfer host (CPU) memory to device (GPU) memory
data_gpu = gpuarray.to_gpu(data_cpu)

# get the kernel code from the template
kernel_code = kernel_code_template

# Compile the kernel code using pycuda.compiler
mod = compiler.SourceModule(kernel_code)

# Get the kernel function from the compiled module
galaxyKernel = mod.get_function("galaxyKernel")


# create two timers so we measure time
start = driver.Event()
end = driver.Event()

start.record()  # start timing

# call the kernel on the card
galaxyKernel(
    # inputs
    data_gpu,
    np.float32(1.0),
    np.int32(1024),

    # Grid definition -> number of blocks x number of blocks.
    grid=(16, 16),
    # block definition -> number of threads x number of threads
    block=(16, 2, 1),
)

end.record()    # end timing

# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print(secs, "seconds")

# Copy back the results
data_cpu = data_gpu.get()
