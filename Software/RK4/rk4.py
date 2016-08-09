#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4th order Runge-Kutta
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2
from progress_lib import progress_bar_init
import os
import time
import h5py

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

INIT_W = 1
INIT_H = 1
NUM_STEPS = 100
STEP_SIZE = 0.02

CONSTANT_A = 25
SYSTEM_SIZE = 2

BLOCK_W = 10
BLOCK_H = 10

OUT_DIR = "./Output"

# ======================== KERNEL TEMPLATE RENDERING ======================== #

# We must construct a FileSystemLoader object to load templates off
# the filesystem
templateLoader = jinja2.FileSystemLoader(searchpath="./")

# An environment provides the data necessary to read and
# parse our templates.  We pass in the loader object here.
templateEnv = jinja2.Environment(loader=templateLoader)

# Read the template file using the environment object.
# This also constructs our Template object.
template = templateEnv.get_template("rk4_kernel.cu")

# Specify any input variables to the template as a dictionary.
templateVars = {
    "CONSTANT_A": CONSTANT_A,
    "SYSTEM_SIZE": SYSTEM_SIZE
}

# Finally, process the template to produce our final text.
kernel = template.render(templateVars)

# =========================== KERNEL COMPILATION =========================== #

# Compile the kernel code using pycuda.compiler
mod = compiler.SourceModule(kernel)

# Get the kernel function from the compiled module
rungeKutta4 = mod.get_function("rungeKutta4")

# ============================== DATA LOADING ============================== #

# Load initial conditions
initConditions = np.random.rand(INIT_W, INIT_H, 3)
initConditions = np.array([[[-1, 1, 1]]], dtype=np.float32)
# Array
cond_cpu = initConditions.astype(np.float32)

# Transfer host (CPU) memory to device (GPU) memory
cond_gpu = gpuarray.to_gpu(cond_cpu)

# ============================= DATA CONTAINER ============================= #

file_name = "rk4.hdf5"
full_path = os.path.join(OUT_DIR, file_name)

print(full_path)

hdf_root = h5py.File(full_path, "w")

# ============================ ACTUAL PROCESSING ============================ #

# Create two timers to measure the time
start = driver.Event()
end = driver.Event()

start.record()  # start timing

progress_bar = progress_bar_init(NUM_STEPS-1)

print(cond_cpu)

for step in range(NUM_STEPS):
    # Start time measure
    pb_start = time.time()

    # Call the kernel on the card
    rungeKutta4(
        # inputs
        cond_gpu,
        np.float32(STEP_SIZE),

        # Grid definition -> number of blocks x number of blocks.
        grid=(1, 1, 1),
        # block definition -> number of threads x number of threads
        block=(1, 1, 1),
    )

    # Copy back the results (the array returned by PyCUDA has the same shape as
    # the one previously sent)
    cond_cpu = cond_gpu.get()

    print(cond_cpu)

    # Create file in HDF5 system
    hdf_root.create_dataset("out_%03d.csv" % step,
                            data=cond_cpu[:, :, :],
                            compression="gzip",
                            compression_opts=9)

    # End time measure and update progress bar
    pb_end = time.time()
    # progress_bar(pb_end - pb_start)

end.record()    # end timing

# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("Finished in", secs, "seconds.")
