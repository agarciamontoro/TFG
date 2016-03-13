#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N-body simulation running in PyCUDA
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
from PIL import Image
import jinja2
from progress_lib import progress_bar_init
import time

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

# ======================== KERNEL TEMPLATE RENDERING ======================== #

# We must construct a FileSystemLoader object to load templates off
# the filesystem
templateLoader = jinja2.FileSystemLoader(searchpath="./")

# An environment provides the data necessary to read and
# parse our templates.  We pass in the loader object here.
templateEnv = jinja2.Environment(loader=templateLoader)

# Read the template file using the environment object.
# This also constructs our Template object.
template = templateEnv.get_template("kernel.cu")

# Specify any input variables to the template as a dictionary.
templateVars = {
    "softeningSquared": "0.01f"
}

# Finally, process the template to produce our final text.
kernel = template.render(templateVars)

# =========================== KERNEL COMPILATION =========================== #

# Compile the kernel code using pycuda.compiler
mod = compiler.SourceModule(kernel)

# Get the kernel function from the compiled module
galaxyKernel = mod.get_function("galaxyKernel")

# ============================== DATA LOADING ============================== #

# Load data from file
data = np.loadtxt("./Data/dubinski.tab.gz")

# Positions and masses of all the objects in the files
d_pos = np.column_stack((data[:49152:6, 1:4], data[:49152:6, 0]))

# Velocities (and a fancy 1.0) of all the objects in the files
d_vel = np.column_stack((data[:49152:6, 4:], np.ones(8192)))

# Array
data_cpu = np.array([d_pos, d_vel]).astype(np.float32)

# Transfer host (CPU) memory to device (GPU) memory
data_gpu = gpuarray.to_gpu(data_cpu)

# ============================ ACTUAL PROCESSING ============================ #

# Create two timers so we measure time
start = driver.Event()
end = driver.Event()

start.record()  # start timing

num_frames = 750
progress_bar = progress_bar_init(num_frames-1)

for frame in range(num_frames):
    # Start time measure
    pb_start = time.time()

    # Call the kernel on the card
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
    # Copy back the results (the array returned by PyCUDA has the same shape as
    # the one previously sent); i.e., [positions+mass, velocities+1.0]
    d_pos, d_vel = data_gpu.get()
    np.savetxt("Output/out_%03d.csv" % frame, d_pos[:, 0:3])

    # End time measure and update progress bar
    pb_end = time.time()
    progress_bar(pb_end - pb_start)

end.record()    # end timing

# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("Finished in", secs, "seconds.")
