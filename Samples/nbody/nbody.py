#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N-body simulation running in PyCUDA
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2
from progress_lib import progress_bar_init
import os
import time
import h5py
import argparse

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

TILE_SIZE = 32

# ============================= PARSE ARGUMENTS ============================= #

parser = argparse.ArgumentParser(description='Simulates the collision between Andromeda and the Milky Way')

parser.add_argument('-b', '--bodies', dest='numBodies', type=int, default=8192,
                    choices=[i for i in range(1, 49153) if 49152 % i == 0]
                    help='Number of bodies (it has to be a divisor of 49152). Defaults to 8192.')

parser.add_argument('-g', '--gravity', dest='gConst', type=float, default=1.0,
                    help='Gravity constant. Defaults to 1.')

parser.add_argument('-d', '--dir', dest='outDir', type=str, default="Output/",
                    help='Directory where the output CSV files will be stored.')

parser.add_argument('-f', '--frames', dest='numFrames', type=int,
                    default=200,
                    help='Number of steps simulated.')

args = parser.parse_args()

NUM_BODIES = args.numBodies
G_CONST = args.gConst
OUT_DIR = args.outDir
NUM_FRAMES = args.numFrames

print("Computing simulation for %d bodies with gravity constant %.2f.\nOutput stored at %s" % (NUM_BODIES, G_CONST, OUT_DIR))

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
    "G_CONST": G_CONST,
    "EPS2": 0.01,
    "NUM_BODIES": NUM_BODIES,
    "TILE_SIZE": TILE_SIZE
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

# Filter out the last 32768 bodies, belonging to halos
# data = data[:49152, :]

# Filter NUM_BODIES bodies, evenly from disks and bulges.
skip = int(len(data) / NUM_BODIES)
data = data[::skip, :]

masses = data[:, 0]
positions = data[:, 1:4]
velocities = data[:, 4:]

# Positions and masses of all the objects in the files
d_pos = np.column_stack((positions, masses))

# Velocities (and a fancy 1.0) of all the objects in the files
d_vel = np.column_stack((velocities, np.ones(NUM_BODIES)))

# Array
pos_cpu = d_pos.astype(np.float32)
vel_cpu = d_vel.astype(np.float32)

# Transfer host (CPU) memory to device (GPU) memory
pos_gpu = gpuarray.to_gpu(pos_cpu)
vel_gpu = gpuarray.to_gpu(vel_cpu)

# ============================= DATA CONTAINER ============================= #

file_name = "nbody.hdf5"
full_path = os.path.join(OUT_DIR, file_name)

hdf_root = h5py.File(full_path, "w")

# ============================ ACTUAL PROCESSING ============================ #

# Create two timers so we measure time
start = driver.Event()
end = driver.Event()

start.record()  # start timing

progress_bar = progress_bar_init(NUM_FRAMES-1)

for frame in range(NUM_FRAMES):
    # Start time measure
    pb_start = time.time()

    # Call the kernel on the card
    galaxyKernel(
        # inputs
        pos_gpu,
        vel_gpu,
        np.float32(0.1),

        # Grid definition -> number of blocks x number of blocks.
        grid=(int(NUM_BODIES / TILE_SIZE), 1, 1),
        # block definition -> number of threads x number of threads
        block=(TILE_SIZE, 1, 1),
    )
    # Copy back the results (the array returned by PyCUDA has the same shape as
    # the one previously sent); i.e., [positions+mass, velocities+1.0]
    pos_cpu = pos_gpu.get()
    vel_cpu = vel_gpu.get()

    # Create file in HDF5 system
    hdf_root.create_dataset("out_%03d.csv" % frame,
                            data=pos_cpu[:, 0:3],
                            compression="gzip",
                            compression_opts=9)

    # End time measure and update progress bar
    pb_end = time.time()
    progress_bar(pb_end - pb_start)

end.record()    # end timing

# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("Finished in", secs, "seconds.")
