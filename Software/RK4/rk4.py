#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4th order Runge-Kutta
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2
from progress_lib import progress_bar_init

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit


class RK4Solver:
    def __init__(self, x0, y0, dx, systemFunctions):
        """Builds a RungeKutta solver of 4h order"""

        # ============================ CONSTANTS ============================ #

        # Shape of the initial conditions matrix: width and height. This shape
        # will define the dimensions of the CUDA grid
        self.INIT_H = y0.shape[0]
        self.INIT_W = y0.shape[1]

        # Number of equations on the system
        self.SYSTEM_SIZE = y0.shape[2]

        # Size of the step
        self.STEP_SIZE = np.float32(dx)

        # System function
        self.F = [(str(i), f) for i, f in enumerate(systemFunctions)]

        # ======================= INITIAL CONDITIONS ======================= #

        self.x0 = np.float32(x0)
        self.y0 = y0.astype(np.float32)

        # ==================== KERNEL TEMPLATE RENDERING ==================== #

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
            "SYSTEM_SIZE": self.SYSTEM_SIZE,
            "SYSTEM_FUNCTIONS": self.F
        }

        # Finally, process the template to produce our final text.
        kernel = template.render(templateVars)

        print(kernel)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        mod = compiler.SourceModule(kernel)

        # Get the kernel function from the compiled module
        self.RK4Solve = mod.get_function("RK4Solve")

        # ========================== DATA TRANSFER ========================== #

        # Transfer host (CPU) memory to device (GPU) memory
        self.y0GPU = gpuarray.to_gpu(self.y0)

    def solve(self):
        # Call the kernel on the card
        self.RK4Solve(
            # Inputs
            self.x0,
            self.y0GPU,
            self.STEP_SIZE,

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes one RK4 step for a single initial condition
            grid=(self.INIT_W, self.INIT_H, 1),
            # block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one equation
            block=(self.SYSTEM_SIZE, 1, 1),
        )

        # Update the time in which the system solution is computed
        self.x0 = self.x0 + self.STEP_SIZE

        # Return the new data
        y1 = self.y0GPU.get()
        return(y1)
