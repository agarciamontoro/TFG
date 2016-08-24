import os
import sys
import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, sqrt
from numpy import pi as Pi

from pycuda import driver, compiler, gpuarray, tools
import jinja2

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

# Import the solver
sys.path.append('../RK4')
from rk4 import RK4Solver

# Set directories for correct handling of paths
selfDir = os.path.dirname(os.path.abspath(__file__))
softwareDir = os.path.abspath(os.path.join(selfDir, os.pardir))


# Kindly borrowed from http://stackoverflow.com/a/14267825
def nextPowerOf2(x):
    return 1 << (x-1).bit_length()


# Dummy object for the camera (computation of the speed is done here)
class Camera:
    def __init__(self, r, theta, phi, focalLength, sensorShape, sensorSize):
        # Define position
        self.r = r
        self.r2 = r**2
        self.theta = theta
        self.phi = phi

        # Define lens properties
        self.focalLength = focalLength

        # Define sensor properties
        self.sensorSize = sensorSize
        self.sensorShape = sensorShape

        # Compute the width and height of a pixel, taking into account the
        # physical measure of the sensor (sensorSize) and the number of pixels
        # per row and per column on the sensor (sensorShape)

        # Sensor height and width in physical units
        H, W = sensorSize[0], sensorSize[1]

        # Number of rows and columns of pixels in the sensor
        rows, cols = sensorShape[0], sensorShape[1]

        # Compute width and height of a pixel
        self.pixelWidth = np.float(W) / np.float(cols)
        self.pixelHeight = np.float(H) / np.float(rows)

        self.minTheta = self.minPhi = np.inf
        self.maxTheta = self.maxPhi = -np.inf

    def setSpeed(self, kerr, blackHole):
        # Retrieve blackhole's spin and some Kerr constants
        a = blackHole.a
        r = self.r
        pomega = kerr.pomega
        omega = kerr.omega
        alpha = kerr.alpha

        # Define speed with equation (A.7)
        Omega = 1. / (a + r**(3./2.))
        self.beta = pomega * (Omega-omega) / alpha

        # FIXME: This is being forced to zero only for testing purposes.
        # Remove this line if you want some real fancy images.
        self.beta = 0


class RayTracer:
    def __init__(self, camera, kerr, blackHole, debug=False):
        self.debug = debug
        self.systemSize = 5

        self.numThreads = nextPowerOf2(self.systemSize)

        # Set up the necessary objects
        self.camera = camera
        self.kerr = kerr
        self.blackHole = blackHole

        self.imageRows = self.camera.sensorShape[0]
        self.imageCols = self.camera.sensorShape[1]

        # Render the kernel
        self._kernelRendering()

        # Compute the initial conditions
        self._setUpInitCond()

        # # Build the solver object
        # self._setUpSolver()

        # Create two timers to measure the time
        self.start = driver.Event()
        self.end = driver.Event()

        # Initialise a variatble to store the total time of computation between
        # all calls
        self.totalTime = 0.

    def _kernelRendering(self):
        # We must construct a FileSystemLoader object to load templates off
        # the filesystem
        templateLoader = jinja2.FileSystemLoader(searchpath=selfDir)

        # An environment provides the data necessary to read and
        # parse our templates.  We pass in the loader object here.
        templateEnv = jinja2.Environment(loader=templateLoader)

        # Read the template file using the environment object.
        # This also constructs our Template object.
        templatePath = os.path.join('Kernel', 'common.jj')
        template = templateEnv.get_template(templatePath)

        codeType = "double"

        # Specify any input variables to the template as a dictionary.
        templateVars = {
            # Camera constants
            "D": self.camera.focalLength,
            "CAM_R": self.camera.r,
            "CAM_THETA": self.camera.theta,
            "CAM_PHI": self.camera.phi,
            "CAM_BETA": self.camera.beta,

            # Black hole constants
            "SPIN": self.blackHole.a,
            "B1": self.blackHole.b1,
            "B2": self.blackHole.b2,
            "HORIZON_RADIUS": self.blackHole.horizonRadius,
            "INNER_DISK_RADIUS": self.blackHole.innerDiskRadius,
            "OUTER_DISK_RADIUS": self.blackHole.outerDiskRadius,

            # Kerr metric constants
            "RO": self.kerr.ro,
            "DELTA": self.kerr.delta,
            "POMEGA": self.kerr.pomega,
            "ALPHA": self.kerr.alpha,
            "OMEGA": self.kerr.omega,

            # RK45 solver constants
            "R_TOL_I": 1e-6,
            "A_TOL_I": 1e-12,
            "SAFE": 0.9,
            "FAC_1": 0.2,
            "FAC_2": 10.0,
            "BETA": 0.04,
            "UROUND": 2.3e-16,

            # Convention for ray status
            "SPHERE": 0,  # A ray that has not yet collide with anything.
            "DISK": 1,  # A ray that has collided with the disk.
            "HORIZON": 2,  # A ray that has collided with the black hole.

            # Data type
            "REAL": codeType,

            # Number of equations
            "SYSTEM_SIZE": self.systemSize,

            # Debug switch
            "DEBUG": "#define DEBUG" if self.debug else ""
        }

        # Finally, process the template to produce our final text.
        kernel = template.render(templateVars)

        # Store it in the file that will be included by all the other compiled
        # files
        filePath = os.path.join(selfDir, 'Kernel', 'common.cu')
        with open(filePath, 'w') as outputFile:
            outputFile.write(kernel)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        kernelFile = os.path.join(selfDir, "Kernel", "raytracer.cu")

        mod = compiler.SourceModule(open(kernelFile, "r").read(),
                                    include_dirs=[selfDir, softwareDir])

        # Get the initial kernel function from the compiled module
        self._setInitialConditions = mod.get_function("setInitialConditions")

        # Get the solver function from the compiled module
        self._solve = mod.get_function("RK4Solve")

        # # Get the collision detection function from the compiled module
        # self._detectCollisions = mod.get_function("detectCollisions")

    def _setUpInitCond(self):
        # Array to compute the ray's initial conditions
        self.systemState = np.empty((self.imageRows, self.imageCols,
                                     self.systemSize))

        # Array to compute the ray's constants
        self.constants = np.empty((self.imageRows, self.imageCols, 2))

        # Array to store the rays status:
        #   0: A ray that has not yet collide with anything.
        #   1: A ray that has collided with the horizon.
        #   2: A ray that has collided with the black hole.
        self.rayStatus = np.zeros((self.imageRows, self.imageCols),
                                  dtype=np.int32)

        # Send them to the GPU
        self.systemStateGPU = gpuarray.to_gpu(self.systemState)
        self.constantsGPU = gpuarray.to_gpu(self.constants)
        self.rayStatusGPU = gpuarray.to_gpu(self.rayStatus)

        # Compute the initial conditions
        self._setInitialConditions(
            self.systemStateGPU,
            self.constantsGPU,

            np.float64(self.imageRows),
            np.float64(self.imageCols),
            np.float64(self.camera.pixelWidth),
            np.float64(self.camera.pixelHeight),
            np.float64(self.camera.focalLength),

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes the direction of one pixel
            grid=(self.imageCols, self.imageRows, 1),

            # Block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one equation
            block=(1, 1, 1)
        )

        # TODO: Remove this copy, inefficient!
        # Retrieve the computed initial conditions
        self.systemState = self.systemStateGPU.get()
        self.constants = self.constantsGPU.get()

    def rayTrace(self, xEnd):
        # Initial time
        x = np.float64(0)

        # Number of calls
        steps = 100

        # Computed iteration interval
        interval = xEnd / steps

        # Send the rays to the outer space!
        for step in range(steps):

            self.start.record()  # start timing
            # Solve the system in order to update the state of each ray
            self._solve(
                x,
                np.float64(x + interval),
                self.systemStateGPU,
                np.float64(0.001),
                np.float64(interval),
                self.constantsGPU,
                np.int32(2),
                self.rayStatusGPU,

                # Grid definition -> number of blocks x number of blocks.
                # Each block computes the direction of one pixel
                grid=(self.imageCols, self.imageRows, 1),

                # Block definition -> number of threads x number of threads
                # Each thread in the block computes one RK4 step for one
                # equation
                block=(self.numThreads, 1, 1)
            )

            x += interval
            self.end.record()   # end timing
            self.end.synchronize()
            print(x)

            # Calculate the run length
            self.totalTime = self.totalTime + self.start.time_till(self.end)*1e-3
        print(self.totalTime)

    def getStatus(self):
        self.status = self.rayStatusGPU.get()
        return self.status
