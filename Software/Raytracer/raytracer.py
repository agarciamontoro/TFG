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

# Convention for pixel colors
CELESTIAL_SPHERE = 1
HORIZON = 0

# Set directories for correct handling of paths
selfDir = os.path.dirname(os.path.abspath(__file__))
softwareDir = os.path.abspath(os.path.join(selfDir, os.pardir))


class BlackHole:
    def __init__(self, spin):
        # Define spin and its square
        self.a = spin
        self.a2 = spin**2

        # Interval over the radius of trapped photons' orbits run. See (A.6)
        self.r1 = 2.*(1. + cos((2./3.)*arccos(-self.a)))
        self.r2 = 2.*(1. + cos((2./3.)*arccos(+self.a)))

        # Necessary constants for the origin algorithm. See (A.13)
        self.b1 = self._b0(self.r2)
        self.b2 = self._b0(self.r1)

    def _b0(self, r):
        a = self.a
        a2 = self.a2

        return - (r**3. - 3.*(r**2.) + a2*r + a2) / (a*(r-1.))


class KerrMetric:
    def __init__(self, camera, blackHole):
        # Retrieve blackhole's spin and its square
        a = blackHole.a
        a2 = blackHole.a2

        # Retrieve camera radius, its square and camera theta
        r = camera.r
        r2 = camera.r2
        theta = camera.theta

        # Compute the constants described between (A.1) and (A.2)
        ro = sqrt(r2 + a2 * cos(theta)**2)
        delta = r2 - 2*r + a2
        sigma = sqrt((r2 + a2)**2 - a2 * delta * sin(theta)**2)
        alpha = ro * sqrt(delta) / sigma
        omega = 2 * a * r / (sigma**2)

        # Wut? pomega? See https://en.wikipedia.org/wiki/Pi_(letter)#Variant_pi
        pomega = sigma * sin(theta) / ro

        # Assign the values to the class
        self.ro = ro
        self.delta = delta
        self.sigma = sigma
        self.alpha = alpha
        self.omega = omega
        self.pomega = pomega


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

        # Build the solver object
        self._setUpSolver()

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
        template = templateEnv.get_template("definitions.jj2")

        codeType = "double"

        # Specify any input variables to the template as a dictionary.
        templateVars = {
            "SPIN": self.blackHole.a,
            "REAL": codeType,
            "SYSTEM_SIZE": 5,
            "DEBUG": "#define DEBUG" if self.debug else ""
        }

        # Finally, process the template to produce our final text.
        kernel = template.render(templateVars)

        # Store it in the file that will be included by all the other compiled
        # files
        filePath = os.path.join(selfDir, 'definitions.cu')
        with open(filePath, 'w') as outputFile:
            outputFile.write(kernel)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        kernelFile = os.path.join(selfDir, "raytracer_kernel.cu")

        mod = compiler.SourceModule(open(kernelFile, "r").read(),
                                    include_dirs=[selfDir, softwareDir])

        # Get the kernel function from the compiled module
        self._setInitialConditions = mod.get_function("setInitialConditions")

    def _setUpInitCond(self):
        # Array to compute the initial conditions
        self.initCond = np.empty((self.imageRows, self.imageCols,
                                  self.systemSize + 2))

        # Send it to the GPU
        self.initCondGPU = gpuarray.to_gpu(self.initCond)

        # Compute the initial conditions
        self._setInitialConditions(
            self.initCondGPU,

            np.float64(self.imageRows),
            np.float64(self.imageCols),
            np.float64(self.camera.pixelWidth),
            np.float64(self.camera.pixelHeight),
            np.float64(self.camera.focalLength),
            np.float64(self.camera.r),
            np.float64(self.camera.theta),
            np.float64(self.camera.phi),
            np.float64(self.camera.beta),

            np.float64(self.blackHole.a),
            np.float64(self.blackHole.b1),
            np.float64(self.blackHole.b2),

            np.float64(self.kerr.ro),
            np.float64(self.kerr.delta),
            np.float64(self.kerr.pomega),
            np.float64(self.kerr.alpha),
            np.float64(self.kerr.omega),

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes the direction of one pixel
            grid=(self.imageCols, self.imageRows, 1),

            # Block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one equation
            block=(1, 1, 1)
        )

        # Retrieve the computed initial conditions
        self.initCond = self.initCondGPU.get()

        # Build and fill the array for the system state with the initial
        # conditions
        self.systemState = np.copy(self.initCond[:, :, :5])

        # Retrieve the constants
        self.constants = np.copy(self.initCond[:, :, 5:])

    def _setUpSolver(self):
        filePath = os.path.abspath(os.path.join(selfDir, "functions.cu"))
        self.solver = RK4Solver(0, self.systemState, -0.001, filePath,
                                additionalData=self.constants,
                                debug=self.debug)

    def rayTrace(self, xEnd):
        self.systemState = self.solver.solve(xEnd)
        self.status = self.solver.status
