from .universe import universe
from .Utils.logging_utils import LoggingClass

import os
import numpy as np
from numpy import pi as Pi
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

from pycuda import driver, compiler, gpuarray
import jinja2

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

__logmodule__ = True

# Set directories for correct handling of paths
selfDir = os.path.dirname(os.path.abspath(__file__))
softwareDir = os.path.abspath(os.path.join(selfDir, os.pardir))


def spher2cart(points):
    # Retrieve the actual data
    r = points[:, 0]
    theta = points[:, 1]
    phi = points[:, 2]

    cosT = np.cos(theta)
    sinT = np.sin(theta)
    cosP = np.cos(phi)
    sinP = np.sin(phi)

    x = r * sinT * cosP
    y = r * sinT * sinP
    z = r * cosT

    return x, y, z


SPHERE = 0
DISK = 1
HORIZON = 2
STRAIGHT = 3


class RayTracer(metaclass=LoggingClass):
    """Relativistic spacetime ray tracer.

    This class generates images of what an observer would see near a rotating
    black hole.

    This is an abstraction layer over the CUDA kernel that integrates the ODE
    system specified in equations (A.15) of Thorne's paper. It integrates,
    backwards in time, a set of rays near a Kerr black hole, computing its
    trajectories from the focal point of a camera located near the black hole.

    The RayTracer class hides all the black magic behind the CUDA code, giving
    a nice and simple interface to the user that just wants some really cool,
    and scientifically accurate, images.

    Given a scene composed by a camera, a Kerr metric and a black hole, the
    RayTracer just expects a time :math:`x_{end}` to solve the system.

    Example:
        Define the characteristics of the black hole and build it::

            spin = 0.9999999999
            innerDiskRadius = 9
            outerDiskRadius = 20
            blackHole = BlackHole(spin, innerDiskRadius, outerDiskRadius)

        Define the specifications of the camera and build it::

            camR = 30
            camTheta = 1.511
            camPhi = 0
            camFocalLength = 3
            camSensorShape = (1000, 1000)  # (Rows, Columns)
            camSensorSize = (2, 2)         # (Height, Width)
            camera = Camera(camR, camTheta, camPhi,
                            camFocalLength, camSensorShape, camSensorSize)

        Create a Kerr metric with the previous two objects::

            kerr = KerrMetric(camera, blackHole)

        Set the speed of the camera once the Kerr metric and the black hole are
        created: it needs some info from both of these objects::

            camera.setSpeed(kerr, blackHole)

        Finally, build the raytracer with the camera, the metric and the black
        hole...::

            rayTracer = RayTracer(camera, kerr, blackHole)

        ...and generate the image!::

            rayTracer.rayTrace(-90)
            rayTracer.synchronise()
            rayTracer.plotImage()
    """
    def __init__(self, camera, debug=False):
        self.debug = debug
        self.systemSize = 5

        # Set up the necessary objects
        self.camera = camera

        # Get the number of rows and columns of the final image
        self.imageRows = self.camera.sensorShape[0]
        self.imageCols = self.camera.sensorShape[1]
        self.numPixels = self.imageRows * self.imageCols

        # Compute the block and grid sizes: given a fixed block dimension of 64
        # threads (in an 8x8 shape), the number of blocks are computed to get
        # at least as much threads as pixels

        # Fixed size block dimension: 8x8x1
        self.blockDimCols = 8
        self.blockDimRows = 8
        self.blockDim = (self.blockDimCols, self.blockDimRows, 1)

        # Grid dimension computed to cover all the pixels with a thread (there
        # will be some idle threads)
        self.gridDimCols = int(((self.imageCols - 1) / self.blockDimCols) + 1)
        self.gridDimRows = int(((self.imageRows - 1) / self.blockDimRows) + 1)

        self.gridDim = (self.gridDimCols, self.gridDimRows, 1)

        print(self.blockDim, self.gridDim)

        # Render the kernel
        self._kernelRendering()

        # Compute the initial conditions
        self._setUpInitCond()

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
            "IMG_ROWS": self.imageRows,
            "IMG_COLS": self.imageCols,
            "NUM_PIXELS": self.imageRows*self.imageCols,

            # Camera constants
            "D": self.camera.focalLength,
            "CAM_R": self.camera.r,
            "CAM_THETA": self.camera.theta,
            "CAM_PHI": self.camera.phi,
            "CAM_BETA": self.camera.speed,

            # Black hole constants
            "SPIN": universe.spin,
            "SPIN2": universe.spinSquared,
            "B1": universe.b1,
            "B2": universe.b2,
            "HORIZON_RADIUS": universe.horizonRadius,
            "INNER_DISK_RADIUS": universe.accretionDisk.innerRadius,
            "OUTER_DISK_RADIUS": universe.accretionDisk.outerRadius,

            # Kerr metric constants
            "RO": self.camera.metric.ro,
            "DELTA": self.camera.metric.delta,
            "POMEGA": self.camera.metric.pomega,
            "ALPHA": self.camera.metric.alpha,
            "OMEGA": self.camera.metric.omega,

            # Camera rotation angles
            "PITCH": np.float64(self.camera.pitch),
            "ROLL": np.float64(self.camera.roll),
            "YAW": np.float64(self.camera.yaw),

            # RK45 solver constants
            "R_TOL_I": 1e-6,
            "A_TOL_I": 1e-12,

            "SAFE": 0.9,
            "SAFE_INV": 1/0.9,

            "FAC_1": 0.2,
            "FAC_1_INV": 1 / 0.2,

            "FAC_2": 10.0,
            "FAC_2_INV": 1 / 10.0,

            "BETA": 0.04,
            "UROUND": 2.3e-16,

            "MIN_RESOL": -0.1,
            "MAX_RESOL": -2.0,

            # Constants for the alternative version of the solver
            "SOLVER_DELTA": 0.03125,
            "SOLVER_EPSILON": 1e-6,

            # Convention for ray status
            "SPHERE": SPHERE,  # A ray that has not yet collide with anything.
            "DISK": DISK,  # A ray that has collided with the disk.
            "HORIZON": HORIZON,  # A ray that has collided with the black hole.

            # Data type
            "REAL": codeType,

            # Number of equations
            "SYSTEM_SIZE": self.systemSize,
            "DATA_SIZE": 2,

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
        self._solve = mod.get_function("kernel")

        # Get the image generation function from the compiled module
        self.generateImage = mod.get_function("generate_image")

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

            np.float64(self.camera.pixelWidth),
            np.float64(self.camera.pixelHeight),

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes the direction of one pixel
            grid=self.gridDim,

            # Block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one equation
            block=self.blockDim
        )

        # TODO: Remove this copy, inefficient!
        # Retrieve the computed initial conditions
        self.systemState = self.systemStateGPU.get()
        self.constants = self.constantsGPU.get()

    def callKernel(self, x, xEnd):
        self._solve(
            np.float64(x),
            np.float64(xEnd),
            self.systemStateGPU,
            np.float64(-0.001),
            np.float64(xEnd - x),
            self.constantsGPU,
            self.rayStatusGPU,

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes the direction of one pixel
            grid=self.gridDim,

            # Block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one
            # equation
            block=self.blockDim
        )

    def rayTrace(self, xEnd, kernelCalls=1):
        """
        Args:
            xEnd (float): Time in which the system will be integrated. After
                this method finishes, the value of the rays at t=xEnd will be
                known
            stepsPerKernel (integer): The number of steps each kernel call will
                compute; i.e., the host will call the kernel
                xEnd / (resolution*stepsPerKernel) times.

            resolution (float): The size of the interval that will be used to
                compute one solver step between successive calls to the
                collision detection method.
        """
        # Initialize current time
        x = np.float64(0)

        # Compute iteration interval
        interval = xEnd / kernelCalls

        # Send the rays to the outer space!
        for _ in range(kernelCalls):
            print(x, x+interval)
            # Start timing
            self.start.record()

            # Call the kernel!
            self.callKernel(x, x + interval)

            # Update time
            x += interval

            # End timing
            self.end.record()
            self.end.synchronize()

            # Calculate the run length
            self.totalTime += self.start.time_till(self.end)*1e-3

        self.synchronise()
        return self.rayStatus, self.systemState

    def slicedRayTrace(self, xEnd, numSteps=100):
        stepSize = xEnd / numSteps

        # Initialize plotData with the initial position of the rays
        self.plotData = np.zeros((self.imageRows, self.imageCols,
                                  3, numSteps+1))
        self.plotData[:, :, :, 0] = self.systemState[:, :, :3]

        # Initialize plotStatus with a matriz full of zeros
        self.plotStatus = np.empty((self.imageRows, self.imageCols,
                                   numSteps+1), dtype=np.int32)
        self.plotStatus[:, :, 0] = 0

        x = 0
        for step in range(numSteps):
            # Solve the system
            self.callKernel(x, x + stepSize)

            # Advance the step and synchronise
            x += stepSize
            self.synchronise()

            # Get the data and store it for future plot
            self.plotData[:, :, :, step + 1] = self.systemState[:, :, :3]
            self.plotStatus[:, :, step + 1] = self.rayStatus

        return self.plotStatus, self.plotData

    def synchronise(self):
        self.rayStatus = self.rayStatusGPU.get()
        self.systemState = self.systemStateGPU.get()

    def texturedImage(self, disk, sphere):
        """Image should be a 2D array where each entry is a 3-tuple of Reals
        between 0.0 and 1.0
        """

        diskGPU = gpuarray.to_gpu(disk)
        sphereGPU = gpuarray.to_gpu(sphere)

        self.image = np.empty((self.imageRows, self.imageCols, 3),
                              dtype=np.float64)
        imageGPU = gpuarray.to_gpu(self.image)

        self.generateImage(
            self.systemStateGPU,
            self.rayStatusGPU,

            diskGPU,
            np.int32(disk.shape[0]),
            np.int32(disk.shape[1]),

            sphereGPU,
            np.int32(sphere.shape[0]),
            np.int32(sphere.shape[1]),

            imageGPU,

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes the direction of one pixel
            grid=self.gridDim,

            # Block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one equation
            block=self.blockDim
        )

        self.image = imageGPU.get()

        return self.image
