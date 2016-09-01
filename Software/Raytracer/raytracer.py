import os
import sys
import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, sqrt
from numpy import pi as Pi
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

from pycuda import driver, compiler, gpuarray, tools
import jinja2

sys.path.append('../Utils')
from logging_utils import LoggingClass

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

# Dummy object for the camera (computation of the speed is done here)
class Camera(metaclass=LoggingClass):
    """Pinhole camera placed near a Kerr black hole.

    This class contains the necessary data to define a camera that is located on the coordinate system of a Kerr black hole.

    Attributes:
        r (double): Distance to the coordinate origin; i.e., distance to the
            black hole centre.
        r2 (double): Square of `r`.
        theta (double): Inclination of the camera with respect to the black
            hole.
        phi (double): Azimuth of the camera with respect to the black hole.
        focalLength (double): Distance between the focal point (where every row
            that reaces the camera has to pass through) and the focal plane
            (where the actual sensor/film is placed).
        sensorSize (tuple): 2-tuple that defines the physical dimensions of the
            sensor in the following way: `(Height, Width)`.
        sensorShape (tuple): 2-tuple that defines the number of pixels of the
            sensor in the following way: `(Number of rows, Number of columns)`.
        pixelWidth (double): Width of one single pixel in physical units. It is
            computed as `Number of columns / Sensor width`.
        pixelHeight (double): Height of one single pixel in physical units. It
            is computed as `Number of rows / Sensor height`.
        beta (double): Speed of the camera, that follows a circular orbit
            around the black hole in the equatorial plane. It is computed using
            the formula (A.7) of Thorne's paper.
    """

    def __init__(self, r, theta, phi, focalLength, sensorShape, sensorSize):
        """Builds the camera defined by `focalLength`, `sensorShape` and
        `sensorSize` and locates it at the passed coordinates :math:`(r_c,
        \\theta_c, \\phi_c)`

        Args:
            r (double): Distance to the coordinate origin; i.e., distance to
                the black hole centre.
            r2 (double): Square of `r`.
            theta (double): Inclination of the camera with respect to the black
                hole.
            phi (double): Azimuth of the camera with respect to the black hole.
            focalLength (double): Distance between the focal point (where every
                row that reaces the camera has to pass through) and the focal
                plane (where the actual sensor/film is placed).
            sensorSize (tuple): 2-tuple that defines the physical dimensions of
                the sensor in the following way: `(Height, Width)`.
            sensorShape (tuple): 2-tuple that defines the number of pixels of
                the sensor in the following way: `(Number of rows, Number of
                columns)`.
        """

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
        """Given a Kerr metric and a black hole, this method sets the speed of
        the camera at a circular orbit in the equatorial plane, following formula (A.7) of :cite:`thorne15`:

        .. math::
            \\beta = \\frac{\\varpi}{\\alpha}(\\Omega - \\omega),

        where :math:`\\Omega = \\frac{1}{a + r_c^{3/2}}` and the other constants are the ones defined in the Kerr metric object. See :class:`~.KerrMetric`.

        Args:
            kerr (:class:`~.KerrMetric`): A :class:`~.KerrMetric` object containing the constants needed for the
                equations.
            blackHole (:class:`~.BlackHole`): A :class:`~.BlackHole` object containing the
                specifications of the black hole located a the coordinate
                origin.
        """

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


class RayTracer(metaclass=LoggingClass):
    """Relativistic spacetime ray tracer.

    This class generates images of what an observer would see near a rotating
    black hole.

    This is an abstraction layer over the CUDA kernel that integrates the ODE
    system specified in equations (A.15) of Thorne's paper. It integrates,
    backwards in time, a set of rays near a Kerr black hole, computing its
    trajectories from the focal point of a camera located near the black hole.

    The RayTracer class hides all the black magic behind the CUDA code, giving
    a nice and simple interface to the user that just wants some really cool, and scientifically accurate, images.

    Given a scene composed by a camera, a Kerr metric and a black hole, the RayTracer just expects a time :math:`x_{end}` to solve the system.

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
    def __init__(self, camera, kerr, blackHole, debug=False):
        self.debug = debug
        self.systemSize = 5

        # Set up the necessary objects
        self.camera = camera
        self.kerr = kerr
        self.blackHole = blackHole

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
            "CAM_BETA": self.camera.beta,

            # Black hole constants
            "SPIN": self.blackHole.a,
            "SPIN2": self.blackHole.a**2,
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

    def generate3Dscene(self, xEnd, numSteps=100):
            stepSize = xEnd / numSteps

            # Initialize plotData with the initial position of the rays
            self.plotData = np.zeros((self.imageRows, self.imageCols,
                                      3, numSteps+1))
            self.plotData[:, :, :, 0] = self.systemState[:, :, :3]

            # Initialize plotStatus with a matriz full of zeros
            self.plotStatus = np.empty((self.imageRows, self.imageCols,
                                       numSteps+1))
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

    def plotImage(self):
        # Start figure
        fig = plt.figure()

        image = np.empty((self.imageRows, self.imageCols, 3))

        for row in range(0, self.imageRows):
            for col in range(0, self.imageCols):
                status = self.rayStatus[row, col]

                if status == DISK:
                    pixel = [1, 0, 0]

                if status == HORIZON:
                    pixel = [0, 0, 0]

                if status == SPHERE:
                    pixel = [1, 1, 1]

                image[row, col, :] = pixel

        plt.imshow(image)
        plt.show()

    def plotScene(self):
            # Start figure
            fig = plt.figure()

            # Start 3D plot
            ax = fig.gca(projection='3d')
            ax.set_axis_off()

            # Set axes limits
            ax.set_xlim3d(-25, 25)
            ax.set_ylim3d(-25, 25)
            ax.set_zlim3d(-25, 25)

            # Draw the scene
            self.drawAxes(ax)
            self.drawBlackHole(ax)
            # self.drawErgoSphere(ax)
            self.drawCamera(ax)

            # Draw the rays
            for row in range(0, self.imageRows):
                for col in range(0, self.imageCols):
                    ray = np.transpose(self.plotData[row, col, :, :])
                    self.drawRay(ax, ray, self.plotStatus[row, col, :])

            # Add a legend
            # ax.legend()

            # Show the plot
            plt.show()

    def drawErgoSphere(self, ax):
        a2 = self.blackHole.a2

        # Draw black hole
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        r = (2 + np.sqrt(4 - 4*a2*np.square(np.cos(v)))) / 2

        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_wireframe(x, y, z)

    def drawRay(self, ax, ray, status):
        rayColour = 'royalblue'

        # Detect if the ray collided with the disk, remove the following steps
        # and change its colour
        indicesDisk = np.where(status == DISK)[0]
        if indicesDisk.size > 0:
            firstCollision = indicesDisk[0]
            ray = ray[:firstCollision, :]
            rayColour = 'darkolivegreen'

        # Detect if the ray entered the horizon, remove the following steps
        # and change its colour
        indicesCollision = np.where(status == HORIZON)[0]
        if indicesCollision.size > 0:
            firstCollision = indicesCollision[0]
            ray = ray[:firstCollision, :]
            rayColour = 'maroon'

        # Compute cartesian coordinates of the ray
        x, y, z = spher2cart(ray)

        # Plot the ray!
        ax.plot(x, y, z, label='Ray', color=rayColour, linewidth=1.5)

    def drawCamera(self, ax):
        d = self.camera.r + self.camera.focalLength
        H = self.camera.sensorSize[0] / 2
        W = self.camera.sensorSize[1] / 2

        points = np.array([
            [d, W, H],
            [d, -W, H],
            [d, -W, -H],
            [d, W, -H],
            [d, W, H]
        ])

        ax.plot(points[:, 0], points[:, 1], points[:, 2])

    def drawAxes(self, ax, d=150):
        ax.plot((-d, d), (0, 0), (0, 0), 'grey')
        ax.plot((0, 0), (-d, d), (0, 0), 'grey')
        ax.plot((0, 0), (0, 0), (-d, d), 'gray')

    def drawBlackHole(self, ax):
        # Draw horizon
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        r = self.blackHole.horizonRadius

        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='black',
                        edgecolors='white', linewidth=0.15)

        # Draw accretion disk
        circle1 = Circle((0, 0), self.blackHole.innerDiskRadius,
                         facecolor='none')
        circle2 = Circle((0, 0), self.blackHole.outerDiskRadius,
                         facecolor='none')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        art3d.pathpatch_2d_to_3d(circle1, z=0, zdir='z')
        art3d.pathpatch_2d_to_3d(circle2, z=0, zdir='z')

    def synchronise(self):
        self.rayStatus = self.rayStatusGPU.get()
        self.systemState = self.systemStateGPU.get()

    def override_initial_conditions(self, r, cam_theta, cam_phi, theta_cs,
                                    phi_cs):
        # Calculate initial vector direction

        Nx = sin(theta_cs) * cos(phi_cs)
        Ny = sin(theta_cs) * sin(phi_cs)
        Nz = cos(theta_cs)

        # Convert the direction of motion to the FIDO's spherical orthonormal
        # basis. See (A.10)
        nR = Nx
        nTheta = Nz
        nPhi = Ny

        # Get canonical momenta

        ro = self.kerr.ro
        delta = self.kerr.delta
        pomega = self.kerr.pomega
        alpha = self.kerr.alpha
        omega = self.kerr.omega

        print(ro, delta, pomega, alpha, omega)

        # Compute energy as measured by the FIDO. See (A.11)
        E = 1

        # Compute the canonical momenta. See (A.11)
        pR = E * ro * nR / sqrt(delta)
        pTheta = E * ro * nTheta
        pPhi = E * pomega * nPhi

        # Set conserved quantities

        # Simplify notation
        theta = cam_theta
        a2 = self.blackHole.a2

        # Set conserved quantities. See (A.12)
        b = pPhi
        q = pTheta**2 + cos(theta)*(b**2 / sin(theta)**2 - a2)

        # HACK THE INITIAL CONDITIONS

        self.systemState[:, :, 0] = r
        self.systemState[:, :, 1] = cam_theta
        self.systemState[:, :, 2] = cam_phi
        self.systemState[:, :, 3] = pR
        self.systemState[:, :, 4] = pTheta

        self.constants[:, :, 0] = b
        self.constants[:, :, 1] = q

        self.systemStateGPU = gpuarray.to_gpu(self.systemState)
        self.constantsGPU = gpuarray.to_gpu(self.constants)
