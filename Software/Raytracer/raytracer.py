from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import newton
from tqdm import tqdm
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

def plotScene(plotData, camera, blackHole):
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
    drawAxes(ax)
    drawBlackHole(ax, blackHole)
    drawErgoSphere(ax, blackHole)
    drawCamera(ax, camera)

    # Draw the rays
    for row in range(10, 100, 10):
        for col in range(10, 100, 10):
            ray = np.transpose(plotData[row, col, :, :])
            drawRay(ax, ray)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


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


def drawErgoSphere(ax, blackHole):
    a2 = blackHole.a2

    # Draw black hole
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    r = (2 + np.sqrt(4 - 4*a2*np.square(np.cos(v)))) / 2

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z)

    # a2 = blackHole.a2
    #
    # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    # r = (2 + np.sqrt(4 - 4*a2*np.cos(v)**2)) / 2
    #
    # print(r)
    #
    # x = r * np.cos(u)*np.sin(v)
    # y = r * np.sin(u)*np.sin(v)
    # z = r * np.cos(v)
    #
    # ax.plot_wireframe(x, y, z, color="b")


def drawRay(ax, ray):
    rayColor = 'black'

    # Clean the data
    rowsZero = np.where(~ray.any(axis=1))[0]
    if(rowsZero.size != 0):
        ray = ray[:rowsZero[0], :]
        rayColor = 'red'

    x, y, z = spher2cart(ray)

    ax.plot(x, y, z, label='', color=rayColor)

def drawSphere(ax, x, y, z):
    pass

def drawCamera(ax, cam):
    d = cam.r + cam.focalLength
    H = cam.sensorSize[0] / 2
    W = cam.sensorSize[1] / 2

    points = np.array([
        [d, W, H],
        [d, -W, H],
        [d, -W, -H],
        [d, W, -H],
        [d, W, H]
    ])

    ax.plot(points[:, 0], points[:, 1], points[:, 2])


def drawAxes(ax, d=150):
    ax.plot((-d, d), (0, 0), (0, 0), 'grey')
    ax.plot((0, 0), (-d, d), (0, 0), 'grey')
    ax.plot((0, 0), (0, 0), (-d, d), 'gray')


def drawBlackHole(ax, blackHole):
    r = (2 + np.sqrt(4 - 4*blackHole.a2)) / 2

    # Draw black hole
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='black',
                    edgecolors='white')


# Necessary functions for the algorithm. See (A.5)
def b0(r, a):
    a2 = a**2
    return - (r**3. - 3.*(r**2.) + a2*r + a2) / (a*(r-1.))


def q0(r, a):
    r3 = r**3.
    a2 = a**2
    return - (r3*(r3 - 6.*(r**2.) + 9.*r - 4.*a2)) / (a2*((r-1.)**2.))


class BlackHole:
    def __init__(self, spin):
        # Define spin and its square
        self.a = spin
        self.a2 = spin**2

        # Interval over the radius of trapped photons' orbits run. See (A.6)
        self.r1 = 2.*(1. + cos((2./3.)*arccos(-self.a)))
        self.r2 = 2.*(1. + cos((2./3.)*arccos(+self.a)))

        # Necessary constants for the origin algorithm. See (A.13)
        self.b1 = b0(self.r2, self.a)
        self.b2 = b0(self.r1, self.a)


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
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        templateLoader = jinja2.FileSystemLoader(searchpath=currentDirectory)

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
        with open('definitions.cu', 'w') as outputFile:
            outputFile.write(kernel)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        ownDir = os.path.dirname(os.path.realpath(__file__))
        softwareDir = os.path.abspath(os.path.join(ownDir, os.pardir))

        mod = compiler.SourceModule(open("raytracer_kernel.cu", "r").read(),
                                    include_dirs=[ownDir, softwareDir])

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

        # # Send everything to the GPU
        # self.systemStateGPU = gpuarray.to_gpu(self.systemState)
        # self.constantsGPU = gpuarray.to_gpu(self.constants)

    def _setUpSolver(self):
        functionPath = os.path.abspath("functions.cu")
        self.solver = RK4Solver(0, self.systemState, -0.001, functionPath,
                                additionalData=self.constants,
                                debug=self.debug)

    def rayTrace(self, xEnd):
        self.systemState = self.solver.solve(xEnd)


if __name__ == '__main__':
    # Debug array
    cosas = []

    for _ in range(1):
        # Black hole spin
        spin = 0.000001

        # Camera position
        camR = 10
        camTheta = Pi/2
        camPhi = 0

        # Camera lens properties
        camFocalLength = 0.001
        camSensorShape = (101, 101)  # (Rows, Columns)
        camSensorSize = (0.01, 0.01)       # (Height, Width)

        # Create the black hole, the camera and the metric with the constants
        # above
        blackHole = BlackHole(spin)
        camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                        camSensorSize)
        kerr = KerrMetric(camera, blackHole)

        # Set camera's speed (it needs the kerr metric constants)
        camera.setSpeed(kerr, blackHole)

        # Create the raytracer!
        rayTracer = RayTracer(camera, kerr, blackHole, debug=False)

        # Set initial and final times, the number of the steps for the
        # simulation and compute the step size
        tInit = 0.
        tEnd = -9.
        numSteps = 50
        stepSize = (tEnd - tInit) / numSteps

        # Retrieve the initial state of the system for plotting purposes
        rays = rayTracer.systemState[:, :, :3]
        plotData = np.zeros(rays.shape + (numSteps+1, ))
        plotData[:, :, :, 0] = rays

        # Simulate!
        t = tInit
        for step in range(numSteps):
            # Advance the step
            t += stepSize

            # Solve the system
            rayTracer.rayTrace(t)

            # Get the data and store it for future plot
            plotData[:, :, :, step + 1] = rayTracer.systemState[:, :, :3]

        # Debug
        cosas.append(plotData)

        # Plot the scene
        plotScene(plotData, camera, blackHole)

    # Debug
    print(np.allclose(cosas[0], cosas[2]))
    print(np.max(np.abs(cosas[0]-cosas[2])))
    print(np.where(np.abs(cosas[0]-cosas[2])>1e-5))
