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


from multiprocessing import Pool
import os
# os.system("taskset -p 0xff %d" % os.getpid())

# Convention for pixel colors
CELESTIAL_SPHERE = 1
HORIZON = 0


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
        omega = 2 * a * r / sigma**2

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
        self.pixelWidth = W / cols
        self.pixelHeight = H / rows

        self.minTheta = self.minPhi = np.inf
        self.maxTheta = self.maxPhi = -np.inf

        for row in range(H):
            for col in range(W):
                theta, phi = self._pixelToRay(row, col)

                self.minTheta = theta if theta < self.minTheta else self.minTheta
                self.minPhi = phi if phi < self.minPhi else self.minPhi


                self.maxTheta = theta if theta > self.maxTheta else self.maxTheta
                self.maxPhi = phi if phi > self.maxPhi else self.maxPhi

    def setSpeed(self, kerr, blackHole):
        # Retrieve blackhole's spin and some Kerr constants
        a = blackHole.a
        pomega = kerr.pomega
        omega = kerr.omega
        alpha = kerr.alpha

        # Define speed with equation (A.7)
        Omega = 1. / (a + self.r**(3./2.))
        self.beta = pomega * (Omega-omega) / alpha

    def _pixelToRay(self, row, col):
        # Place a coordinate system centered in the focal point following
        # this convention for the axes:
        #   - The X axis is in the direction of the line that joins the black
        #   hole and the focal point. The positive part of the axis starts at
        #   the focal point and goes away from the black hole.
        #   - The Y axis is tangential to the camera orbit. Its positive part
        #   goes to the right (the same convention as with the pixels).
        #   - The Z axis is perpendicular to both of them, pointing upwards.
        # Now place a spherical coordinate system at the same place in the
        # usual manner; i.e.:
        #   - Theta starts at zero when aligned with the positive part of Z,
        #   goes down until it aligns with the line that joins the system
        #   center and the black hole (theta = pi/2) and continues its way
        #   until it aligns with the negative part of Z (theta = pi).
        #   - Phi starts at zero when aligned with the positive part of X
        #   (going away from the black hole), starts turning to its left (as
        #   seen from above) until it aligns with the negative part of Y (phi =
        #   pi/2), continues until it aligns with the line joining the focal
        #   point and the black hole (again the X axis, but now pointing to the
        #   black hole) (phi = pi), follows the rotation until it aligns with
        #   the positive part of Y (phi = 3*pi/2) and finish its travel in the
        #   original position (phi = 2*pi)
        # Place the CCD sensor center at the point (-d, 0, 0) and facing the
        # black hole: the pixels in each of its rows spread over the Y axis and
        # the pixels of the columns spread over the Z axis.
        # For every pixel -which has the form (-d, y, z), where y is the number
        # of the pixel column and z the number of the pixel row-, trace a ray
        # that starts at (-d, y, z) and passes through the focal points, whose
        # coordinates are (0, 0, 0). The direction of this ray, as measured by
        # the specified spherical system, are the following:

        # Retrieve the focal length
        d = self.focalLength

        # First compute the position of the pixel in physical units (taking
        # into accout the sensor size) measured in the cartesian coordinate
        # system described above
        y0 = - col * self.pixelWidth
        z0 = row * self.pixelHeight

        # Compute phi and theta using the above information.
        # TODO: Add a drawing here in some way, it will ease everything.
        rayPhi = Pi + arctan(y0 / d)
        rayTheta = Pi/2 + arctan(z0 / sqrt(d**2 + y0**2))

        return rayTheta, rayPhi


    def createRay(self, row, col, kerr, blackHole):
        imprimir = True if row == 184 and col == 170 else False

        row -= self.sensorShape[0] / 2.
        col -= self.sensorShape[1] / 2

        rayTheta, rayPhi = self._pixelToRay(row, col)

        # We can now create and return our ray :)
        return Ray(rayTheta, rayPhi, self, kerr, blackHole, imprimir)


# Dummy object for a ray (pixel->spherical transformation is done here)
class Ray:
    def __init__(self, theta, phi, camera, kerr, blackHole, imprimir):
        # Set direction in the camera's reference frame
        self.theta = theta
        self.phi = phi

        # Compute all necessary information for the ray
        self._setNormal()
        self._setDirectionOfMotion(camera)
        self._setCanonicalMomenta(kerr)
        self._setConservedQuantities(camera, blackHole)

        self.imprimir = imprimir
        if self.imprimir:
            print("b = ", self.b, " q = ", self.q)

    def _setNormal(self):
        # Cartesian components of the unit vector N pointing in the direction
        # of the incoming ray
        self.Nx = sin(self.theta) * cos(self.phi)
        self.Ny = sin(self.theta) * sin(self.phi)
        self.Nz = cos(self.theta)

    def _setDirectionOfMotion(self, camera):
        # Compute denominator, common to all the cartesian components
        den = 1. - camera.beta * self.Ny

        # Compute factor common to nx and nz
        fac = -sqrt(1. - camera.beta**2.)

        # Compute cartesian coordinates of the direction of motion. See(A.9)
        self.nY = (-self.Ny + camera.beta) / den
        self.nX = fac * self.Nx / den
        self.nZ = fac * self.Nz / den

        # Convert the direction of motion to the FIDO's spherical orthonormal
        # basis. See (A.10)
        self.nR = self.nX
        self.nTheta = -self.nZ
        self.nPhi = self.nY

    def _setCanonicalMomenta(self, kerr):
        # Retrieve constants
        ro = kerr.ro
        delta = kerr.delta
        pomega = kerr.pomega
        alpha = kerr.alpha
        omega = kerr.omega

        # Simplify notation, always worth it
        nR = self.nR
        nTheta = self.nTheta
        nPhi = self.nPhi

        # Compute energy as measured by the FIDO. See (A.11)
        E = 1 / (alpha + omega * pomega * nPhi)

        # Set conserved energy to unity. See (A.11)
        self.pt = -1

        # Compute the canonical momenta. See (A.11)
        self.pR = E * ro * nR / sqrt(delta)
        self.pTheta = E * ro * nTheta
        self.pPhi = E * pomega * nPhi

    def _setConservedQuantities(self, camera, blackHole):
        # Simplify notation
        theta = camera.theta
        a2 = blackHole.a2
        pPhi = self.pPhi
        pTheta = self.pTheta

        # Set conserved quantities. See (A.12)
        b = pPhi
        q = pTheta**2 + cos(theta)*(b**2 / sin(theta)**2 - a2)

        self.b = b
        self.q = q

    def traceRay(self, camera, blackHole):
        # Simplify notation
        b = self.b
        q = self.q
        b1 = blackHole.b1
        b2 = blackHole.b2
        a = blackHole.a
        a2 = blackHole.a2

        # Compute r0 such that b0(r0) = b. The computation of this number
        # involves complex numbers (there is a square root of a negative
        # number). Nevertheless, the imaginary parts cancel each other when
        # substracting the final terms. In order not to get np.sqrt errors
        # because of the negative argument, a conversion to complex is
        # forced summing a null imaginary part in the argument of sqrt (see
        # the + 0j below, in the innerSqrt assignation). After the final
        # computation is done, the real part is retrieved (the imaginary
        # part can be considered null).

        # Simplify notation by computing this factor before
        fac = -9 + 3*a2 + 3*a*b

        # Compute the square root of a complex number. Note the +0j
        innerSqrt = sqrt((54 - 54*a2)**2 + 4*(fac**3) + 0j)

        # Simplify notation by computing this cubic root
        cubicRoot = (54 - 54*a2 + innerSqrt)**(1/3)

        # Finish the computation with the above computed numbers
        r0 = 1 - ((2**(1/3))*fac)/(3*cubicRoot) + cubicRoot/(3*(2**(1/3)))

        # Retrieve the real part:
        r0 = np.real(r0)

        if self.imprimir:
            print("r_0 = ", r0);

        # No radial turning points (see A.13 and A.14)
        if b1 < b < b2 and q < q0(r0, a):
            if(self.pR > 0.):
                return HORIZON
            else:
                return CELESTIAL_SPHERE
        # There are two radial turning points. See (v), (c)
        else:
            # Coefficients of r^4, r^3, r^2, r^1 and r^0 from R(r)
            coefs = [1.,
                     0.,
                     -q - b**2. + a2,
                     2.*q + 2.*(b**2.) - 4.*a*b + 2.*a2,
                     -a2*q]

            # Get the roots of R(r) = 0
            roots = np.roots(coefs)

            # If there are real roots, get the maximum of them; otherwise, set
            # rUp to -inf
            realIndices = np.isreal(roots)
            rUp = np.amax(roots[realIndices]) if realIndices.any() else -np.inf

            # Decide if the camera radius is lower than rUp
            if camera.r < rUp:
                return HORIZON
            else:
                return CELESTIAL_SPHERE



class RayTracer:
    def __init__(self, camera, kerr, blackHole, debug=False):
        self.debug = debug

        # Set up the necessary objects
        self.camera = camera
        self.kerr = kerr
        self.blackHole = blackHole

        # Render the kernel
        self._kernelRendering()

        # Create image array in both the CPU and GPU
        self._createAndTransferImage()

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
        template = templateEnv.get_template("raytracer_kernel.cu")

        codeType = "double"

        # Specify any input variables to the template as a dictionary.
        templateVars = {
            # "SYSTEM_SIZE": self.SYSTEM_SIZE,
            "Real": codeType,
            "DEBUG": "#define DEBUG" if self.debug else ""
        }

        # Finally, process the template to produce our final text.
        kernel = template.render(templateVars)

        if(self.debug):
            kernelTmpFile = open("lastKernelRendered.cu", "w")
            kernelTmpFile.write(kernel)
            kernelTmpFile.close()

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        mod = compiler.SourceModule(kernel)

        # Get the kernel function from the compiled module
        self.rayTrace = mod.get_function("rayTrace")

    def _createAndTransferImage(self):
        # Define a numpy array of 3-coloured pixels with the shape of the
        # camera sensor
        self.imageRows = self.camera.sensorShape[0]
        self.imageCols = self.camera.sensorShape[1]
        self.image = np.empty((self.imageRows, self.imageCols, 3))
        self.image[:, :, 0] = 1
        self.image[:, :, 0] = 0
        self.image[:, :, 0] = 0

        # Transfer host (CPU) memory to device (GPU) memory
        # FIXME: Does this free the previous memory or no?
        self.imageGPU = gpuarray.to_gpu(self.image)

    def getImage(self):
        # Call the kernel raytracer
        self.rayTrace(
            # Image properties
            self.imageGPU,
            np.float64(self.imageRows),
            np.float64(self.imageCols),
            np.float64(self.camera.pixelWidth),
            np.float64(self.camera.pixelHeight),
            np.float64(self.camera.focalLength),

            # Camera constants
            np.float64(self.camera.r),
            np.float64(self.camera.theta),
            np.float64(self.camera.phi),
            np.float64(self.camera.beta),

            # Black hole constants
            np.float64(self.blackHole.a),
            np.float64(self.blackHole.b1),
            np.float64(self.blackHole.b2),

            # Kerr metric constants
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

        self.image = self.imageGPU.get()

        return(self.image)



if __name__ == '__main__':
    # Black hole spin
    spin = 0.00001

    # Camera position
    camR = 20
    camTheta = Pi/2
    camPhi = 0

    # Camera lens properties
    camFocalLength = 1.6
    camSensorShape = (1000, 1000)  # (Rows, Columns)
    camSensorSize = (2, 2)       # (Height, Width)

    # Create the black hole, the camera and the metric with the constants above
    blackHole = BlackHole(spin)
    camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                    camSensorSize)
    kerr = KerrMetric(camera, blackHole)

    # Set camera's speed (it needs the kerr metric constants)
    camera.setSpeed(kerr, blackHole)

    # Create the raytracer!
    rayTracer = RayTracer(camera, kerr, blackHole)
    test = rayTracer.getImage()
    plt.imshow(test, interpolation='nearest')
    plt.show()

    # # Define image parameters
    # imageRows = camSensorShape[0]
    # imageCols = camSensorShape[1]
    # image = np.empty((imageRows, imageCols, 3))
    #
    # def calculate_ray_parallel(pixel_pos):
    #     row, col = pixel_pos
    #     ray = camera.createRay(row, col, kerr, blackHole)
    #     # Compute pixel and store it in the image
    #     pixel = ray.traceRay(camera, blackHole)
    #     return pixel_pos, [pixel, pixel, pixel]
    #
    # # Raytracing!
    # pool = Pool(8)
    # conditions = [(x, y) for x in range(imageRows) for y in range(imageCols)]
    # results = pool.map(calculate_ray_parallel, conditions)
    # for pixel_pos, result in results:
    #     x, y = pixel_pos
    #     image[x, y] = result
    #
    # # Show image
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # strT = r'$\theta \in [' + str(camera.minTheta) + ', ' + str(camera.maxTheta) + ']; Length = ' + str(camera.maxTheta - camera.minTheta) + '$'
    # strP = r'$\phi \in [' + str(camera.minPhi) + ', ' + str(camera.maxPhi) + ']; Length = ' + str(camera.maxPhi - camera.minPhi) + '$'
    #
    # ax.annotate(strT, xy=(10, 30), backgroundcolor='white')
    # ax.annotate(strP, xy=(10, 60), backgroundcolor='white')
    # ax.annotate(r'$a = '+str(spin)+'$', xy=(10, 90), backgroundcolor='white')
    # ax.annotate(r'$d = '+str(camFocalLength)+'$', xy=(10, 120),
    #             backgroundcolor='white')
    # plt.imshow(image, interpolation='nearest')
    # plt.show()
