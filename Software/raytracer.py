import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import newton
from tqdm import tqdm
from numpy import sin, cos, arccos, arctan2, sqrt
from numpy import pi as Pi
from multiprocessing import Pool
import os
os.system("taskset -p 0xff %d" % os.getpid())

# Convention for pixel colors
CELESTIAL_SPHERE = 1
HORIZON = 0

# # -------- Functions appearing in the equations for a null geodesic --------
# # See (A.4)
# def P(r, theta, b, q):
#     return r**2. + A2 - A*b
#
#
# def R(r, theta, b, q):
#     return P(r, theta, b, q)**2. - delta(r, theta)*((b-A)**2. + q)
#
#
# def Z(r, theta, b, q):
#     return q - np.cos(theta)**2. * ((b**2./(np.sin(theta)**2.)) - A2)


# See (A.5)
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
    def __init__(self, r, theta, phi, focalLenght, sensorShape, sensorSize):
        # Define position
        self.r = r
        self.r2 = r**2
        self.theta = theta
        self.phi = phi

        # Define lens properties
        self.focalLenght = focalLenght

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

    def setSpeed(self, kerr, blackHole):
        # Retrieve blackhole's spin and some Kerr constants
        a = blackHole.a
        pomega = kerr.pomega
        omega = kerr.omega
        alpha = kerr.alpha

        # Define speed with equation (A.7)
        Omega = 1. / (a + self.r**(3./2.))
        self.beta = pomega * (Omega-omega) / alpha

    def createRay(self, row, col, kerr, blackHole):
        # Compute position of the point in cartesian coordinates.
        # We are basically transforming from pixel coordinates (with zero in
        # the center of the image) to cartesian oordinates in the camera's
        # reference frame (X axis is in the direction of imageCenter -
        # blackHoleCenter and positive when going away from the black hole, Y
        # axis is in the direction and sense of motion -horizontal axis in the
        # image-, positive when going to the left of the image and Z axis is
        # perpendicular to both of them -it follows the vertical axis in the
        # image-, positive when going up).
        # This method expects the (row,col) coordinates to be in a reference in
        # which the center of the image is the zero, X is positive to the right
        # and Y is positive going up.
        # Let's compute the pixel's position in the camera's reference frame,
        # multiplying the pixel coordinate for the widht and height of a single
        # pixel and naming the axes as explained before
        pixelY = - col * self.pixelWidth
        pixelZ = row * self.pixelHeight

        # Retrieve the focal length to ease the notation
        d = self.focalLenght

        # Now we have to compute the ray's direction in spherical coordinates
        # in the camera's reference frame following the pinhole camera model:
        # from the pixel's position in the camera's reference frame, we trace a
        # ray passing through the pinhole; then, we measure the angle between
        # this ray and the center axis (the line passing through the center of
        # the sensor and through the pinhole) and call it theta; finally, we
        # project the ray to the plane that contains both the center axis and
        # the horizontal axis (the equatorial plane), we measure the angle
        # between this projection and the center axis and call it phi.
        # The following conventions is now used: imagine that you place
        # yourself at the very center of the sensor, looking at the black hole;
        # then:
        #    - The angle theta comes from over your head (where theta = 0) and
        #    goes all the way down, in front of you (where theta = pi/2), until
        #    it reaches your feet (where theta = Pi).
        #    - The angle phi starts just at your back (where phi = 0), start
        #    turning to your right arm, where you first see it (phi = pi/2),
        #    goes right in front of you (where phi = pi), reaches your left arm
        #    (where phi = 3pi/2) and disappear behind your back again, until it
        #    reaches its original position.
        # Following this method, the spherical coordinates are computed as
        # follows:
        # rayTheta = (arctan2(sqrt(pixelX**2. + pixelY**2.), d) + Pi)/2.
        # rayPhi = arctan2(x, d) + Pi

        # Let's try another thing
        # P = (-1, pixelY, pixelZ)
        # F = (0, 0, 0) # Focal point is behind the image (in the positive X axis)
        #
        # FP = (-1, pixelY, pixelZ)

        # rayPhi = arctan2(pixelY, -d)
        # rayTheta = arccos(-pixelX / sqrt(d**2 + pixelX**2 + pixelY**2))

        r = sqrt(d**2 + pixelY**2 + pixelZ**2)
        rayTheta = arccos(pixelZ / r)
        rayPhi = arctan2(pixelY, -1)

        # rayTheta = arctan2(y, np.sqrt(D**2 - x**2))
        # rayPhi = arctan2(x, D)

        self.maxTheta = rayTheta if rayTheta > self.maxTheta else self.maxTheta
        self.maxPhi = rayPhi if rayPhi > self.maxPhi else self.maxPhi

        self.minTheta = rayTheta if rayTheta < self.minTheta else self.minTheta
        self.minPhi = rayPhi if rayPhi < self.minPhi else self.minPhi

        # We can now create and return our ray :)
        return Ray(rayTheta, rayPhi, self, kerr, blackHole)


# Dummy object for a ray (pixel->spherical transformation is done here)
class Ray:
    def __init__(self, theta, phi, camera, kerr, blackHole):
        # Set direction in the camera's reference frame
        self.theta = theta
        self.phi = phi

        # Compute all necessary information for the ray
        self._setNormal()
        self._setDirectionOfMotion(camera)
        self._setCanonicalMomenta(kerr)
        self._setConservedQuantities(camera, blackHole)

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

        # No radial turning points (see A.13 and A.14)
        if b1 < b < b2 and q < q0(r0, a):
            if(self.pR > 0.):
                return HORIZON
            else:
                return CELESTIAL_SPHERE
        # There are two radial turning points. See (v), (c)
        else:
            # Coefficientes of r^4, r^3, r^2, r^1 and r^0 from R(r)
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


if __name__ == '__main__':
    # Black hole spin
    spin = 0.999

    # Camera position
    camR = 20
    camTheta = Pi/2
    camPhi = 0

    # Camera lens properties
    camFocalLength = 0.1
    camSensorShape = (500, 500)  # (Rows, Columns)
    camSensorSize = (2, 2)       # (Height, Width)

    # Create the black hole, the camera and the metric with the constants above
    blackHole = BlackHole(spin)
    camera = Camera(camR, camTheta, camPhi, camFocalLength, camSensorShape,
                    camSensorSize)
    kerr = KerrMetric(camera, blackHole)

    # Set camera's speed (it needs the kerr metric constants)
    camera.setSpeed(kerr, blackHole)

    # Define image parameters
    imageRows = camSensorShape[0]
    imageCols = camSensorShape[1]
    image = np.empty((imageRows, imageCols, 3))

    def calculate_ray_parallel( pixel_pos ):
        row,col = pixel_pos
        ray = camera.createRay(row - imageRows/2, col - imageCols/2,
                                   kerr, blackHole)
        # Compute pixel and store it in the image
        pixel = ray.traceRay(camera, blackHole)
        return pixel_pos,[pixel, pixel, pixel]
    # Raytracing!

    pool = Pool(8)
    conditions = [(x,y) for x in range(imageRows) for y in range(imageCols)]
    results = pool.map(calculate_ray_parallel,conditions)
    for pixel_pos,result in results:
        x,y = pixel_pos
        image[x,y] = result

    # Show image

    fig = plt.figure()
    ax = fig.add_subplot(111)

    strT = r'$\theta \in$ [' + str(camera.minTheta) + ', ' + str(camera.maxTheta) + ']'
    strP = r'$\phi \in$ [' + str(camera.minPhi) + ', ' + str(camera.maxPhi) + ']'

    ax.annotate(strT, xy=(10, 25), backgroundcolor='white')
    ax.annotate(strP, xy=(10, 50), backgroundcolor='white')
    ax.annotate(r'$a = $'+str(spin), xy=(10, 75), backgroundcolor='white')
    plt.imshow(image, interpolation='nearest')
    plt.show()
