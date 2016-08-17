import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import newton


from numpy import sin, cos, arccos, sqrt
from numpy import pi as Pi

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
    def __init__(self, blackHole, camera):
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
    def __init__(self, r, theta, phi, focalLenght, lensAngle):
        # Define position
        self.r = r
        self.r2 = r**2
        self.theta = theta
        self.phi = phi

        # Define lens properties
        self.focalLenght = focalLenght
        self.lensAngle = lensAngle

    def setSpeed(self, blackHole, kerr):
        # Retrieve blackhole's spin and some Kerr constants
        a = blackHole.a
        pomega = kerr.pomega
        omega = kerr.omega
        alpha = kerr.alpha

        # Define speed with equation (A.7)
        Omega = 1. / (a + self.r**(3./2.))
        self.beta = pomega * (Omega-omega) / alpha


# Dummy object for a ray (pixel->spherical transformation is done here)
class Ray:
    def __init__(self, x, y, camera, kerr):
        self.x = x
        self.y = y

        # Mío 1
        # self.theta = np.arctan2(y, np.sqrt(D**2 - x**2))
        # self.phi = np.arctan2(x, D)

        # # Pablo
        # self.theta = np.arctan2(np.sqrt(D**2.+x**2.), y)
        # self.phi = np.arctan2(x, D)

        # # Mío 2
        # self.theta = np.arctan2(np.sqrt(x**2. + y**2.), D) + np.pi
        # self.phi = np.arctan2(y, D) + np.pi/2

        # Compute spherical coordinates (theta, phi), in the camera's local
        # sky, for pixel (x, y)
        self.theta = x
        self.phi = y

        # Compute all necessary information for the ray
        self.setNormal()
        self.setDirectionOfMotion(camera)
        self.setCanonicalMomenta(kerr)
        self.setConservedQuantities(camera, blackHole)

    def setNormal(self):
        # Cartesian components of the unit vector N pointing in the direction
        # of the incoming ray
        self.Nx = sin(self.theta) * cos(self.phi)
        self.Ny = sin(self.theta) * sin(self.phi)
        self.Nz = cos(self.theta)

    def setDirectionOfMotion(self, camera):
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

    def setCanonicalMomenta(self, kerr):
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

        # Compute the canonical momenta See (A.11)
        self.pR = E * ro * nR / sqrt(delta)
        self.pTheta = E * ro * nTheta
        self.pPhi = E * pomega * nPhi

    def setConservedQuantities(self, camera, blackHole):
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

    def traceRay(self, blackHole, camera):
        # Simplify notation
        b = self.b
        q = self.q
        b1 = blackHole.b1
        b2 = blackHole.b2
        a = blackHole.a
        a2 = blackHole.a2

        # Compute r0 such that b0(r0) = b
        # fac = a2 + a*b - 3
        # cubicRoot = (sqrt(108*(fac**3) + (54-54*a2)**2) - 54*a2 + 54)**(1/3)
        # r0 = -(2**(1/3)*fac)/(cubicRoot) + cubicRoot/(3*(2**(1/3))) + 1


        # No radial turning points (see A.13 and A.14)
        if b1 < b < b2:
            fac = -9 + 3*a2 + 3*a*b
            innerSqrt = sqrt((54 - 54*a2)**2 + 4*(fac**3))
            cubicRoot = (54 - 54*a2 + innerSqrt)**(1/3)
            r0 = 1 - ((2**(1/3))*fac)/(3*cubicRoot) + cubicRoot/(3*(2**(1/3)))
            print(b, r0)

            if q < q0(r0, a):
                if(self.pR > 0.):
                    return HORIZON
                else:
                    return CELESTIAL_SPHERE
        # There are two radial turning points
        else:
            # # Get the maximum root of R(r) = 0. See (v), (c)
            # rUp1 = -a2 - P + a*b
            # rUp2 = -a2 + P + a*b
            # rUp = rUp1 if rUp1 > rUp2 else rUp2

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
                return CELESTIAL_SPHERE
            else:
                return HORIZON


if __name__ == '__main__':
    # Black hole spin
    spin = 0.5

    # Camera position
    camR = 20
    camTheta = Pi/2
    camPhi = Pi/3

    # Camera lens properties
    camFocalLength = 1
    camLensAngle = Pi/2

    # Create the black hole, the camera and the metric with the constants above
    blackHole = BlackHole(spin)
    camera = Camera(camR, camTheta, camPhi, camFocalLength, camLensAngle)
    kerr = KerrMetric(blackHole, camera)

    # Set camera's speed (it needs the kerr metric constants)
    camera.setSpeed(blackHole, kerr)

    # Define image parameters
    imageRows = 500
    imageCols = 500
    image = np.empty((imageRows, imageCols))

    # Variables to sweep the image
    arcLengthHoriz = camera.lensAngle
    arcLengthVert = (imageRows/imageCols) * arcLengthHoriz

    horizStep = arcLengthHoriz/imageCols
    vertStep = arcLengthVert/imageRows

    # Raytracing!
    for row in range(imageRows):
        # Define ray's theta
        rayTheta = Pi/2 - arcLengthVert/2 + row*vertStep

        for col in range(imageCols):
            # Define ray's phi
            rayPhi = Pi + arcLengthHoriz/2 - col*horizStep

            # Create actual ray
            ray = Ray(rayTheta, rayPhi, camera, kerr)

            # Compute pixel and store it in the image
            pixel = ray.traceRay(blackHole, camera)
            image[row, col] = pixel

    # Show image
    plt.imshow(image, cmap='Greys', interpolation='nearest')
    plt.show()
