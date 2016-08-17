import numpy as np
from matplotlib import pyplot as plt

# Convention for pixel colors
CELESTIAL_SPHERE = 1
HORIZON = 0

# Model constants
SPIN = 0.5
SPIN2 = SPIN**2
R1 = 2.*(1. + np.cos((2./3.)*np.arccos(-SPIN)))  # See (A.6)
R2 = 2.*(1. + np.cos((2./3.)*np.arccos(+SPIN)))  # See (A.6)
D = 5  # Focal length


# -------- (A.1) and (A.2) definitions --------
def ro(r, theta):
    return np.sqrt(r**2. + SPIN2*np.cos(theta)**2.)


def delta(r, theta):
    return r**2. - 2.*r + SPIN2


def sigma(r, theta):
    return np.sqrt((r**2.+SPIN2)**2.-SPIN2*delta(r, theta)*np.sin(theta)**2.)


def alpha(r, theta):
    return ro(r, theta) * np.sqrt(delta(r, theta)) / sigma(r, theta)


def omega(r, theta):
    return 2. * SPIN * r / (sigma(r, theta)**2.)


# H for hat
def omegaH(r, theta):
    return sigma(r, theta) * np.sin(theta) / ro(r, theta)


# -------- Functions appearing in the equations for a null geodesic --------
# See (A.4)
def P(r, theta, b, q):
    return r**2. + SPIN2 - SPIN*b


def R(r, theta, b, q):
    return P(r, theta, b, q)**2. - delta(r, theta)*((b-SPIN)**2. + q)


def Z(r, theta, b, q):
    return q - np.cos(theta)**2. * ((b**2./(np.sin(theta)**2.)) - SPIN2)


# See (A.5)
def b0(r):
    return - (r**3. - 3.*r**2. + SPIN2*r + SPIN2) / (SPIN*(r-1.))


def q0(r):
    r3 = r**3.
    return - (r3*(r3 - 6.*r**2. + 9.*r - 4.*SPIN2)) / (SPIN2*(r-1.)**2.)

# Necessary constants (see A.13)
B1 = b0(R2)
B2 = b0(R1)


# Dummy object for the camera (computation of the speed is done here)
class Camera:
    def __init__(self, r, theta, phi):
        # Define position
        self.r = r
        self.theta = theta
        self.phi = phi

        # Define speed
        Omega = 1. / (SPIN + r**(3./2.))
        self.beta = omegaH(r, theta)*(Omega-omega(r, theta))/alpha(r, theta)


# Dummy object for a ray (pixel->spherical transformation is done here)
class Ray:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Mío 1
        # self.theta = np.arctan2(y, np.sqrt(D**2 - x**2))
        # self.phi = np.arctan2(x, D)

        # # Pablo
        # self.theta = np.arctan2(np.sqrt(D**2.+x**2.), y)
        # self.phi = np.arctan2(x, D)

        # Mío 2
        self.theta = np.arctan2(np.sqrt(x**2. + y**2.), D)
        self.phi = np.arctan2(y, D) + np.pi/4


def traceRay(cam, ray):
    # Cartesian components of the unit vector N pointing in the direction of
    # the incoming ray
    Nx = np.sin(ray.theta) * np.cos(ray.phi)
    Ny = np.sin(ray.theta) * np.sin(ray.phi)
    Nz = np.cos(ray.theta)

    # Cartesian components ot the direction of motion of the incoming ray as
    # measured by the FIDO
    den = 1. - cam.beta * Ny
    fac = -np.sqrt(1. - cam.beta**2.)

    ny = -Ny + cam.beta / den
    nx = fac * Nx / den
    nz = fac * Nz / den

    # Spherical components on the FIDO's spherical orthonormal basis
    nr = nx
    ntheta = -nz
    nphi = ny

    # Compute energy
    Ef = 1. / (alpha(cam.r, ray.theta) + omega(cam.r, ray.theta)*omegaH(cam.r, ray.theta)*nphi)

    # Ray's canonical momenta
    pt = -1.
    pr = Ef * ro(cam.r, ray.theta) * nr / np.sqrt(delta(cam.r, ray.theta))
    ptheta = Ef * ro(cam.r, ray.theta) * ntheta
    pphi = Ef * omegaH(cam.r, ray.theta) * nphi

    # Ray's constants
    b = pphi
    q = b**2. + np.cos(ray.theta)**2. * (b**2.*(np.sin(ray.theta)**2.) - SPIN2)

    # No radial turning points (see A.13 and A.14)
    if B1 < b < B2 and q < q0(b):
        if(pr > 0.):
            return HORIZON
        else:
            return CELESTIAL_SPHERE
    # There are two radial turning points
    else:
        # Coefficientes of r^4, r^3, r^2, r^1 and r^0 of R(r)
        coefs = [1.,
                 0.,
                 -q - b**2. + SPIN2,
                 2.*q + 2.*(b**2.) - 4.*SPIN*b + 2.*SPIN2,
                 -SPIN2*q]

        # Get the roots of R(r) = 0
        roots = np.roots(coefs)

        # If there are real roots, get the maximum of them; in any other case,
        # set rup to -inf
        realIndices = np.isreal(roots)
        rup = np.amax(roots[realIndices]) if realIndices.any() else -np.inf

        # See (v), case (c)
        if cam.r < rup:
            return CELESTIAL_SPHERE
        else:
            return HORIZON

if __name__ == '__main__':
    # Size of the CCD sensor
    W = H = 1

    # Number of pixels in each side of the CCD
    numPixels = 250

    # Get the pixels location
    rangeX = np.linspace(-W, W, numPixels)
    rangeY = np.linspace(-H, H, numPixels)

    # Define camera position
    distance = 500
    camera = Camera(distance, 0., 0.)

    # Figure initialisation
    fig = plt.figure()

    # Compute each pixel value, tracing the ray from the pixel
    img = [[traceRay(camera, Ray(x, y)) for x in rangeX] for y in rangeY]

    # Show image
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()
