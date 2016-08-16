from matplotlib import pyplot as plt
import numpy as np
import scipy.misc as smp

CELESTIAL_SPHERE = 1
HORIZON = 0

SPIN = 0.5
SPIN2 = SPIN**2
R1 = 2.*(1. + np.cos((2./3.)*np.arccos(-SPIN)))
R2 = 2.*(1. + np.cos((2./3.)*np.arccos(+SPIN)))
D = 0.01


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


def omegaH(r, theta):
    return sigma(r, theta) * np.sin(theta) / ro(r, theta)


# Equations for a null geodesic
def P(r, theta, b, q):
    return r**2. + SPIN2 - SPIN*b


def R(r, theta, b, q):
    return P(r, theta, b, q)**2. - delta(r, theta)*((b-SPIN)**2. + q)


def Z(r, theta, b, q):
    return q - np.cos(theta)**2. * ((b**2./(np.sin(theta)**2.)) - SPIN2)


def b0(r):
    # assert(R1 <= r <= R2)
    return - (r**3. - 3.*r**2. + SPIN2*r + SPIN2) / (SPIN*(r-1.))


def q0(r):
    # assert(R1 <= r <= R2)
    r3 = r**3.
    return - (r3*(r3 - 6.*r**2. + 9.*r - 4.*SPIN2)) / (SPIN2*(r-1.)**2.)

B1 = b0(R2)
B2 = b0(R1)

class Camera:
    def __init__(self, r, theta, phi):
        self.r = r
        self.theta = theta
        self.phi = phi

        Omega = 1. / (SPIN + r**(3./2.))
        self.beta = omegaH(r, theta)*(Omega-omega(r, theta))/alpha(r, theta)


class Ray:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # self.theta = np.arctan2(y, np.sqrt(D**2 - x**2))
        # self.phi = np.arctan2(x, D)

        self.theta = np.arctan2(np.sqrt(D**2.+x**2.), y)
        self.phi = np.arctan2(x, D)+np.pi


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

    # Ray's canonical momenta
    Ef = 1. / (alpha(cam.r, ray.theta) + omega(cam.r, ray.theta)*omegaH(cam.r, ray.theta)*nphi)

    pt = -1.
    pr = Ef * ro(cam.r, ray.theta) * nr / np.sqrt(delta(cam.r, ray.theta))
    ptheta = Ef * ro(cam.r, ray.theta) * ntheta
    pphi = Ef * omegaH(cam.r, ray.theta) * nphi

    # Ray's constants
    b = pphi
    q = b**2. + np.cos(ray.theta)**2. * (b**2.*(np.sin(ray.theta)**2.) - SPIN2)

    # print(B1, b, B2)

    # No radial turning points
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

        # Get maximum of the real roots of R(r) = 0
        roots = np.roots(coefs)

        rup = 0. if np.iscomplex(roots).all() else np.amax(roots[np.isreal(roots)])

        if cam.r < rup:
            return CELESTIAL_SPHERE
        else:
            return HORIZON

if __name__ == '__main__':
    W = H = 500
    camera = Camera(1000., 0., 0.)
    img = [[traceRay(camera, Ray(x, y)) for x in range(-W, W+1)] for y in range(-H, H+1)]
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()
