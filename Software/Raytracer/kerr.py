import os
from numpy import sin, cos, arccos, sqrt

# Set directories for correct handling of paths
selfDir = os.path.dirname(os.path.abspath(__file__))
softwareDir = os.path.abspath(os.path.join(selfDir, os.pardir))


class BlackHole:
    def __init__(self, spin, innerDiskRadius=9, outerDiskRadius=20):
        # Define spin and its square
        self.a = spin
        self.a2 = spin**2

        # Interval over the radius of trapped photons' orbits run. See (A.6)
        self.r1 = 2.*(1. + cos((2./3.)*arccos(-self.a)))
        self.r2 = 2.*(1. + cos((2./3.)*arccos(+self.a)))

        # Necessary constants for the origin algorithm. See (A.13)
        self.b1 = self._b0(self.r2)
        self.b2 = self._b0(self.r1)

        # Horizon radius
        self.horizonRadius = (2 + sqrt(4 - 4*self.a2)) / 2

        # Disk inner and outer radius
        self.innerDiskRadius = innerDiskRadius
        self.outerDiskRadius = outerDiskRadius

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
