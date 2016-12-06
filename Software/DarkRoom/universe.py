from Utils.attr_dict import AttrDict

from numpy import cos, arccos, sqrt


class Universe:
    def __init__(self, spin=0.999, innerDiskRadius=9, outerDiskRadius=20):
        self._spin = spin
        self.spinSquared = spin*spin
        self.accretionDisk = AttrDict(innerRadius=innerDiskRadius,
                                      outerRadius=outerDiskRadius)

        self.cameras = set()

        self.update()

    def update(self):
        # Compute radius of trapped photons' orbits
        self.r1, self.r2 = self.computeTrappedOrbits()

        # Compute necessary constants for the origin algorithm. See (A.13)
        self.b1, self.b2 = self.computeBConstants()

        # Compute horizon radius
        self.horizonRadius = self.computeHorizonRadius()

        # Propagate the changes to all cameras
        for camera in self.cameras:
            camera.update()

    def computeTrappedOrbits(self):
        # Interval over the radius of trapped photons' orbits run. See (A.6)
        r1 = 2.*(1. + cos((2./3.)*arccos(-self.spin)))
        r2 = 2.*(1. + cos((2./3.)*arccos(+self.spin)))

        return r1, r2

    def _b0(self, r):
        a = self.spin
        a2 = self.spinSquared

        return - (r**3. - 3.*(r**2.) + a2*r + a2) / (a*(r-1.))

    def computeBConstants(self):
        # Necessary constants for the origin algorithm. See (A.13)
        b1 = self._b0(self.r2)
        b2 = self._b0(self.r1)

        return b1, b2

    def computeHorizonRadius(self):
        return 1 + sqrt(1 - self.spinSquared)

    @property
    def spin(self):
        return self._spin

    @spin.setter
    def spin(self, value):
        # Set new spin value and its square
        self._spin = value
        self.spinSquared = value*value

        # Update Universe properties
        self.update()
