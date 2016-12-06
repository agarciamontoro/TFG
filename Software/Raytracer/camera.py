from .universe import universe
from .raytracer import RayTracer
from .geodesics_congruence import Congruence, CongruenceSnapshot

from Utils.attr_dict import AttrDict

import numpy as np
from numpy import sin, cos, sqrt


class Camera:
    """Pinhole camera placed near a Kerr black hole.

    This class contains the necessary data to define a camera that is located
    on the coordinate system of a Kerr black hole.

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

        TODO: Add a parameter to show where the camera is looking at.

        Args:
            r (double): Distance to the coordinate origin; i.e., distance to
                the black hole centre.
            theta (double): Inclination of the camera with respect to the black
                hole.
            phi (double): Azimuth of the camera with respect to the black hole.
            focalLength (double): Distance between the focal point (where every
                row that reaces the camera has to pass through) and the focal
                plane (where the actual sensor/film is placed).
            sensorShape (tuple): 2-tuple that defines the number of pixels of
                the sensor in the following way: `(Number of rows, Number of
                columns)`.
            sensorSize (tuple): 2-tuple that defines the physical dimensions of
                the sensor in the following way: `(Height, Width)`.
        """

        # Define position
        self._r = r
        self.r2 = r*r
        self._theta = theta
        self.phi = phi

        # Define lens properties
        self.focalLength = focalLength

        # Define sensor properties
        self._sensorShape = sensorShape
        self._sensorSize = sensorSize

        # Compute the width and height of a pixel on physical units
        self.pixelWidth, self.pixelHeight = self.computePixelSize()

        # Compute every property for the camera: metric, speed and engine
        self.update()

        # Add camera to the Universe
        universe.cameras.add(self)

    def __del__(self):
        # Remove camera from the Universe
        universe.cameras.remove(self)

    def update(self):
        # Compute the value of the metric on the camera position
        self.metric = self.computeMetricValue()

        # Compute the speed
        self.speed = self.computeSpeed()

        # Add the raytracer engine
        self.engine = RayTracer(self)

    def computePixelSize(self):
        """Compute the width and height of a pixel, taking into account the
        physical measure of the sensor (sensorSize) and the number of pixels
        per row and per column on the sensor (sensorShape)
        """

        # Shorten long named variables to ease the notation
        H, W = self.sensorSize[0], self.sensorSize[1]
        rows, cols = self.sensorShape[0], self.sensorShape[1]

        # Compute width and height of a pixel
        pixelWidth = np.float(W) / np.float(cols)
        pixelHeight = np.float(H) / np.float(rows)

        return pixelWidth, pixelHeight

    def computeMetricValue(self):
        # Shorten long named variables to ease the notation
        a = universe.spin
        a2 = universe.spinSquared
        r = self.r
        r2 = self.r2
        theta = self.theta

        # Compute the constants described between (A.1) and (A.2)
        ro = sqrt(r2 + a2 * cos(theta)**2)
        delta = r2 - 2*r + a2
        sigma = sqrt((r2 + a2)**2 - a2 * delta * sin(theta)**2)
        alpha = ro * sqrt(delta) / sigma
        omega = 2 * a * r / (sigma**2)


        # Wut? pomega? See https://en.wikipedia.org/wiki/Pi_(letter)#Variant_pi
        pomega = sigma * sin(theta) / ro

        return AttrDict(ro=ro, delta=delta, sigma=sigma, alpha=alpha,
                        omega=omega, pomega=pomega)

    def computeSpeed(self):
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
        a = universe.spin
        r = self.r
        pomega = self.metric.pomega
        omega = self.metric.omega
        alpha = self.metric.alpha

        # Define speed with equation (A.7)
        Omega = 1. / (a + r**(3./2.))
        beta = pomega * (Omega-omega) / alpha

        # FIXME: This is being forced to zero only for testing purposes.
        # Remove this line if you want some real fancy images.
        beta = 0

        return beta

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, newValue):
        # Set r coordinate and its square
        self._r = newValue
        self.r2 = self.r * self.r

        # Compute again the value of the metric on the camera position
        self.update()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, newValue):
        # Set theta coordinate
        self._theta = newValue

        # Compute again the value of the metric on the camera position
        self.update()

    @property
    def sensorShape(self):
        return self._sensorShape

    @sensorShape.setter
    def sensorShape(self, newValue):
        # Set sensor shape
        self._sensorShape = newValue

        # Update the computation on pixel width and height
        self.pixelWidth, self.pixelHeight = self.computePixelSize()

    @property
    def sensorSize(self):
        return self._sensorSize

    @sensorSize.setter
    def sensorSize(self, newValue):
        # Set sensor shape
        self._sensorSize = newValue

        # Update the computation on pixel width and height
        self.pixelWidth, self.pixelHeight = self.computePixelSize()

    def shoot(self, finalTime=-150):
        raysStatus, raysCoordinates = self.engine.rayTrace(finalTime)
        return CongruenceSnapshot(raysStatus, raysCoordinates)

    def slicedShoot(self, finalTime=-150, slicesNum=100):
        raysStatus, raysCoordinates = self.engine.slicedRayTrace(finalTime,
                                                                 slicesNum)
        return Congruence(raysStatus, raysCoordinates)
