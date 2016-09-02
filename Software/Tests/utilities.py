from numpy import sqrt,sin,cos
import numpy as np
from pycuda import gpuarray


def collect_rays(self, xEnd=-30, numSteps=300):
    """
    This function is supposed to monkey-patch the RayTracer class in order to
    integrate and collect data for all the rays. In order to use this function
    the RayTracer initial conditions must be already prepared.

    All the collected data is stored in a numpy array of dimension (rows,cols,5,steps)
    that can be found in the "rayData" attribute.

    :param xEnd: Float
        The end point of the independent variable for the integrator
    :param numSteps: int
        The number of steps to be collected by the raytracer. This number represent also
        the number of times the kernel will be called.
    """
    stepSize = xEnd/numSteps

    # Initialize rayData with the initial position of the rays
    self.rayData = np.zeros((self.imageRows, self.imageCols,
                               5, numSteps+1))

    self.rayData[:, :, :, 0] = self.systemState[:, :, :]

    x = 0

    # Call the kernel numSteps times advancing the time variable
    # one stepSize increment each time and collect the systemState
    # data in the rayData array.

    for step in range(numSteps):
        # Solve the system
        self.callKernel(x, x + stepSize)

        # Advance the step and synchronise
        x += stepSize
        self.synchronise()

        # Get the data and store it for future plot
        self.rayData[:, :, :, step + 1] = self.systemState

def override_initial_conditions(self, r, cam_theta, cam_phi, theta_cs, phi_cs):
    """
    This function is supposed to monkey-patch the RayTracer class in order to
    provide a way to assign the same initial conditions to all the rays. This can
    be used for multiple pourposes as comparing the solved ray coordinates to another
    numerical integrator or test if all the rays are solved in the same manner.


    :param r: Float
        The initial value for the r coordinate of the camera.
    :param cam_theta:
        The initial value for the theta coordinate of the camera.
    :param cam_phi:
        The initial value for the phi coordinate of the camera.
    :param theta_cs:
        The initial inclination angle (theta) for the ray.
    :param phi_cs:
        The initial inclination angle (phi) for the ray.
    """
    # Calculate initial vector direction

    Nx = sin(theta_cs) * cos(phi_cs)
    Ny = sin(theta_cs) * sin(phi_cs)
    Nz = cos(theta_cs)

    # Convert the direction of motion to the FIDO's spherical orthonormal
    # basis. See (A.10)

    #TODO: Fix this mess.
    # IMPORTANT: This is not computed as in (A.10) because the MATHEMATICA DATA
    # has been generated without the aberration computation. Sorry for that!

    nR = Nx
    nTheta = Nz
    nPhi = Ny

    # Get canonical momenta

    ro = self.kerr.ro
    delta = self.kerr.delta
    pomega = self.kerr.pomega

    # Compute energy as measured by the FIDO. See (A.11)

    # TODO: Fix this mess
    # IMPORTANT: This is not computed as in (A.11) because the MATHEMATICA DATA
    # has been generated with this quantity as 1. Sorry for that!
    E = 1

    # Compute the canonical momenta. See (A.11)
    pR = E * ro * nR / sqrt(delta)
    pTheta = E * ro * nTheta
    pPhi = E * pomega * nPhi

    # Calculate the conserved quantities b and q.

    # Simplify notation
    theta = cam_theta
    a2 = self.blackHole.a2

    # Set conserved quantities. See (A.12)
    b = pPhi
    q = pTheta**2 + cos(theta)**2*(b**2 / sin(theta)**2 - a2)

    # Store the initial conditions in all the pixels of the systemState array.

    self.systemState[:,:,0] = r
    self.systemState[:,:,1] = cam_theta
    self.systemState[:,:,2] = cam_phi
    self.systemState[:,:,3] = pR
    self.systemState[:,:,4] = pTheta

    # Store the constants in all the pixels of the constant array.

    self.constants[:,:,0] = b
    self.constants[:,:,1] = q

    # Send the calculated data to the gpu overriding whatever was there before.

    self.systemStateGPU = gpuarray.to_gpu(self.systemState)
    self.constantsGPU = gpuarray.to_gpu(self.constants)

