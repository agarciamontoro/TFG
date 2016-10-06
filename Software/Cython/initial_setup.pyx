#cython: language_level=3, boundscheck=False,cdivision=True
from cython.parallel import prange
from libc.math cimport *
from libc.string cimport memcpy
cimport numpy as np
import numpy as np
cimport cython


################################################
##            GLOBAL DEFINITIONS              ##
################################################


################################################
##         PYTHON-ACCESIBLE FUNCTIONS         ##
################################################

# Notice that these functions HAVE overhead because they interface with python code and
# python objects will be constructed and unpacked each time the function is summoned.


################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a this_file_name.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # tuern off zerodivisioncheck
cdef void getCanonicalMomenta( double  rayTheta,  double  rayPhi,double* camBeta,
                          double *pR,  double *pTheta, double *pPhi,
                          double *ro, double *delta,
                          double *alpha, double *omega, double *pomega): 

    # **************************** SET NORMAL **************************** #
    # Cartesian components of the unit vector N pointing in the direction of
    # the incoming ray
    cdef double  Nx = sin(rayTheta) * cos(rayPhi)
    cdef double  Ny = sin(rayTheta) * sin(rayPhi)
    cdef double  Nz = cos(rayTheta)

    # ********************** SET DIRECTION OF MOTION ********************** #
    # Compute denominator, common to all the cartesian components
    cdef double  den = 1. - camBeta[0] * Ny

    # Compute factor common to nx and nz
    cdef double  fac = -sqrt(1. - camBeta[0]*camBeta[0])

    # Compute cartesian coordinates of the direction of motion. See(A.9)
    cdef double  nY = (-Ny + camBeta[0]) / den
    cdef double  nX = fac * Nx / den
    cdef double  nZ = fac * Nz / den

    # Convert the direction of motion to the FIDO's spherical orthonormal
    # basis. See (A.10)
    cdef double  nR = nX
    cdef double  nTheta = -nZ
    cdef double  nPhi = nY

    # *********************** SET CANONICAL MOMENTA *********************** #
    # Compute energy as measured by the FIDO. See (A.11)
    cdef double  E = 1. / (alpha[0] + omega[0] * pomega[0] * nPhi)

    # Set conserved energy to unity. See (A.11)
    # cdef double  pt = -1

    # Compute the canonical momenta. See (A.11)
    pR[0] = E * ro[0] * nR / sqrt(delta[0])
    pTheta[0] = E * ro[0] * nTheta
    pPhi[0] = E * pomega[0] * nPhi



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # tuern off zerodivisioncheck
cdef void getConservedQuantities( double  pTheta,  double  pPhi,
                             double  theta, double a, double* b,  double* q):
    # ********************* GET CONSERVED QUANTITIES ********************* #
    # Get conserved quantities. See (A.12)
    b[0] = pPhi

    cdef double a2 = a*a

    cdef double  sinT = sin(theta)
    cdef double  sinT2 = sinT*sinT

    cdef double  cosT = cos(theta)
    cdef double  cosT2 = cosT*cosT

    cdef double  pTheta2 = pTheta*pTheta
    cdef double  b2 = pPhi*pPhi

    q[0] = pTheta2 + cosT2*((b2/sinT2) - a2)


# void setInitialConditions( double * globalInitCond, double * globalConstants,
#                            double  imageRows,  double  imageCols,
#                            double  pixelWidth, double  pixelHeight,
#                            double  r, double theta, double phi):
#     # Calculate common functions

#     cdef double r2 = r*r
#     cdef double a2 = a*a

#     cdef double ro = sqrt(r2 + a2 * cos(theta)**2)
#     cdef double delta = r2 - 2*r + a2
#     cdef double sigma = sqrt((r2 + a2)**2 - a2 * delta * sin(theta)**2)
#     cdef double pomega = sigma * sin(theta) / ro
#     cdef double omega = 2 * a * r / ( sigma * sigma )
#     cdef double alpha = ro *  sqrt(delta) / sigma

#     for i in prange(imageRows, nogil=True):
#         for j in range(imageCols):
#             # Retrieve the id of the block in the grid
#             cdef int blockId =  col  + row  * imageCols

#             # Pointer for the initial conditions of this ray (block)
#             cdef double * initCond = globalInitCond + blockId*SYSTEM_SIZE

#             # Pointer for the constants of this ray (block)
#             cdef double * constants = globalConstants + blockId*2

#             # Compute pixel position in the physical space
#             cdef double  x = - (col + 0.5 - imageCols/2) * pixelWidth
#             cdef double  y = (row + 0.5 - imageRows/2) * pixelHeight

#             # Compute direction of the incoming ray in the camera's reference
#             # frame
#             cdef double  rayPhi = Pi + atan(x / __d)
#             cdef double  rayTheta = Pi/2 + atan(y / sqrt(__d*__d + x*x))

#             # Compute canonical momenta of the ray and the conserved quantites
#             # b and q
#             cdef double  pR, pTheta, pPhi, b, q
#             getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi)
#             getConservedQuantities(pTheta, pPhi, &b, &q)

#             # Save ray's initial conditions
#             initCond[0] = r
#             initCond[1] = theta
#             initCond[2] = phi 
#             initCond[3] = pR
#             initCond[4] = pTheta

#             # Save ray's constants
#             constants[0] = b
#             constants[1] = q
