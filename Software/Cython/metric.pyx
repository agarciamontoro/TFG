#cython: language_level=3, boundscheck=False,cdivision=True

from libc.math cimport *
from libc.string cimport memcpy
cimport numpy as np
import numpy as np
cimport cython



################################################
##         PYTHON-ACCESIBLE FUNCTIONS         ##
################################################

# Notice that these functions HAVE overhead because they interface with python code and
# python objects will be constructed and unpacked each time the function is summoned.

cpdef np.ndarray[np.float64_t, ndim=2] metric(double r, double theta, double phi, double a):

    # Create output matrix

    cdef np.ndarray[np.float64_t, ndim=2] kerr_metric = np.zeros((4,4))

    # Calculate values

    calculate_metric(r, theta, phi, a , kerr_metric)

    return kerr_metric

cpdef np.ndarray[np.float64_t, ndim=2] inverse_metric(double r, double theta, double phi, double a):

    # Create output matrix

    cdef np.ndarray[np.float64_t, ndim=2] kerr_metric = np.zeros((4,4))

    # Calculate values

    calculate_inverse_metric(r, theta, phi, a , kerr_metric)

    return kerr_metric



################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a {this_file_name}.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.


cdef void calculate_inverse_metric(double r, double theta,
        double phi, double a, double [:,:] kerr_metric):

    cdef double r2 = r*r
    cdef double a2 = a*a
    cdef double sintheta2 = sin(theta) * sin(theta)

    cdef double ro = sqrt(r2 + a2 * cos(theta)**2)
    cdef double delta = r2 - 2*r + a2
    cdef double sigma = sqrt( (r2 + a2) * (r2 + a2) - a2 * delta * sintheta2)
    cdef double alpha = ro * sqrt(delta) / sigma
    cdef double omega = 2.0 * a * r / (sigma * sigma)
    cdef double pomega = sigma * sin(theta) / ro

    # Compute auxiliar products

    cdef double ro2 = ro * ro
    cdef double pomega2 = pomega*pomega
    cdef double omega2 = omega * omega
    # Construct the nonzero components of the metric

    kerr_metric[0,0] = - 1.0 / delta * ( r2 + a2 + a * pomega2 * omega)
    kerr_metric[1,1] = delta / ro2
    kerr_metric[2,2] = 1 / ro2
    kerr_metric[3,3] = ( delta - a2 * sintheta2 ) / (ro2 * delta * sintheta2 )
    kerr_metric[0,3] = - 2.0 * a * r / ro2 / delta
    kerr_metric[3,0] = kerr_metric[0,3]


cdef void calculate_metric(double r, double theta,
        double phi, double a, double [:,:] kerr_metric):

    # Calculate metric quantities 

    cdef double r2 = r*r
    cdef double a2 = a*a

    cdef double ro = sqrt(r2 + a2 * cos(theta)**2)
    cdef double delta = r2 - 2*r + a2
    cdef double sigma = sqrt( (r2 + a2) * (r2 + a2) - a2 * delta * sin(theta)*sin(theta))
    cdef double alpha = ro * sqrt(delta) / sigma
    cdef double omega = 2.0 * a * r / (sigma * sigma)
    cdef double pomega = sigma * sin(theta) / ro

    # Compute auxiliar products

    cdef double ro2 = ro * ro
    cdef double pomega2 = pomega*pomega
    cdef double omega2 = omega * omega
    # Construct the nonzero components of the metric

    kerr_metric[0,0] = - alpha * alpha + pomega2 * omega2
    kerr_metric[1,1] = ro2 / delta
    kerr_metric[2,2] = ro2
    kerr_metric[3,3] = pomega2
    kerr_metric[0,3] = - pomega2 * omega
    kerr_metric[3,0] = kerr_metric[0,3]

