from libc.math cimport *
from libc.string cimport memcpy
cimport numpy as np
import numpy as np
cimport cython


################################################
##            GLOBAL DEFINITIONS              ##
################################################

####  Butcher's tableau coefficients   ####

# These coefficients are needed for the RK45 Solver to work. 
# When calculating the different samples for the derivative, each
# sample is weighted differently according to some coefficients: These
# numbers the weighting coefficients. Each Runge-Kutta method has its
# coefficients, which in turns define the method completely and these
# are the coefficients of Dopri5, an adaptative RK45 schema.

# For more information on this matter:

# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods


cdef double A21 = (1./5.)

cdef double A31 = (3./40.)
cdef double A32 = (9./40.)

cdef double A41 = (44./45.)
cdef double A42 = (- 56./15.)
cdef double A43 = (32./9.)

cdef double A51 = (19372./6561.)
cdef double A52 = (- 25360./2187.)
cdef double A53 = (64448./6561.)
cdef double A54 = (- 212./729.)

cdef double A61 = (9017./3168.)
cdef double A62 = (- 355./33.)
cdef double A63 = (46732./5247.)
cdef double A64 = (49./176.)
cdef double A65 = (- 5103./18656.)

cdef double A71 = (35./384.)
cdef double A72 = (0)
cdef double A73 = (500./1113.)
cdef double A74 = (125./192.)
cdef double A75 = (- 2187./6784.)
cdef double A76 = (11./84.)

cdef double C2 = (1./5.)
cdef double C3 = (3./10.)
cdef double C4 = (4./5.)
cdef double C5 = (8./9.)
cdef double C6 = (1)
cdef double C7 = (1)

cdef double E1 = (71./57600.)
cdef double E2 = (0)
cdef double E3 = (- 71./16695.)
cdef double E4 = (71./1920.)
cdef double E5 = (- 17253./339200.)
cdef double E6 = (22./525.)
cdef double E7 = (- 1./40.)


#### System Size ####

# We are not constructing a general-pourpose integrator, instead we knoe that we
# want to integrate Kerr's geodesics. The system of differential equations for
# our version of the geodesic equations ( our version <-> the Hamiltonian we are
# using ) has 5 equations: 3 for the coordinates and 2 for the momenta because
# the third momenta equation vanishes explicitly. 

cdef int SYSTEM_SIZE = 5


################################################
##         PYTHON-ACCESIBLE FUNCTIONS         ##
################################################

# Notice that these functions HAVE overhead because they interface with python code and
# python objects will be constructed and unpacked each time the function is summoned.

### Integrate ray ###

# This functions constructs the initial conditions for a null geodesic and integrate
# this light ray. Its here mainly for testing the integrator against Mathematica's NDSOLVE.

cpdef np.ndarray[np.float64_t, ndim=2] integrate_ray(double r, double cam_theta,
                                                     double cam_phi, double theta_cs,
                                                     double phi_cs, double a, n_steps):

    # Simplify notation
    theta = cam_theta
    a2 = a*a
    r2 = r*r

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

    ro = sqrt(r2 + a2 * cos(theta)**2)
    delta = r2 - 2*r + a2
    sigma = sqrt((r2 + a2)**2 - a2 * delta * sin(theta)**2)
    pomega = sigma * sin(theta) / ro

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

    # Set conserved quantities. See (A.12)
    b = pPhi
    q = pTheta**2 + cos(theta)**2*(b**2 / sin(theta)**2 - a2)

    # Store the initial conditions in all the pixels of the systemState array.


    x0 = 0.0
    xend = -30.0
    result = np.zeros((n_steps+1,5))
    init = np.array([r, cam_theta, cam_phi, pR, pTheta])
    data = np.array([b,q,a])
    
    Solver(x0, xend, n_steps, init, data, result)
 
    return result

################################################
##                C FUNCTIONS                 ##
################################################

# This functios will be (hopefully) compliled in pure C and they will not have any Python overhead
# of any kind. This means that all the loops and math operations are free of the python-lag.

# ¡IMPORTANT!
# Please, check using cython -a {this_file_name}.pyx that these functions do not have python-related code,
# which is indicated by yellow lines in the html output.



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # tuern off zerodivisioncheck
cdef void Solver(double x, double xend, int n_steps, 
                 double [:] initCond,double[:] data,
                 double [:,:] result):
    """
    This function acts as an interface with the RK45 Solver. Its pourpose is avoid the
    overhead of calling the Runge-Kutta solver multiple times from python, which creates
    and unpacks a lot of python objects. The caller must pass a result buffer ( a python
    memoryview) and this function will populate the buffer with the result of successive
    calls to the Runge-Kutta solver.

    :param x: double
        The initial point of the independent variable for the integration
    :param xend: double
        The ending point of the independent variable for the integration.
    :param n_steps: int
        The number of steps to make when performing the integration.
        ¡IMPORTANT! This number has to be larger than the dimension of the `result` buffer.
    :param initCond: memoryview
        The initial conditions at the point x.
        ¡IMPORTANT! The content of this memoryview will be overwritten by the solver.
    :param data: memoryview
        Aditional data needed for computing the right hand side of the equations to integrate.
    :param result: memoryview (2D)
        A buffer to store the result of the integration in each RK45 step.
        ¡IMPORTANT! The first dimension of the buffer must be larger or equal than n_steps.
    """
    cdef double hOrig = 0.01
    cdef double globalFacold = 1.0e-4
    cdef double step = (xend - x) / n_steps
    cdef int current_step
    
    
    # Store initial step conditions

    result[0,0] = initCond[0]
    result[0,1] = initCond[1]
    result[0,2] = initCond[2]
    result[0,3] = initCond[3]
    result[0,4] = initCond[4]


    for current_step in range(n_steps):
        SolverRK45( initCond, &x, x+step, &hOrig, 0.1, data, &globalFacold)
        
        # Store the result of the integration in the buffer

        result[current_step + 1,0] = initCond[0]
        result[current_step + 1,1] = initCond[1]
        result[current_step + 1,2] = initCond[2]
        result[current_step + 1,3] = initCond[3]
        result[current_step + 1,4] = initCond[4]







@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # tuern off zerodivisioncheck
cdef void KerrGeodesicEquations(double* y, double* f,double [:] data):
    """
    This function computes the right hand side of the Kerr geodesic equations described
    in http://arxiv.org/abs/1502.03808.

    All the computations are highly optimized to avoid calculating the same term twice.

    :param y: pointer to double
        The current coordinate values to compute the equations. ( Will not be modified)
        The array must follow the following convention for the variables:
            0 -> r
            1 -> theta
            2 -> phi
            3 -> p_r
            4 -> p_theta

    :param f: pointer to double
        The place where the values of the equations will be stored.
        The array will follow this convention:
            0 -> d(r)/dt
            1 -> d(theta)/dt
            2 -> d(phi)/dt
            3 -> d(p_r)/dt
            4 -> d(p_theta)/dt

            Where t is the independent variable (propper time).

    :param data: memoryview (acting as pointer to double)
        Aditional data needed for the computation. Explicitly:

            0 -> b ( Angular momentum)
            1 -> q ( Carter's constant)
            2 -> a ( Black Hole spin)
    """
    # Variables to hold the position of the ray, its momenta and related
    # operations between them and the constant a, which is the spin of the
    # black hole.
    cdef double r, r2, twor, theta, pR, pR2, pTheta, pTheta2, b, twob, b2, q, bMinusA,a, a2

    # Variables to hold the sine and cosine of theta, along with some
    # operations with them
    cdef double sinT, cosT, sinT2, sinT2Inv, cosT2

    # Variables to hold the value of the functions P, R, Delta (which is
    # called D), Theta (which is called Z) and rho, along with some operations
    # involving these values.
    cdef double P, R, D, Dinv, Z, DZplusR, rho2Inv, twoRho2Inv, rho4Inv

    # Retrieval of the input data (position of the ray, momenta and
    # constants).
    r = y[0]
    theta = y[1]
    pR = y[3]
    pTheta = y[4]

    # Computation of the square of r, widely used in the computations.
    r2 = r*r

    # Sine and cosine of theta, as well as their squares and inverses.
    sinT = sin(theta)
    cosT = cos(theta)
    sinT2 = sinT*sinT
    sinT2Inv = 1/sinT2
    cosT2 = cosT*cosT

    # Retrieval of the constants data: b and q, along with the computation of
    # the square of b and the number b - a, repeateadly used throughout the
    # computation
    b = data[0]
    q = data[1]
    a = data[2]
    a2 = a*a
    
    b2 = b*b
    bMinusA = b - a

    # Commonly used variables: R, D, Theta (that is called Z) and
    # rho (and its square and cube).
    D = r2 - 2*r + a2
    Dinv = 1/D

    P = r2 - a * bMinusA
    R = P*P - D*(bMinusA*bMinusA + q)

    Z = q - cosT2*(b2*sinT2Inv - a2)

    rho2Inv = 1/(r2 + a2*cosT2)
    twoRho2Inv = rho2Inv/2
    rho4Inv = rho2Inv*rho2Inv

    # Squares of the momenta components
    pR2 = pR*pR
    pTheta2 = pTheta*pTheta

    # Double b and double r, that's it! :)
    twob = 2*b
    twor = 2*r

    # Declaration of variables used in the actual computation: dR, dZ, dRho
    # and dD will store the derivatives of the corresponding functions (with
    # respect to the corresponding variable in each thread). The sumX values
    # are used as intermediate steps in the final computations, in order to
    # ease notation.
    cdef double dR, dZ, dRhoTimesRho, dD, sum1, sum2, sum3, sum4, sum5, sum6

    # *********************** EQUATION 1 *********************** //
    f[0] = D * pR * rho2Inv

    # *********************** EQUATION 2 *********************** //
    f[1] = pTheta * rho2Inv

    # *********************** EQUATION 3 *********************** //
    # Derivatives with respect to b
    dR = 4*bMinusA*r - twob*r2
    dZ = - twob * cosT2 * sinT2Inv

    f[2] = - (dR + D*dZ)*Dinv*twoRho2Inv

    # *********************** EQUATION 4 *********************** //
    # Derivatives with respect to r
    dD = twor - 2
    dR = 2*twor*(r2 - a*bMinusA) - (q + bMinusA*bMinusA)*(twor - 2)

    DZplusR = D*Z + R

    sum1 = + pTheta2
    sum2 = + D*pR2
    sum3 = - (DZplusR * Dinv)
    sum4 = - (dD*pR2)
    sum5 = + (dD*Z + dR) * Dinv
    sum6 = - (dD*DZplusR * Dinv * Dinv)

    f[3] = r*(sum1 + sum2 + sum3)*rho4Inv + (sum4 + sum5 + sum6)*twoRho2Inv

    # *********************** EQUATION 5 *********************** //
    # Derivatives with respect to theta (called z here)
    dRhoTimesRho = - a2*cosT*sinT

    cdef double cosT3 = cosT2*cosT
    cdef double sinT3 = sinT2*sinT

    dZ = 2*cosT*((b2*sinT2Inv) - a2)*sinT + (2*b2*cosT3)/(sinT3)

    sum1 = + pTheta2
    sum2 = + D*pR2
    sum3 = - DZplusR * Dinv
    sum4 = + dZ * twoRho2Inv

    f[4] = dRhoTimesRho*(sum1 + sum2 + sum3)*rho4Inv + sum4







@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)    # tuern off zerodivisioncheck
cdef int SolverRK45( double [:] initCond, double* globalX0, double xend,
                     double* hOrig   , double hmax, double [:] data,
                     double* globalFacold,
                     double rtoli   = 1e-06,
                     double atoli   = 1e-12,
                     double safe    = 0.9,
                     double beta    = 0.04,
                     double uround  = 2.3e-16,
                     double fac1    = 0.2,
                     double fac2    = 10.0  ):

    """
     /**
      * Applies the DOPRI5 algorithm over the system defined in the KerrGeodesicEquations
      * function, using the initial conditions specified in InitCond,
      * and returning the solution found at xend.
      * @param[in,out]  Real*  globalX0     Start of the integration interval
      *                        [x_0, x_{end}]. At the output, this variable is set
      *                        to the final time the solver reached.
      * @param[in]      Real   xend         End of the integration interval
      *                        [x_0, x_{end}].
      * @param[in,out]  Real*  initCond     Device pointer to a serialized matrix of
      *                        initial conditions; i.e., given a 2D matrix of R rows
      *                        and C columns, where every entry is an n-tuple of
      *                        initial conditions (y_0[0], y_0[1], ..., y_0[n-1]),
      *                        the vector pointed by devInitCond contains R*C*n
      *                        serialized entries, starting with the first row from
      *                        left to right, then the second one in the same order
      *                        and so on.
      *                        The elements of vector pointed by initCond are
      *                        replaced with the new computed values at the end of
      *                        the algorithm; please, make sure you will not need
      *                        them after calling this procedure.
      * @param[in,out]  Real*  hOrig        Step size. This code controls
      *                        automatically the step size, but this value is taken
      *                        as a test for the first try; furthermore, the method
      *                        returns the last computed value of h to let the user
      *                        know the final state of the solver.
      * @param[in]      Real   hmax         Value of the maximum step size allowed,
      *                        usually defined as x_{end} - x_0, as we do not to
      *                        exceed x_{end} in one iteration.
      * @param[in]      Real*  data         Device pointer to a serialized matrix of
      *                        additional data to be passed to computeComonent;
      *                        currently, this is used to pass the constants b and q
      *                        of each ray to the KerrGeodesicEquations method.
      * @param[out]     int*   iterations   Output variable to know how many
      *                        iterations were spent in the computation
      * @param[in,out]  float* globalFacold Input and output variable, used as a
      *                        first value for facold and to let the caller know the
      *                        final value of facold.
      */
    """

    ################################
    ##### Configuration vars #######
    ################################

    cdef double safeInv = 1.0 / safe
    cdef double fac1_inverse = 1.0 / fac1
    cdef double fac2_inverse = 1.0 / fac2
    
    #################################
    #####  Variable definitions #####
    #################################

    # Declare a counter for the loops, in order not to declare it multiple
    # times :)

    cdef int i
    
    # Loop variable to manage the automatic step size detection.
    
    cdef double hnew
    
    # Retrieve the value of h and the value of x0
    cdef double h = hOrig[0]
    cdef double x0 = globalX0[0]

    # Check the direction of the integration: to the future or to the past
    # and get the absolute value of the maximum step size.

    cdef double integrationDirection = +1. if xend - x0 > 0. else -1.
    hmax = abs(hmax)

    cdef size_t sizeBytes = sizeof(double)*SYSTEM_SIZE
    

    # Copy initial conditions to initial array

    cdef double y0[5] # TODO Esto no es un poco movida en el codigo original?
    
    for i in range(5): # TODO: SYSTEM_SIZE
        y0[i] = initCond[i]


    # Auxiliar arrays to store the intermediate K1, ..., K7 computations
    # TODO: This 2 is SYSTEM_SIZE!!
    cdef double k1[5]
    cdef double k2[5]
    cdef double k3[5]
    cdef double k4[5]
    cdef double k5[5]
    cdef double k6[5]
    cdef double k7[5]

    # Auxiliar array to store the intermediate calls to the
    # KerrGeodesicEquations function

    cdef double y1[5]  # TODO: SYSTEM_SIZE

    # Auxiliary variables used to compute the errors at each step.
    
    cdef float sqr                 # Scaled differences in each eq.
    cdef float errors[5]           # TODO: SYSTEM_SIZE Local error of each eq.
    cdef float err = 0             # Global error of the step
    cdef float sk                  # Scale based on the tolerances

    # Initial values for the step size automatic prediction variables.
    # They are basically factors to maintain the new step size in known
    # bounds, but you can see the corresponding chunk of code far below to
    # know more about the puropose of each of these variables.

    cdef float facold = globalFacold[0]
    cdef float expo1 = 0.2 - beta * 0.75
    cdef float fac11, fac

    # Loop variables initialisation. The current step is repeated when
    # `reject` is set to true, event that happens when the global error
    # estimation exceeds 1.

    cdef int reject = False  # TODO: Avisar a alejandro de que esto esta como double

    cdef float horizonRadius = 2.0
    cdef int last = False


    while x0 > xend:

        # Check that the step size is not too small and that the horizon is
        # not too near. Although the last condition belongs to the raytracer
        # logic, it HAS to be checked here.
        
        if (0.1 * abs(h) <= abs(x0) * uround and not last):
            hOrig[0] = h
            return -1

        # PHASE 0. Check if the current time x_0 plus the current step
        # (multiplied by a safety factor to prevent steps too small)
        # exceeds the end time x_{end}.

        if ((x0 + 1.01*h - xend) * integrationDirection > 0.0):
            h = xend - x0
            last = True
        

        # K1 computation
        KerrGeodesicEquations(y0, k1, data)

        # K2 computation

        for i in range(5): # TODO: SYSTEM_SIZE
            y1[i] = y0[i] + h * A21 * k1[i]

        KerrGeodesicEquations(y1, k2, data)

        # K3 computation

        for i in range(5): # TODO: SYSTEM_SIZE
             y1[i] = y0[i] + h*(A31 * k1[i] + A32 * k2[i])

        KerrGeodesicEquations(y1, k3, data)

        # K4 computation

        for i in range(5): # TODO: SYSTEM_SIZE
            y1[i] = y0[i] + h*(A41 * k1[i] +
                               A42 * k2[i] +
                               A43 * k3[i])

        KerrGeodesicEquations(y1, k4, data)

        # K5 computation

        for i in range(5): # TODO: SYSTEM_SIZE   
            y1[i] = y0[i] + h*( A51 * k1[i] +
                                A52 * k2[i] +
                                A53 * k3[i] +
                                A54 * k4[i])

        KerrGeodesicEquations(y1, k5, data)

        # K6 computation
        
        for i in range(5): # TODO: SYSTEM_SIZE
            y1[i] = y0[i] + h*(A61 * k1[i] +
                               A62 * k2[i] +
                               A63 * k3[i] +
                               A64 * k4[i] +
                               A65 * k5[i])

        KerrGeodesicEquations(y1, k6, data)

        # K7 computation

        for i in range(5): # TODO: SYSTEM_SIZE
            y1[i] = y0[i] + h*(A71 * k1[i] +
                               A73 * k3[i] +
                               A74 * k4[i] +
                               A75 * k5[i] +
                               A76 * k6[i])

        KerrGeodesicEquations(y1, k7, data)

        # The Butcher's table (Table 5.2, [1]), shows that the estimated
        # solution has exactly the same coefficients as the ones used to
        # compute K7. Then, the solution is the last computed y1!

        # The local error of each equation is computed as the difference
        # between the solution y and the higher order solution \hat{y}, as
        # specified in the last two rows of the Butcher's table (Table
        # 5.2, [1]). Instead of computing \hat{y} and then substract it
        # from y, the differences between the coefficientes of each
        # solution have been computed and the error is directly obtained
        # using them:

        for i in range(5): # TODO: SYSTEM_SIZE
            errors[i] = h*(E1 * k1[i] +
                           E3 * k3[i] +
                           E4 * k4[i] +
                           E5 * k5[i] +
                           E6 * k6[i] +
                           E7 * k7[i])



        err = 0

        for i in range(5): # TODO: SYSTEM_SIZE
            # The local estimated error has to satisfy the following
            # condition: |err[i]| < Atol[i] + Rtol*max(|y_0[i]|, |y_j[i]|)
            # (see equation (4.10), [1]). The variable sk stores the right
            # hand size of this inequality to use it as a scale in the local
            # error computation this way we "normalize" the error and we can
            # compare it against 1.
            sk = atoli + rtoli*fmax(fabs(y0[i]), fabs(y1[i]))

            # Compute the square of the local estimated error (scaled with the
            # previous factor), as the global error will be computed as in
            # equation 4.11 ([1]): the square root of the mean of the squared
            # local scaled errors.
            sqr = (errors[i])/sk
            errors[i] = sqr*sqr
            err += errors[i]
        
        # The sum of the local squared errors in now in errors[0], but the
        # global error is the square root of the mean of those local
        # errors: we finish here the computation and store it in err.
        err = sqrt(err / SYSTEM_SIZE)  # TODO: SYSTEM_SIZE

        # For full information about the step size computation, please see
        # equation (4.13) and its surroundings in [1] and the notes in
        # Section IV.2 in [2].
        # Mainly, the new step size is computed from the previous one and
        # the current error in order to assure a high probability of
        # having an acceptable error in the next step. Furthermore, safe
        # factors and minimum/maximum factors are taken into account.
        # The stabilization of the step size behaviour is done with the
        # variable beta (expo1 depends only of beta), taking into account
        # the previous accepted error

        # Stabilization computations:
        fac11 = pow(err, expo1)
        fac = fac11 / pow(facold, beta)
        # We need the multiplying factor (always taking into account the
        # safe factor) to be between fac1 and fac2 i.e., we require
        # fac1 <= hnew/h <= fac2:
        fac = fmax(fac2_inverse, fmin(fac1_inverse, fac * safeInv))
        # New step final (but temporary) computation
        hnew = h / fac

        # Check whether the normalized error, err, is below or over 1.:
        # REJECT STEP if err > 1.

        if err > 1.0:

            # Stabilization technique with the minimum and safe factors
            #  when the step is rejected.
            hnew = h / fmin(fac1_inverse, fac11 * safeInv)
            reject = True

        else:

            # Update old factor to new current error (upper bounded to 1e-4)
            facold = fmax(err, 1.0e-4)

            # Advance current time!
            x0 += h

            # Assure the new step size does not exceeds the provided
            # bounds.

            if (fabs(hnew) > hmax):
                hnew = integrationDirection * hmax

            # If the previous step was rejected, take the minimum of the
            # old and new step sizes

            if reject:
                hnew = integrationDirection * fmin(fabs(hnew), fabs(h))

            # Necessary update for next steps: the local y0 variable holds
            # the current initial condition (now the computed solution)

            for i in range(5): # TODO: SYSTEM_SIZE
                y0[i] = y1[i]

            # This step was accepted, so it was not rejected, so reject is
            # false. SCIENCE.

            reject = False


        # Final step size update!

        h = hnew

    # END WHILE LOOP

    # Aaaaand that's all, folks! Update system value (each thread its
    # result) in the global memory :)
    
    for i in range(5): # TODO: SYSTEM_SIZE
         initCond[i] = y0[i]

    # Update the user's h, facold and x0 
    
    hOrig[0] = h
    globalFacold[0] = facold
    globalX0[0] = x0
    
    return 1


