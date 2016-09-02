/**
 * This file implements two numerical solvers for systems of N first order
 * ordinary differential equations (y' = F(x,y)).
 * The first solver, implemented in SolverRK45 and originally called DOPRI5, is
 * described in [1] and [2]. Specifically, see Table 5.2 ([1]) for the
 * Butcher's table and Section II.4, subsection "Automatic Step Size Control"
 * ([1]), for the automatic control of the solver inner step size. See also
 * Section IV.2 ([2]) for the stabilized algorithm.
 * The second solver, SolverRK45, is adapted from [3], and described at [4].
 *
 * Given an ODEs system, a matrix of initial conditions, each one of the form
 * (x_0, y_0), and an interval [x_0, x_{end}], this code computes the value of
 * the system at x_{end} for each one of the initial conditions. The
 * computation is GPU parallelized using CUDA.
 *
 * -------------------------------
 * [1] "Solving Ordinary Differential Equations I", by E. Hairer, S. P. Nørsett and G. Wanner.
 * [2] "Solving Ordinary Differential Equations II", by E. Hairer, S. P. Nørsett and G. Wanner.
 * [3] "A massive parallel ODE integrator for performing general relativistic
 * radiative transfer using ray tracing", by chanchikwan.
 * https://github.com/chanchikwan/gray
 * [4] "GRay: A massively parallel GPU-based code for ray tracing in
 * relativistic spacetimes", by Chi-Kwan Chan, Dimitrios P. Saltis and Feryal
 * Özel.
 */

#include <stdio.h>
#include <math.h>

#include "Raytracer/Kernel/common.cu"
#include "Raytracer/Kernel/functions.cu"

/**
 * Applies the DOPRI5 algorithm over the system defined in the computeComponent
 * function, using the initial conditions specified in devX0 and devInitCond,
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
 *                        of each ray to the computeComponent method.
 * @param[out]     int*   iterations   Output variable to know how many
 *                        iterations were spent in the computation
 * @param[in,out]  float* globalFacold Input and output variable, used as a
 *                        first value for facold and to let the caller know the
 *                        final value of facold.
 */
 __device__ SolverStatus SolverRK45(Real* globalX0, Real xend, Real* initCond,
                          Real* hOrig, Real hmax, Real* data, int* iterations, float* globalFacold){
    #ifdef DEBUG
        printf("ThreadId %d - INITS: x0=%.30f, xend=%.30f, y0=(%.30f, %.30f)\n", threadId, x0, xend, ((Real*)initCond)[0], ((Real*)initCond)[1]);
    #endif

    // Each equation to solve has a thread that compute its solution. Although
    // in an ideal situation the number of threads is exactly the same as the
    // number of equations, it is usual that the former is greater than the
    // latter. The reason is that the number of threads has to be a power of 2
    // (see reduction technique in the only loop you will find in this function
    // code to know why): then, we can give some rest to the threads that exceeds the number of equations :)

    // Loop variable to manage the automatic step size detection.
    // TODO: Implement the hinit method
    Real hnew;

    // Retrieve the value of h and the value of x0
    Real h = *hOrig;
    Real x0 = *globalX0;

    // Check the direction of the integration: to the future or to the past
    // and get the absolute value of the maximum step size.
    Real integrationDirection = xend - x0 > 0. ? +1. : -1.;
    hmax = abs(hmax);

    size_t sizeBytes = sizeof(Real)*SYSTEM_SIZE;

    // Each thread of each block has to know only the initial condition
    // associated to its own equation:
    Real y0[SYSTEM_SIZE];
    memcpy(y0, initCond, sizeBytes);

    // Auxiliar arrays to store the intermediate K1, ..., K7 computations
    Real k1[SYSTEM_SIZE],
         k2[SYSTEM_SIZE],
         k3[SYSTEM_SIZE],
         k4[SYSTEM_SIZE],
         k5[SYSTEM_SIZE],
         k6[SYSTEM_SIZE],
         k7[SYSTEM_SIZE];

    // Auxiliar array to store the intermediate calls to the
    // computeComponent function
    Real y1[SYSTEM_SIZE];

    // Auxiliary variables used to compute the errors at each step.
    float sqr;                            // Scaled differences in each eq.
    float errors[SYSTEM_SIZE]; // Local error of each eq.
    float err = 0;                            // Global error of the step
    float sk; // Scale based on the tolerances

    // Initial values for the step size automatic prediction variables.
    // They are basically factors to maintain the new step size in known
    // bounds, but you can see the corresponding chunk of code far below to
    // know more about the puropose of each of these variables.
    float facold = *globalFacold;
    float expo1 = 0.2 - beta * 0.75;
    float fac11, fac;

    // Loop variables initialisation. The current step is repeated when
    // `reject` is set to true, event that happens when the global error
    // estimation exceeds 1.
    Real reject = false;

    // Declare a counter for the loops, in order not to declare it multiple
    // times :)
    int i;

    // Main loop. Each iteration computes a single step of the DOPRI5 algorithm, that roughly comprises the next phases:
    //  0. Check if this step has to be the last one.
    //  1. Computation of the system new value and the estimated error.
    //  2. Computation of the new step size.
    //  3. Check if the current step has to be repeated:
    //      3.1 If the estimated global error > 1., repeat the step.
    //      3.2 In any other case, update the current time and:
    //          3.2.1 If this is the last step, finish.
    //          3.2.2 In any other case, iterate again.
    do{
        *iterations = *iterations + 1;

        // Check that the step size is not too small and that the horizon is
        // not too near. Although the last condition belongs to the raytracer
        // logic, it HAS to be checked here.
        if (0.1 * abs(h) <= abs(x0) * uround || (y0[0] - horizonRadius <= 1e-3)){
            // Let the user knwo the final step
            *hOrig = h;

            // Let the user know the computation stopped before xEnd
            return SOLVER_FAILURE;
        }

        // PHASE 0. Check if the current time x_0 plus the current step
        // (multiplied by a safety factor to prevent steps too small)
        // exceeds the end time x_{end}.
         if ((x0 + 1.01*h - xend) * integrationDirection > 0.0){
             h = xend - x0;
         }

        // PHASE 1. Compute the K1, ..., K7 components and the estimated
        // solution, using the Butcher's table described in Table 5.2 ([1])

        // K1 computation
        computeComponent(y0, k1, data);

        // K2 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + h * A21 * k1[i];
        }
        computeComponent(y1, k2, data);

        // K3 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + h*(A31 * k1[i] +
                               A32 * k2[i]);
        }
        computeComponent(y1, k3, data);

        // K4 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + h*(A41 * k1[i] +
                               A42 * k2[i] +
                               A43 * k3[i]);
        }
        computeComponent(y1, k4, data);

        // K5 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + h*( A51 * k1[i] +
                                A52 * k2[i] +
                                A53 * k3[i] +
                                A54 * k4[i]);
        }
        computeComponent(y1, k5, data);

        // K6 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + h*(A61 * k1[i] +
                               A62 * k2[i] +
                               A63 * k3[i] +
                               A64 * k4[i] +
                               A65 * k5[i]);
        }
        computeComponent(y1, k6, data);

        // K7 computation.
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + h*(A71 * k1[i] +
                               A73 * k3[i] +
                               A74 * k4[i] +
                               A75 * k5[i] +
                               A76 * k6[i]);
        }
        computeComponent(y1, k7, data);

        // The Butcher's table (Table 5.2, [1]), shows that the estimated
        // solution has exactly the same coefficients as the ones used to
        // compute K7. Then, the solution is the last computed y1!

        // The local error of each equation is computed as the difference
        // between the solution y and the higher order solution \hat{y}, as
        // specified in the last two rows of the Butcher's table (Table
        // 5.2, [1]). Instead of computing \hat{y} and then substract it
        // from y, the differences between the coefficientes of each
        // solution have been computed and the error is directly obtained
        // using them:
        for(i = 0; i < SYSTEM_SIZE; i++){
            errors[i] = h*(E1 * k1[i] +
                           E3 * k3[i] +
                           E4 * k4[i] +
                           E5 * k5[i] +
                           E6 * k6[i] +
                           E7 * k7[i]);
        }


        #ifdef DEBUG
            printf("ThreadId %d - K 1-7: K1:%.30f, K2:%.30f, K3:%.30f, K4:%.30f, K5:%.30f, K6:%.30f, K7:%.30f\n", threadId, k1[threadId], k2[threadId], k3[threadId], k4[threadId], k5[threadId], k6[threadId], k7[threadId]);
            printf("ThreadId %d - Local: sol: %.30f, error: %.30f\n", threadId, solution[threadId], errors[threadId]);
        #endif

        err = 0;
        for(i = 0; i < SYSTEM_SIZE; i++){
            // The local estimated error has to satisfy the following
            // condition: |err[i]| < Atol[i] + Rtol*max(|y_0[i]|, |y_j[i]|)
            // (see equation (4.10), [1]). The variable sk stores the right
            // hand size of this inequality to use it as a scale in the local
            // error computation; this way we "normalize" the error and we can
            // compare it against 1.
            sk = atoli + rtoli*fmax(fabs(y0[i]), fabs(y1[i]));

            // Compute the square of the local estimated error (scaled with the
            // previous factor), as the global error will be computed as in
            // equation 4.11 ([1]): the square root of the mean of the squared
            // local scaled errors.
            sqr = (errors[i])/sk;
            errors[i] = sqr*sqr;

            err += errors[i];
        }

        // The sum of the local squared errors in now in errors[0], but the
        // global error is the square root of the mean of those local
        // errors: we finish here the computation and store it in err.
        err = sqrt(err / SYSTEM_SIZE);

        // For full information about the step size computation, please see
        // equation (4.13) and its surroundings in [1] and the notes in
        // Section IV.2 in [2].
        // Mainly, the new step size is computed from the previous one and
        // the current error in order to assure a high probability of
        // having an acceptable error in the next step. Furthermore, safe
        // factors and minimum/maximum factors are taken into account.
        // The stabilization of the step size behaviour is done with the
        // variable beta (expo1 depends only of beta), taking into account
        // the previous accepted error

        // Stabilization computations:
        fac11 = pow (err, expo1);
        fac = fac11 / pow(facold, (float)beta);
        // We need the multiplying factor (always taking into account the
        // safe factor) to be between fac1 and fac2; i.e., we require
        // fac1 <= hnew/h <= fac2:
        fac = fmax(fac2_inverse, fmin(fac1_inverse, fac * safeInv));
        // New step final (but temporary) computation
        hnew = h / fac;

        #ifdef DEBUG
            printf("ThreadId %d - H aux: expo1: %.30f, err: %.30f, fac11: %.30f, facold: %.30f, fac: %.30f\n", threadId, expo1, err, fac11, facold, fac);
            printf("ThreadId %d - H new: prevH: %.30f, newH: %.30f\n", threadId, hnew);
        #endif

        // Check whether the normalized error, err, is below or over 1.:
        // REJECT STEP if err > 1.
        if( err > 1.){
            // Stabilization technique with the minimum and safe factors
            // when the step is rejected.
            hnew = h / fmin(fac1_inverse, fac11 * safeInv);

            // Set reject variable to true for the next step and make sure
            // this one is not the last step!
            reject = true;
        }
        // ACCEPT STEP if err <= 1.
        else{
            // TODO: Stiffness detection

            // Update old factor to new current error (upper bounded to
            // 1e-4)
            facold = fmax(err, 1.0e-4);

            // Advance current time!
            x0 += h;

            // Assure the new step size does not exceeds the provided
            // bounds.
            if (fabs(hnew) > hmax)
                hnew = integrationDirection * hmax;

            // If the previous step was rejected, take the minimum of the
            // old and new step sizes
            if (reject)
                hnew = integrationDirection * fmin(fabs(hnew), fabs(h));

            // Necessary update for next steps: the local y0 variable holds
            // the current initial condition (now the computed solution)
            memcpy(y0, y1, sizeBytes);

            // This step was accepted, so it was not rejected, so reject is
            // false. SCIENCE.
            reject = false;
        }

        // Final step size update!
        // if(!last)
        h = hnew;

        #ifdef DEBUG
            if(threadId == 0){
                if(err > 1.){
                    printf("\n###### CHANGE: err: %.30f, h: %.30f\n\n", err, h);
                }
                else{
                    printf("\n###### ======:  err: %.30f, h: %.30f\n\n", err, h);
                }
            }
        #endif
    }while(x0 > xend);

    // Aaaaand that's all, folks! Update system value (each thread its
    // result) in the global memory :)
    memcpy(initCond, y0, sizeBytes);

    // Update the user's h, facold and x0
    *hOrig = h;
    *globalFacold = facold;
    *globalX0 = x0;


    // Finally, let the user know everything's gonna be alright
    return SOLVER_SUCCESS;
}

/**
 * Auxiliar method used by SolverRK4 in order to obtain an adapted step size.
 * This is heavily influenced by the raytracer logic, and it has sense only in
 * this context, not as a general automatic step controller. The returned step
 * size depends on the distance to the horizon, the speed and the angular
 * change speed of the current ray state, passed to this method in the pos and
 * vel arrays.
 * This code is adapted from [3], his original author is chanchikwan.
 *
 * @param[in]       Real* pos           Pointer to an array of at least three
 *                        componentes in the following order: current distance
 *                        to the horizon, current theta, current phi. It is
 *                        usually called with the ray state; i.e., the array
 *                        with the 5 components used throughout this sofware:
 *                        r, theta, phi, pR and pTheta. Currently, the last two
 *                        components are completely ignored.
 *
 * @param[in]       Real* vel           Pointer to an array of at least three
 *                        componentes in the following order: current value of
 *                        the derivative of the distance to the horizon,
 *                        current value of the derivative of theta, current
 *                        value of the derivative of phi. It is usually called
 *                        with the returned values of computeComponent, that
 *                        gives the values of theSE derivatives we need.
 *                        Although the current passed array has also the values
 *                        of the derivatives of pR and pTheta, these two
 *                        components are ignored.
 *
 * @param[in]       Real  hmax          Value of the maximum step size allowed.
 *
 * @return          Real                Absolute value of the step size that
 *                        shall be used in the RK4 solver. Please, note that in
 *                        order to use this value, you have to
 *                        take into account the direction of the integration:
 *                        if the integration is to the past, the value you need
 *                        is the opposite of the returned value.
 */
static inline __device__ Real getStepSize(Real* pos, Real* vel, Real hmax){
    // Make sure the hmax value is positive, as we are only interested on the
    // size of the step, not the sign.
    hmax = fabs(hmax);

    // If the current position is zero, return a special value that tells the
    // solver to stop
    if(pos[0] < horizonRadius + SOLVER_EPSILON){
        return 0; // too close to the black hole
    }

    // Compute the two values from which we are gonna take the minimum one.
    // See equation (2) in Chi-kwan Chan's paper [4] for more information.
    float f1 = SOLVER_DELTA / (fabs(vel[0] / pos[0]) + fabs(vel[1]) +
                               fabs(vel[2]));

    float f2 = (pos[0] - horizonRadius) / fabs((2*vel[0]));

    // Take the minimum of the previous computed values, always uppper bounded
    // by the provided hmax.
    return fmin(hmax, fmin(f1, f2));
}


/**
 * Usual Runge-Kutta 4 solver, adapted from [3]. The solver uses the system
 * defined in the computeComponent function, using the initial conditions
 * specified in x0 and initCond, and returning the solution found at xend.
 * Furthermore, the step size is automatically computed using the getStepSize
 * function, also adapted from [3].
 * @param[in]      Real  x0       Start of the integration interval
 *                       [x_0, x_{end}].
 * @param[in]      Real  xend     End of the integration interval
 *                       [x_0, x_{end}].
 * @param[in,out]  Real* initCond Device pointer to a serialized matrix of
 *                       initial conditions; i.e., given a 2D matrix of R rows
 *                       and C columns, where every entry is an n-tuple of
 *                       initial conditions (y_0[0], y_0[1], ..., y_0[n-1]),
 *                       the vector pointed by devInitCond contains R*C*n
 *                       serialized entries, starting with the first row from
 *                       left to right, then the second one in the same order
 *                       and so on.
 *                       The elements of vector pointed by initCond are
 *                       replaced with the new computed values at the end of
 *                       the algorithm; please, make sure you will not need
 *                       them after calling this procedure.
 * @param[out]     Real* hOrig    This code controls automatically the step
 * 						 size. This output variable is used to comunicate the
 * 						 caller the last step size computed.
 * @param[in]      Real  hmax     Value of the maximum step size allowed,
 *                       usually defined as x_{end} - x_0, as we do not to
 *                       exceed x_{end} in one iteration.
 * @param[in]      Real* data     Device pointer to a serialized
 *                       matrix of additional data to be passed to
 *                       computeComonent; currently, this is used to pass the
 *                       constants b and q of each ray to the computeComponent
 *                       method.
 */
 __device__ SolverStatus SolverRK4(Real x0, Real xend, Real* initCond,
                          Real* hOrig, Real hmax, Real* data){
    #ifdef DEBUG
        printf("ThreadId %d - INITS: x0=%.20f, xend=%.20f, y0=(%.20f, %.20f)\n", threadId, x0, xend, ((Real*)initCond)[0], ((Real*)initCond)[1]);
    #endif

    // Check the direction of the integration: to the future or to the past
    // and get the absolute value of the maximum step size.
    Real integrationDirection = xend - x0 > 0. ? +1. : -1.;

    // Each thread of each block has to know only the initial condition
    // associated to its own equation:
    Real y0[SYSTEM_SIZE];
    memcpy(y0, initCond, sizeof(Real)*SYSTEM_SIZE);

    // Auxiliar arrays to store the intermediate K1, ..., K7 computations
    Real k1[SYSTEM_SIZE],
         k2[SYSTEM_SIZE],
         k3[SYSTEM_SIZE],
         k4[SYSTEM_SIZE];

    // Auxiliar array to store the intermediate calls to the
    // computeComponent function
    Real y1[SYSTEM_SIZE];

    // Loop variables initialisation. The main loop finishes when `last` is
    // set to true, event that happens when the current x0 plus the current
    // step exceeds x_{end}. Furthermore, the current step is repeated when
    // `reject` is set to true, event that happens when the global error
    // estimation exceeds 1.
    bool last  = false;

    // Retrieve the value of h
    Real h, half_h;

    // Declare a counter for the loops, in order not to declare it multiple
    // times :)
    int i;

    // Main loop. Each iteration computes a single step of the RK4 algorithm,
    // that roughly comprises the next phases:
    //  0. Compute the new step size.
    //  1. Check if the current step has to be the last one; i.e., check that
    //  x0+h exceeds or not xend
    //  2. Check if the ray has collided with the horizon, inspecting the value
    //  returned by getStepSize;
    //  3. Computation of the system new value.
    do{
        // PHASE 0. Compute the new step size (we need K1 in order to do that)
        // and check if the current time x_0 plus the computed step exceeds the
        // end time x_{end}.

        // K1 computation
        computeComponent(y0, k1, data);

        // Compute the step size
        h = integrationDirection * getStepSize(y0, k1, hmax);

        // PHASE 1. Check if this step has to be the last one
        if ((x0 + h - xend) * integrationDirection > 0.0){
            h = xend - x0;
            last = true;
        }

        // PHASE 2. See if we've collided with the horizon (getStepSize returns
        // 0 if the horizon is too close to continue with the computation)
        if(h == 0){
            return SOLVER_FAILURE;
        }

        // PHASE 3. Compute the K1, ..., K4 components and the estimated
        // solution, using the RK4 Butcher's table

        // Pre-compute a value that is used later in the code.
        half_h = h*0.5;

        // K2 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + half_h * k1[i];
        }
        computeComponent(y1, k2, data);

        // K3 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + half_h * k2[i];
        }
        computeComponent(y1, k3, data);

        // K4 computation
        for(i = 0; i < SYSTEM_SIZE; i++){
            y1[i] = y0[i] + h * k3[i];
        }
        computeComponent(y1, k4, data);


        for(i = 0; i < SYSTEM_SIZE; i++){
            y0[i] = y0[i] + (1./6.) * h * (k1[i] + 2*(k2[i] + k3[i]) + k4[i]);
        }

        // Advance current time!
        x0 += h;

    }while(!last && (xend - x0) > 0.0);

    // Aaaaand that's all, folks! Update system value (each thread its
    // result) in the global memory :)
    for(int i = 0; i < SYSTEM_SIZE; i++){
        initCond[i] = y0[i];
    }

    // Update the user's h
    *hOrig = h;

    // Finally, let the user know everything's gonna be alright
    return SOLVER_SUCCESS;
}


/**
 * This function receives the current state of a ray that has just crossed the equatorial plane (theta = pi/2) and makes a binary search of the exact (with a tolerance of BISECT_TOL) point in which the ray crossed it. This code expects the following:
 * 		- In time = x, the ray is at one side of the equatorial plane.
 * 		- In time = x-step, the ray was at the opposite side of the equatorial
 * 		plane.
 * @param[in,out]   Real* yOriginal     Pointer to the array where the ray
 *                        state is stored, following the usual order used
 *                        throughout this code: r, theta, phi, pR and pTheta.
 * @param[in]       Real* data          Device pointer to a serialized matrix
 *                        of additional data to be passed to computeComonent;
 *                        currently, this is used to pass the constants b and q
 *                        of each ray to the computeComponent method.
 * @param[in]       Real  step          x-step was the last time in which the
 *                        ray was found in the opposite side of the equatorial
 *                        plane it is in the current time; i.e., at time = x.
 * @param[in]       Real  x             Current time.
 * @return          int                 Number of iterations used in the binary
 *                        search.
 */
__device__ int bisect(Real* yOriginal, Real* data, Real step, Real x){
    // It is necessary to maintain the previous theta to know the direction
    // change; we'll store it centered in zero, and not in pi/2 in order to
    // removes some useless substractions in the main loop.
    Real prevThetaCentered, currentThetaCentered;
    prevThetaCentered = yOriginal[1] - HALF_PI;

    // The first step shall be to the other side and half of its length.
    step = - step * 0.5;

    // Loop variables, to control that the iterations does not exceed a maximum
    // number
    int iter = 0;

    // Step passed to the RK4 solver (currently not used)
    Real h = -step*0.01;

    // Variable to control the success or failure of the solver
    SolverStatus solverStatus;

    // This loop implements the main behaviour of the algorithm; basically,
    // this is how it works:
    //      1. It advance the point one single step with the RK4 algorithm.
    //      2. If theta has crossed pi/2, it changes the direction of the
    //      new step. The magnitude of the new step is always half of the
    //      magnitude of the previous one.
    //      3. It repeats 1 and 2 until the current theta is very near of Pi/2
    //      ("very near" is defined by BISECT_TOL) or until the number of
    //      iterations exceeds a maximum number previously defined.
    while(fabs(prevThetaCentered) > BISECT_TOL && iter < BISECT_MAX_ITER){
        // 1. Advance the ray one step.
        solverStatus = SolverRK4(x, x + step, yOriginal, &h, step, data);
        x += step;

        // Safety guard in case the solver fails.
        if(solverStatus == SOLVER_FAILURE){
            return -1;
        }

        // Compute the current theta, centered in zero
        currentThetaCentered = yOriginal[1] - HALF_PI;

        // 2. Change the step direction whenever theta crosses the target,
        // pi/2, and make it half of the previous one.
        step = step * sign(currentThetaCentered)*sign(prevThetaCentered) * 0.5;

        // Update the previous theta, centered in zero, with the current one
        prevThetaCentered = currentThetaCentered;

        iter++;
    } // 3. End of while

    // Return the number of iterations spent in the loop
    return iter;
}