/**
 * @file
 * This file implements two numerical solvers for systems of N first order
 * ordinary differential equations (y' = F(x,y)).
 * \rst
 * The first solver, implemented in SolverRK45 and originally called DOPRI5, is
 * described in :cite:`hairer93` and :cite:`hairer96`. Specifically, see Table
 * 5.2 :cite:`hairer93` for the Butcher's table and Section II.4, subsection
 * "Automatic Step Size Control" :cite:`hairer93`, for the automatic control
 * of the solver inner step size.
 * See also Section IV.2 :cite:`hairer96` for the stabilized algorithm.
 * The second solver, SolverRK45, is adapted from :cite:`chanrepo16`, and
 * described at :cite:`chan13`.
 * \endrst
 *
 * Given an ODEs system, a matrix of initial conditions, each one of the form
 * \f$ (x_0, y_0) \f$, and an interval \f$ [x_0, x_{end}] \f$, this code
 * computes the value of the system at \f$ x_{end} \f$ for each one of the
 * initial conditions. The computation is GPU parallelized using CUDA.
 */

 // [1] "Solving Ordinary Differential Equations I", by E. Hairer, S. P.
 // Nørsett and G. Wanner.
 // [2] "Solving Ordinary Differential Equations II", by E. Hairer, S. P.
 // Nørsett and G. Wanner.
 // [3] "A massive parallel ODE integrator for performing general relativistic
 // radiative transfer using ray tracing", by chanchikwan.
 // https://github.com/chanchikwan/gray
 // [4] "GRay: A massively parallel GPU-based code for ray tracing in
 // relativistic spacetimes", by Chi-Kwan Chan, Dimitrios P. Saltis and Feryal
 // Özel.

#include <stdio.h>
#include <math.h>
// #include <string.h>

#include "common.c"
#include "functions.c"

/**
 * This method uses DOPRI5 to advance a time of \p h the system stored in \p
 * y0.
 * It reads the system state passed as \p y0, advance it using the step \p h
 * and stores the result in \p y1. The last parameter, data, is used by the
 * rhs of the system.
 * @param[in,out]   y0   Initial state of the system.
 * @param[in]       h    Step that shall be advanced.
 * @param[out]      y1   Final state of the system,
 * @param[in]       data Additional data used by the rhs of the system.
 * @return      The normalized estimated error of the step.
 */
static inline Real advanceStep(Real* y0, Real h, Real* y1,
                                          Real* data){
    // Auxiliary variables used to compute the errors at each step.
    float sqr;                  // Scaled differences in each eq.
    float errors[SYSTEM_SIZE];  // Local error of each eq.
    float err = 0;              // Global error of the step
    float sk;                   // Scale based on the tolerances

    // Declare a counter for the loops, in order not to declare it multiple
    // times :)
    int i;

    // Auxiliar arrays to store the intermediate K1, ..., K7 computations
    Real k1[SYSTEM_SIZE],
         k2[SYSTEM_SIZE],
         k3[SYSTEM_SIZE],
         k4[SYSTEM_SIZE],
         k5[SYSTEM_SIZE],
         k6[SYSTEM_SIZE],
         k7[SYSTEM_SIZE];

    // Compute the K1, ..., K7 components and the estimated solution, using
    // the Butcher's table described in Table 5.2 ([1])

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

    return err;
}


/**
 * This function receives the current state of a ray that has just crossed the
 * equatorial plane (theta = pi/2) and makes a binary search of the exact
 * (with a tolerance of BISECT_TOL) point in which the ray crossed it. This
 * code expects the following:
 * 		- In time = x, the ray is at one side of the equatorial plane.
 * 		- In time = x - step, the ray was at the opposite side of the
 * 		equatorial plane.
 *
 * @param[in,out]   yOriginal Pointer to the array where the ray
 *                  state is stored, following the usual order used
 *                  throughout this code: r, theta, phi, pR and pTheta.
 * @param[in]       data Device pointer to a serialized matrix of additional
 *                  data to be passed to computeComonent; currently, this is
 *                  used to pass the constants b and q of each ray to the
 *                  computeComponent method.
 * @param[in]       step x - step was the last time in which the ray was found
 *                  in the opposite side of the equatorial plane it is in the
 *                  current time; i.e., at time = x.
 * @param[in]       x Current time.
 * @return          Number of iterations used in the binary search.
 */
int bisect(Real* yOriginal, Real* data, Real step, Real x){
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

    // Array used by advanceStep() routine, which expects a pointer where the
    // computed new state should be stored
    Real yNew[SYSTEM_SIZE];

    // This loop implements the main behaviour of the algorithm; basically,
    // this is how it works:
    //      1. It advance the point one single step with the RK45 algorithm.
    //      2. If theta has crossed pi/2, it changes the direction of the
    //      new step. The magnitude of the new step is always half of the
    //      magnitude of the previous one.
    //      3. It repeats 1 and 2 until the current theta is very near of Pi/2
    //      ("very near" is defined by BISECT_TOL) or until the number of
    //      iterations exceeds a maximum number previously defined.
    while(fabs(prevThetaCentered) > BISECT_TOL && iter < BISECT_MAX_ITER){
        // 1. Advance the ray one step.
        advanceStep(yOriginal, step, yNew, data);
        memcpy(yOriginal, yNew, sizeof(Real)*SYSTEM_SIZE);
        x += step;

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

/**
 * Applies the DOPRI5 algorithm over the system defined in the computeComponent
 * function, using the initial conditions specified in \p devX0 and \p
 * devInitCond, and returning the solution found at \p xend.
 * @param[in,out] globalX0     Start of the integration interval
 *                [x_0, x_{end}]. At the output, this variable is set
 *                to the final time the solver reached.
 * @param[in]     xend         End of the integration interval
 *                [x_0, x_{end}].
 * @param[in,out] initCond     Device pointer to a serialized matrix of
 *                initial conditions; i.e., given a 2D matrix of R rows
 *                and C columns, where every entry is an n-tuple of
 *                initial conditions (y_0[0], y_0[1], ..., y_0[n-1]),
 *                the vector pointed by devInitCond contains R*C*n
 *                serialized entries, starting with the first row from
 *                left to right, then the second one in the same order
 *                and so on.
 *                The elements of vector pointed by initCond are
 *                replaced with the new computed values at the end of
 *                the algorithm; please, make sure you will not need
 *                them after calling this procedure.
 * @param[in,out] hOrig        Step size. This code controls
 *                automatically the step size, but this value is taken
 *                as a test for the first try; furthermore, the method
 *                returns the last computed value of h to let the user
 *                know the final state of the solver.
 * @param[in]     hmax         Value of the maximum step size allowed,
 *                usually defined as x_{end} - x_0, as we do not to
 *                exceed x_{end} in one iteration.
 * @param[in]     data         Device pointer to a serialized matrix of
 *                additional data to be passed to computeComonent;
 *                currently, this is used to pass the constants b and q
 *                of each ray to the computeComponent method.
 * @param[out]    iterations   Output variable to know how many
 *                iterations were spent in the computation
 */
 int SolverRK45(Real* globalX0, Real xend, Real* initCond,
                           Real hOrig, Real hmax, Real* data, int* iterations){
    // Loop variable to manage the automatic step size detection.
    Real hnew;

    // Retrieve the initial values of h and x0
    // TODO: Implement the hinit method
    Real h = hOrig;
    Real x0 = *globalX0;

    // Check the direction of the integration, to the future or to the past,
    // and get the absolute value of the maximum step size.
    Real integrationDirection = xend - x0 > 0. ? +1. : -1.;
    hmax = fabs(hmax);

    // Precompute size of the ray's state in bytes
    size_t sizeBytes = sizeof(Real)*SYSTEM_SIZE;

    // Each thread of each block has to know only the initial condition
    // associated to its own equation:
    Real y0[SYSTEM_SIZE];
    memcpy(y0, initCond, sizeBytes);

    // Auxiliar array to store the intermediate calls to the
    // computeComponent function
    Real y1[SYSTEM_SIZE];

    // Auxiliary variable used to control the global error at each step
    float err = 0;

    // Initial values for the step size automatic prediction variables.
    // They are basically factors to maintain the new step size in known
    // bounds, but you can see the corresponding chunk of code far below to
    // know more about the purpouse of each of these variables.
    float facold = 1.0e-4;
    float expo1 = 0.2 - beta * 0.75;
    float fac11, fac;

    // Loop variables initialisation. The current step is repeated when
    // `reject` is set to true, event that happens when the global error
    // estimation exceeds 1.
    Real reject = false;

    // Variables to keep track of the current r and the previous and
    // current theta
    Real currentR;
    int prevThetaSign, currentThetaSign;

    // Initialize previous theta to the initial conditions
    prevThetaSign = sign(y0[1] - HALF_PI);

    // Local variable to know how many iterations spent the bisect in the
    // current step.
    int bisectIter = 0;

    // Initial status of the ray: SPHERE
    int status = SPHERE;

    // Main loop. Each iteration computes a single step of the DOPRI5 algorithm, that roughly comprises the next phases:
    //  0. Check that the expected step does not exceed the final time.
    //  1. Computation of the system new value and the estimated error.
    //  2. Computation of the new step size.
    //  3. Check if the current step has to be repeated:
    //      3.1 If the estimated global error > 1., repeat the step.
    //      3.2 In any other case, update the current time and:
    //          3.2.1 If the ray has crossed the equatorial plane (theta =
    //          pi/2), find the exact point where this cross occured, updating
    //          the ray's status to DISK if a collision is found.
    do{
        *iterations = *iterations + 1;

        // Check that the step size is not too small and that the horizon is
        // not too near. In both cases, set the ray's status to HORIZON and
        // stop the computation
        if (0.1 * fabs(h) <= fabs(x0) * uround || (y0[0] - horizonRadius <= 1e-3)){
            if(0.1 * fabs(h) <= fabs(x0) * uround)
                printf("%.30f, %.30f, %.30f, %.30f\n", 0.1, fabs(h), fabs(x0), uround);
            // Let the user know the computation stopped before xEnd
            status = HORIZON;
            break;
        }

        // PHASE 0. Check if the current time x_0 plus the current step
        // (multiplied by a safety factor to prevent steps too small)
        // exceeds the end time x_{end}.
         if ((x0 + 1.01*h - xend) * integrationDirection > 0.0){
             h = xend - x0;
         }

        // PHASE 1. Compute the new state of the system and the estimated
        // error of the step.
        err = advanceStep(y0, h, y1, data);

        // PHASE 2. Compute the new step size.
        //
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

        // Compute the (temporary) new step
        hnew = h / fac;

        // PHASE 3. Check whether the current step has to be repeated,
        // depending on its estimated error:
        //
        // Check whether the normalized error, err, is below or over 1.:
        // PHASE 3.1: REJECT STEP if err > 1.
        if( err > 1.){
            // Stabilization technique with the minimum and safe factors
            // when the step is rejected.
            hnew = h / fmin(fac1_inverse, fac11 * safeInv);

            // Set reject variable to true for the next step.
            reject = true;
        }
        // PHASE 3.2: ACCEPT STEP if err <= 1.
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

            // This step was accepted, so it was not rejected, so reject is
            // false. SCIENCE.
            reject = false;

            // Necessary update for next steps: the local y0 variable holds
            // the current initial condition (now the computed solution)
            memcpy(y0, y1, sizeBytes);

            // PHASE 3.2.1: Check if theta has crossed pi/2
            // Update current theta
            currentThetaSign = sign(y1[1] - HALF_PI);

            // Check whether the ray has crossed theta = pi/2
            if(prevThetaSign != currentThetaSign){
                // Call bisect in order to find the exact spot where theta =
                // pi/2
                bisectIter += bisect(y1, data, h, x0);

                // Retrieve the current r
                currentR = y1[0];

                // Finally, check whether the current r is inside the disk,
                // updating the status and copying back the data in the
                // case it is.
                if(innerDiskRadius<currentR && currentR<outerDiskRadius){
                    memcpy(y0, y1, sizeof(Real)*SYSTEM_SIZE);
                    status = DISK;
                    break;
                }
            }

            // Update the previous variable for the next step computation
            prevThetaSign = currentThetaSign;
        }

        // Final step size update!
        h = hnew;
    }while(x0 > xend);

    // Aaaaand that's all, folks! Update system value (each thread its
    // result) in the global memory :)
    memcpy(initCond, y0, sizeBytes);

    // Update the user's x0
    *globalX0 = x0;

    // Finally, let the user know everything's gonna be alright
    return status;
}
