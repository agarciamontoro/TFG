/**
 * This file implements a numerical solver for systems of N first order
 * ordinary differential equations (y' = F(x,y)), using the Dormand and Prince
 * algorithm: a Runge-Kutta method of order 5(4) called DOPRI5.
 * The algorithm is described in [1] and [2]. Specifically, see Table 5.2 ([1])
 * for the Butcher's table and Section II.4, subsection "Automatic Step Size Control" ([1]), for the automatic control of the solver inner step size. See also Section IV.2 ([2]) for the stabilized algorithm.
 *
 * Given an ODEs system, a matrix of initial conditions, each one of the form
 * (x_0, y_0), and an interval [x_0, x_{end}], this code computes the value of
 * the system at x_{end} for each one of the initial conditions. The
 * computation is GPU parallelized using CUDA.
 *
 * -------------------------------
 * [1] "Solving Ordinary Differential Equations I", by E. Hairer, S. P. Nørsett and G. Wanner.
 * [2] "Solving Ordinary Differential Equations II", by E. Hairer, S. P. Nørsett and G. Wanner.
 */

#include <stdio.h>
#include <math.h>

#include "common.c"
#include "functions.c"

/**
 * Applies the DOPRI5 algorithm over the system defined in the computeComponent
 * function, using the initial conditions specified in devX0 and devInitCond,
 * and returning the solution found at xend.
 *
 * @param[in,out]  Real  x0           Start of the integration interval
 * 						 [x_0, {end}].
 * @param[in]      Real  xend         End of the integration interval
 *                       [x_0, x{end}].
 * @param[in,out]  void  *devInitCond Device pointer to a serialized matrix of
 *                       initial conditions; i.e., given a 2D matrix of R rows
 *                       and C columns, where every entry is an n-tuple of
 *                       initial conditions (y_0[0], y_0[1], ..., y_0[n-1]),
 *                       the vector pointed by devInitCond contains R*C*n
 *                       serialized entries, starting with the first row from
 *                       left to right, then the second one in the same order
 *                       and so on.
 *                       The elements of vector pointed by devInitCond are
 *                       replaced with the new computed values at the end of
 *                       the algorithm; please, make sure you will not need
 *                       them after calling this procedure.
 * @param[in]      Real  h            Step size. This code controls
 *                       automatically the step size, but this value is taken
 *                       as a test for the first try.
 * @param[in]      Real  hmax         Value of the maximum step size allowed,
 * 						 usually defined as x_{end} - x_0, as we do not to
 * 						 exceed x_{end} in one iteration.
 * @param[in]      void* globalRtoler Device pointer to a vector of n Reals,
 *                       where the i-th element specifies the relative error
 *                       tolerance for the i-th equation at each step, as
 *                       defined in the following inequality, that has to hold
 *                       for every intermediate step j:
 *                       	|err[i]| < Atol[i] + Rtol*max(|y_0[i]|, |y_j[i]|),
 *                       where err[i] is the estimation of the error in the
 *                       i-th equation at each step. Defaults to 1e-6 for every
 *                       equation.
 * @param[in]      void* globalAtoler Device pointer to a vector of n Reals,
 *                       where the i-th element specifies the absolute error
 *                       tolerance for the i-th equation at each step, as
 *                       defined in the previous inequality. Defaults to 1e-12
 *                       for every equation.
 * @param[in]      Real  safe         Factor in [0., 1.] used in the step size
 *                       prediction to increase the probability of an
 *                       acceptable error in the next iteration. Defaults to
 *                       0.9.
 * @param[in]      Real  fac1         Minimum factor used in the step size
 *                       prediction. The predicted new step is then:
 *                       	h_{new} > fac1 * h_{prev}
 * @param[in]      Real  fac2         Minimum factor used in the step size
 *                       prediction. The predicted new step is then:
 *                       	h_{new} < fac1 * h_{prev}
 * @param[in]      Real  beta         Factor used in the stabilization of the
 *                       step size control. Defaults to 0.04.
 * @param[in]      Real  uround       Rounding unit -should be equal to the
 *                       machine precision-, used in the detection of really
 *                       small step sizes. If the following inequality holds
 *                       0.1|h| <= |x|uround, then the step size is
 *                       too small to continue the computation. Defaults to
 *                       2.3E-16. TODO: This detection is not yet implemented,
 *                       so this variable is useless.
 */
void RK4Solve(Real x0, Real xend, Real* initCond,
                          Real* hOrig, Real hmax, Real* data, bool* success,
                          int blockId){
    // Each equation to solve has a thread that compute its solution. Although
    // in an ideal situation the number of threads is exactly the same as the
    // number of equations, it is usual that the former is greater than the
    // latter. The reason is that the number of threads has to be a power of 2
    // (see reduction technique in the only loop you will find in this function
    // code to know why): then, we can give some rest to the threads that exceeds the number of equations :)
    // Flag shared between all threads in block to stop execution when one
    // of them cannot continue
    bool keepRunning = true;

    // Shared array between the block threads to store intermediate
    // solutions.
    Real solution[SYSTEM_SIZE];

    // Loop variable to manage the automatic step size detection.
    // TODO: Implement the hinit method
    Real hnew;

    // Check the direction of the integration: to the future or to the past
    // and get the absolute value of the maximum step size.
    Real integrationDirection = xend - x0 > 0. ? +1. : -1.;
    hmax = fabs(hmax);

    // Each thread of each block has to know only the initial condition
    // associated to its own equation:
    Real y0[SYSTEM_SIZE];
    for(int thread = 0; thread < SYSTEM_SIZE; thread++){
        y0[thread] = initCond[thread];
    }

    // Auxiliar arrays to store the intermediate K1, ..., K7 computations
    Real k1[SYSTEM_SIZE], k2[SYSTEM_SIZE], k3[SYSTEM_SIZE], k4[SYSTEM_SIZE],
         k5[SYSTEM_SIZE], k6[SYSTEM_SIZE], k7[SYSTEM_SIZE];

    // Auxiliar array to store the intermediate calls to the
    // computeComponent function
    Real y1[SYSTEM_SIZE];

    // Auxiliary variables used to compute the errors at each step.
    Real sqr;                            // Scaled differences in each eq.
    Real errors[SYSTEM_SIZE]; // Local error of each eq.
    Real err;                            // Global error of the step

    // Initial values for the step size automatic prediction variables.
    // They are basically factors to maintain the new step size in known
    // bounds, but you can see the corresponding chunk of code far below to
    // know more about the puropose of each of these variables.
    Real facold = 1.0E-4;
    Real expo1 = 0.2 - beta * 0.75;
    Real fac1_inverse = 1.0 / fac1;
    Real fac2_inverse = 1.0 / fac2;
    Real fac11, fac;

    // Loop variables initialisation. The main loop finishes when `last` is
    // set to true, event that happens when the current x0 plus the current
    // step exceeds x_{end}. Furthermore, the current step is repeated when
    // `reject` is set to true, event that happens when the global error
    // estimation exceeds 1.
    bool last  = false;
    Real reject = false;

    // Retrieve the value of h
    Real h = *hOrig;

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
        // TODO: Check that this flag is really necessary
        if (0.1 * fabs(h) <= fabs(x0) * uround){
            keepRunning = false;

            // Let the user know the computation stopped before xEnd
            *success = false;
            *hOrig = h;

            // Finish all execution in this block
            return;
        }


        // PHASE 0. Check if the current time x_0 plus the current step
        // (multiplied by a safety factor to prevent steps too small)
        // exceeds the end time x_{end}.
        if ((x0 + 1.01*h - xend) * integrationDirection > 0.0){
          h = xend - x0;
          last = true;
        }

        // PHASE 1. Compute the K1, ..., K7 components and the estimated
        // solution, using the Butcher's table described in Table 5.2 ([1])

        // K1 computation
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            y1[thread] = y0[thread];
        }
        computeComponent(x0, y1, k1, data);

        // K2 computation
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            y1[thread] = y0[thread] + h * A21 * k1[thread];
        }
        computeComponent(x0 + C2*h, y1, k2, data);

        // K3 computation
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            y1[thread] = y0[thread] + h*(A31 * k1[thread] +
                               A32 * k2[thread]);
        }
        computeComponent(x0 + C3*h, y1, k3, data);

        // K4 computation
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            y1[thread] = y0[thread] + h*(A41 * k1[thread] +
                               A42 * k2[thread] +
                               A43 * k3[thread]);
        }
        computeComponent(x0 + C4*h, y1, k4, data);

        // K5 computation
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            y1[thread] = y0[thread] + h*( A51 * k1[thread] +
                                A52 * k2[thread] +
                                A53 * k3[thread] +
                                A54 * k4[thread]);
        }
        computeComponent(x0 + C5*h, y1, k5, data);

        // K6 computation
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            y1[thread] = y0[thread] + h*(A61 * k1[thread] +
                               A62 * k2[thread] +
                               A63 * k3[thread] +
                               A64 * k4[thread] +
                               A65 * k5[thread]);
        }
        computeComponent(x0 + C6*h, y1, k6, data);

        // K7 computation.
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            y1[thread] = y0[thread] + h*(A71 * k1[thread] +
                               A73 * k3[thread] +
                               A74 * k4[thread] +
                               A75 * k5[thread] +
                               A76 * k6[thread]);
        }
        computeComponent(x0 + C7*h, y1, k7, data);

        // The Butcher's table (Table 5.2, [1]), shows that the estimated
        // solution has exactly the same coefficients as the ones used to
        // compute K7. Then, the solution is the last computed y1:
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            solution[thread] = y1[thread];
        }

        // The local error of each equation is computed as the difference
        // between the solution y and the higher order solution \hat{y}, as
        // specified in the last two rows of the Butcher's table (Table
        // 5.2, [1]). Instead of computing \hat{y} and then substract it
        // from y, the differences between the coefficientes of each
        // solution have been computed and the error is directly obtained
        // using them:
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            errors[thread] = h*(E1 * k1[thread] +
                              E3 * k3[thread] +
                              E4 * k4[thread] +
                              E5 * k5[thread] +
                              E6 * k6[thread] +
                              E7 * k7[thread]);
        }

        Real sk;
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            // The local estimated error has to satisfy the following
            // condition: |err[i]| < Atol[i] + Rtol*max(|y_0[i]|, |y_j[i]|)
            // (see equation (4.10), [1]). The variable sk stores the right
            // hand size of this inequality to use it as a scale in the local
            // error computation; this way we "normalize" the error and we can
            // compare it against 1.
            sk = atoli + rtoli*fmax(fabs(initCond[thread]), fabs(solution[thread]));

            // Compute the square of the local estimated error (scaled with the
            // previous factor), as the global error will be computed as in
            // equation 4.11 ([1]): the square root of the mean of the squared
            // local scaled errors.
            sqr = (errors[thread])/sk;
            errors[thread] = sqr*sqr;
        }

        // Add the square of the local scaled errors with the usual
        // parallel reduction technique, storing the result in errors[0]:
        // Basically, in every step of the loop, the second half of the
        // remaining array will be added to the first half, considering
        // half of the previous array at each step. Let's see it with more
        // detail:
        // Starting with the first half of the threads (the other half will
        // be idle until this process finish), each one of them will add to
        // their own local element of the errors array the element placed
        // at [its own global thread index in the block + half of the
        // number of threads]; after this is completed, only the first
        // quarter of the threads will be considered, adding to their own
        // elements the values at [its own global thread index in the block
        // + a quarter of the number of threads]. This process continues,
        // considering half of the threads considered in the previous step,
        // until there is only one working thread, the first one, that
        // makes the final addition: errors[0] = errors[0] + errors[1]
        // Note that this process forces the number of threads to be a
        // power of 2: the variable controlling which threads will work at
        // each step (the variable s in the loop), is initially set to the
        // half of the block total threads, and successively divided by 2.
        err = 0;
        for(int thread = 0; thread < SYSTEM_SIZE; thread++){
            err += errors[thread];
        }

        // The sum of the local squared errors in now in errors[0], but the
        // global error is the square root of the mean of those local
        // errors: we finish here the computation and store it in err.
        err = sqrt(err/(Real)SYSTEM_SIZE);

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
        fac = fac11 / pow(facold,beta);
        // We need the multiplying factor (always taking into account the
        // safe factor) to be between fac1 and fac2; i.e., we require
        // fac1 <= hnew/h <= fac2:
        fac = fmax(fac2_inverse, fmin(fac1_inverse, fac/safe));
        // New step final (but temporary) computation
        hnew = h / fac;

        // Check whether the normalized error, err, is below or over 1.:
        // REJECT STEP if err > 1.
        if( err > 1.){
            // Stabilization technique with the minimum and safe factors
            // when the step is rejected.
            hnew = h / fmin(fac1_inverse, fac11/safe);

            // Set reject variable to true for the next step and make sure
            // this one is not the last step!
            reject = true;
            last = false;
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
            for(int thread = 0; thread < SYSTEM_SIZE; thread++){
                y0[thread] = solution[thread];
            }

            // This step was accepted, so it was not rejected, so reject is
            // false. SCIENCE.
            reject = false;
        }

        // Final step size update!
        if(!last)
            h = hnew;
    }while(!last);

    // Finally, let the user know everything's gonna be alright
    *success = true;
    *hOrig = h;

    // Aaaaand that's all, folks! Update system value (each thread its
    // result) in the global memory :)
    for(int thread = 0; thread < SYSTEM_SIZE; thread++){
        initCond[thread] = solution[thread];
    }
}


int sign(Real x){
    return x < 0 ? -1 : +1;
}


// Given a system point, p1, and a target,
int bisect(Real* yOriginal, Real* data, Real step){
    // Set the current point to the original point and declare an array to
    // store the value of the system function
    Real* yCurrent = yOriginal;
    Real yVelocity[SYSTEM_SIZE];

    // It is necessary to maintain the previous theta to know the direction
    // change
    Real prevTheta;
    prevTheta = yCurrent[1];

    // The first step shall be to the other side and half of its length;
    step = - step / 2;

    // Loop variables, to control the inner for and to control the iterations
    // does not exceed a maximum number
    int i;
    int iter = 0;

    // This loop implements the main behaviour of the algorithm; basically,
    // this is how it works:
    //      1. It advance the point one single step with the Euler algorithm.
    //      2. If theta has crossed pi/2, it changes the direction of the
    //      new step. The magnitude of the new step is always half of the
    //      magnitude of the previous one.
    //      3. It repeats 1 and 2 until the current theta is very near of Pi/2
    //      ("very near" is defined by BISECT_TOL) or until the number of
    //      iterations exceeds a manimum number previously defined
    while(fabs(yCurrent[1] - HALF_PI) > BISECT_TOL && iter < BISECT_MAX_ITER){
        // 1. Compute value of the function in the current point
        computeComponent(0, yCurrent, yVelocity, data);

        // 1. Advance point with Euler algorithm
        // TODO: See if this is more efficient than splitting between threads
        for(i = 0; i < SYSTEM_SIZE; i++){
            yCurrent[i] = yCurrent[i] + yVelocity[i]*step;
        }

        // 2. Change the step direction whenever theta crosses the target,
        // pi/2, and make it half of the previous one.
        step = step * sign((yCurrent[1] - HALF_PI)*(prevTheta - HALF_PI)) / 2;

        prevTheta = yCurrent[1];

        iter++;
    } // 3. End of while
    return iter;
}
