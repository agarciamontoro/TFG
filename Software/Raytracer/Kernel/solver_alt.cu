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

#include "Raytracer/Kernel/common.cu"
#include "Raytracer/Kernel/functions.cu"

// hmax has to be positive
static inline __device__ Real getStepSize(Real* pos, Real* vel, Real hmax){
    if(pos[0] < horizonRadius + SOLVER_EPSILON){
        return 0; // too close to the black hole
    }

    // See equation (2) in Chi-kwan Chan's paper
    float f1 = SOLVER_DELTA / (fabs(vel[0] / pos[0]) + fabs(vel[1]) +
                               fabs(vel[2]));

    float f2 = (pos[0] - horizonRadius) / fabs((2*vel[0]));

    return fmin(hmax, fmin(f1, f2));
}



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
 __device__ SolverStatus RK4Solve_ALT(Real x0, Real xend, Real* initCond,
                          Real* hOrig, Real hmax, Real* data){
    #ifdef DEBUG
        printf("ThreadId %d - INITS: x0=%.20f, xend=%.20f, y0=(%.20f, %.20f)\n", threadId, x0, xend, ((Real*)initCond)[0], ((Real*)initCond)[1]);
    #endif

    // Each equation to solve has a thread that compute its solution. Although
    // in an ideal situation the number of threads is exactly the same as the
    // number of equations, it is usual that the former is greater than the
    // latter. The reason is that the number of threads has to be a power of 2
    // (see reduction technique in the only loop you will find in this function
    // code to know why): then, we can give some rest to the threads that exceeds the number of equations :)

    // Shared array between the block threads to store intermediate
    // solutions.
    Real solution[SYSTEM_SIZE];

    // Loop variable to manage the automatic step size detection.
    // TODO: Implement the hinit method
    Real hnew;

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

    int iterations = 0;

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
        iterations += 1;
        // PHASE 1. Compute the K1, ..., K7 components and the estimated
        // solution, using the Butcher's table described in Table 5.2 ([1])

        // K1 computation
        computeComponent(y0, k1, data);

        // Compute the step size
        h = integrationDirection * getStepSize(y0, k1, fabs(hmax));

        // PHASE 0. Check if the current time x_0 plus the current step
        // (multiplied by a safety factor to prevent steps too small)
        // exceeds the end time x_{end}.
        if ((x0 + /*1.01**/h - xend) * integrationDirection > 0.0){
            h = xend - x0;
            last = true;
        }

        // See if we've collided with the horizon
        if(h == 0){
            return RK45_FAILURE;
        }

        // if(blockIdx.x == 23 && blockIdx.y == 55 && threadIdx.x == 6 && threadIdx.y == 0)
        // printf("(%d, %d, %d, %d), x: %.30f, xend: %.30f, h: %.30f, hmax: %.30f, r: %.30f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x0, xend, h, hmax, y0[0]);

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

        // if(blockIdx.x == 43 && blockIdx.y == 124 && threadIdx.x == 0 && threadIdx.y == 0)
        //     printf("x: %.30f, r: %.30f\n", x0, y0[0]);

    }while(!last && (xend - x0) > 0.0);


    // printf("(%d, %d, %d, %d), Iterations: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, iterations);
    // Finally, let the user know everything's gonna be alright
    // *success = true;
    *hOrig = h;

    // Aaaaand that's all, folks! Update system value (each thread its
    // result) in the global memory :)
    for(int i = 0; i < SYSTEM_SIZE; i++){
        initCond[i] = y0[i];
    }

    return RK45_SUCCESS;
}


// __device__ int sign(Real x){
//     return x < 0 ? -1 : +1;
// }


// // Given a system point, p1, and a target,
// __device__ int bisect(Real* yOriginal, Real* data, Real step){
//     // Set the current point to the original point and declare an array to
//     // store the value of the system function
//     Real* yCurrent = yOriginal;
//     Real yVelocity[SYSTEM_SIZE];
//
//     // It is necessary to maintain the previous theta to know the direction
//     // change
//     Real prevTheta;
//     prevTheta = yCurrent[1];
//
//     // The first step shall be to the other side and half of its length;
//     step = - step * 0.5;
//
//     // Loop variables, to control the inner for and to control the iterations
//     // does not exceed a maximum number
//     int i;
//     int iter = 0;
//
//     // This loop implements the main behaviour of the algorithm; basically,
//     // this is how it works:
//     //      1. It advance the point one single step with the Euler algorithm.
//     //      2. If theta has crossed pi/2, it changes the direction of the
//     //      new step. The magnitude of the new step is always half of the
//     //      magnitude of the previous one.
//     //      3. It repeats 1 and 2 until the current theta is very near of Pi/2
//     //      ("very near" is defined by BISECT_TOL) or until the number of
//     //      iterations exceeds a manimum number previously defined
//     while(fabs(yCurrent[1] - HALF_PI) > BISECT_TOL && iter < BISECT_MAX_ITER){
//         // 1. Compute value of the function in the current point
//         computeComponent(yCurrent, yVelocity, data);
//
//         // 1. Advance point with Euler algorithm
//         // TODO: See if this is more efficient than splitting between threads
//         for(i = 0; i < SYSTEM_SIZE; i++){
//             yCurrent[i] = yCurrent[i] + yVelocity[i]*step;
//         }
//
//         // 2. Change the step direction whenever theta crosses the target,
//         // pi/2, and make it half of the previous one.
//         step = step * sign((yCurrent[1] - HALF_PI)*(prevTheta - HALF_PI)) * 0.5;
//
//         prevTheta = yCurrent[1];
//
//         iter++;
//     } // 3. End of while
//     return iter;
// }
