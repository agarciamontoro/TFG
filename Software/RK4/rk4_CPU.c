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
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define SYSTEM_SIZE 2

typedef double Real;

/**
 * Computes the value of the threadId-th component of the function
 * F(t) = (f1(t), ..., fn(t)) and stores it in the memory pointed by f
 * @param Real  x  Value of the time in which the system is solved
 * @param Real* y  Initial conditions for the system: a pointer to a vector
 *                 whose lenght shall be the same as the number of equations in
 *                 the system.
 * @param Real* f  Computed value of the function: a pointer to a vector whose
 *                 lenght shall be the same as the number of equations in the
 *                 system.
 */
void computeComponent(Real x, Real* y, Real* f){
    // Jinja template that renders to a switch in which every thread computes
    // a different equation and stores it in the corresponding position in f.
    // If you want to hard-code the system function manually, fill the switch
    // cases with the right hand side of the n equations: the i-th equation has
    // to be defined in the `case i-1:` and stored in the i-1 position of the
    // output vector f.
    // Please, note that this is a parallelized code, so it is mandatory to
    // follow the explained structure in order to successfully manage the local
    // and global work of the threads.
    //
    // Example: If you want to solve an harmonic oscillator system as the
    // following:
    //      y_0' = y_1
    //      y_1' = -5*y_0
    // then the code in the body of this function will read as follows:
    //      switch(threadId){
    //          case 0:
    //              f[0] = y[1];
    //              break;
    //          case 1:
    //              f[1] = -5 * y[0];
    //              break;
    //      }
    f[0] = y[1];
    f[1] = x*y[0];

}

/**
 * Applies the DOPRI5 algorithm over the system defined in the computeComponent
 * function, using the initial conditions specified in devX0 and devInitCond,
 * and returning the solution found at xend.
 *
 * @param[in,out]  void* devX0        Device pointer to a Real specifying the
 *                       start of the integration interval [x_0, x{end}]. The
 *                       last x for which the system is computed (if there is
 *                       no problem, it should be x_{end}) replaces this value
 *                       at the end of the algorithm, in order to let the user
 *                       know the finish place of the integration process.
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
 void RK4Solve(Real* devX0, Real xend, Real* devInitCond, Real h,
                          Real hmax, Real* globalRtoler, Real* globalAtoler, Real safe, Real fac1, Real fac2, Real beta,
                          Real uround, int conditionsNumber){
    #ifdef DEBUG
        printf("ThreadId %d - INITS: x0=%.20f, xend=%.20f, y0=(%.20f, %.20f)\n", threadId, *((Real*)devX0), xend, ((Real*)devInitCond)[0], ((Real*)devInitCond)[1]);
    #endif

    // Shared array between the block threads to store intermediate
    // solutions.
    Real* solution;

    // Loop variable to manage the automatic step size detection.
    // TODO: Implement the hinit method
    Real hnew;

    // Retrieve the initial time and store it in x0.
    Real x0 = *devX0;

    // Check the direction of the integration: to the future or to the past
    // and get the absolute value of the maximum step size.
    Real integrationDirection = xend - x0 > 0. ? +1. : -1.;
    hmax = abs(hmax);

    Real y0[SYSTEM_SIZE];
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

    // Retrieve the absolute and relative error tolerances (see the
    // function header's comment to know their purpose) provided to predict
    // the step size and get the ones associated to this only thread.
    Real* atoler = (Real*) globalAtoler;
    Real* rtoler = (Real*) globalRtoler;

    for(int condition = 0; condition < conditionsNumber; condition++){
        // Retrieve the position where the initial conditions this block will
        // work with are.
        // Each block, absolutely identified in the grid by blockId, works with
        // only one initial condition (that has N elements, as N equations are
        // in the system). Then, the position of where these initial conditions
        // are stored in the serialized vector can be computed as blockId * N.
        Real* initCond = (Real*)devInitCond + condition*SYSTEM_SIZE;
        for(int i=0; i<SYSTEM_SIZE; i++)
            y0[i] = initCond[i];

        // Auxiliary variables used to compute the errors at each step.
        Real sqr;     // Scaled differences in each eq.
        Real localError;  // Local error of each eq.
        Real err;                  // Global error of the step

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
            // TODO: Check that the step size is not too small

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
            computeComponent(x0, y0, k1);

            // K2 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*(1./5.)*k1[i];

            computeComponent(x0 + (1./5.)*h, y1, k2);

            // K3 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*((3./40.)*k1[i] +
                                   (9./40.)*k2[i]);

            computeComponent(x0 + (3./10.)*h, y1, k3);


            // K4 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*(  (44./45.)*k1[i]
                                   - (56./15.)*k2[i]
                                   + (32./9.)*k3[i]);

            computeComponent(x0 + (4./5.)*h, y1, k4);


            // K5 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*( (19372./6561.)*k1[i]
                                  - (25360./2187.)*k2[i]
                                  + (64448./6561.)*k3[i]
                                  - (212./729.)*k4[i]);

            computeComponent(x0 + (8./9.)*h, y1, k5);


            // K6 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*((9017./3168.)*k1[i]
                                 - (355./33.)*k2[i]
                                 + (46732./5247.)*k3[i]
                                 + (49./176.)*k4[i]
                                 - (5103./18656.)*k5[i]);

            computeComponent(x0 + h, y1, k6);


            // K7 computation.
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*((35./384.)*k1[i]
                                 + (500./1113.)*k3[i]
                                 + (125./192.)*k4[i]
                                 - (2187./6784.)*k5[i]
                                 + (11./84.)*k6[i]);

            computeComponent(x0 + h, y1, k7);


            // The Butcher's table (Table 5.2, [1]), shows that the estimated
            // solution has exactly the same coefficients as the ones used to
            // compute K7. Then, the solution is the last computed y1:
            solution = y1;

            // The local error of each equation is computed as the difference
            // between the solution y and the higher order solution \hat{y}, as
            // specified in the last two rows of the Butcher's table (Table
            // 5.2, [1]). Instead of computing \hat{y} and then substract it
            // from y, the differences between the coefficientes of each
            // solution have been computed and the error is directly obtained
            // using them:
            err = 0.;
            for(int i=0; i<SYSTEM_SIZE; i++){
                localError = h*((71./57600.)*k1[i]
                             - (71./16695.)*k3[i]
                             + (71./1920.)*k4[i]
                             - (17253./339200.)*k5[i]
                             + (22./525.)*k6[i]
                             - (1./40.)*k7[i]);

                 // The local estimated error has to satisfy the following
                 // condition: |err[i]| < Atol[i] + Rtol*max(|y_0[i]|,
                 // |y_j[i]|) (see equation (4.10), [1]). The variable sk
                 // stores the right hand size of this inequality to use it as
                 // a scale in the local error computation; this way we
                 // "normalize" the error and we can compare it against 1.
                 Real sk = atoler[i] + rtoler[i]*fmax(abs(y0[i]), abs(solution[i]));

                 // Compute the square of the local estimated error (scaled
                 // with the previous factor), as the global error will be
                 // computed as in equation 4.11 ([1]): the square root of the
                 // mean of the squared local scaled errors.
                 sqr = (localError)/sk;
                 err += sqr*sqr;
            }

            #ifdef DEBUG
                printf("ThreadId %d - K 1-7: K1:%.20f, K2:%.20f, K3:%.20f, K4:%.20f, K5:%.20f, K6:%.20f, K7:%.20f\n", threadId, k1[threadId], k2[threadId], k3[threadId], k4[threadId], k5[threadId], k6[threadId], k7[threadId]);
                printf("ThreadId %d - Local: sol: %.20f, error: %.20f\n", threadId, solution[threadId], errors[threadId]);
            #endif

            #ifdef DEBUG
                printf("ThreadId %d - Diffs: sqr: %.20f, sk: %.20f\n", threadId, sqr, sk);
            #endif

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

            #ifdef DEBUG
                printf("ThreadId %d - H aux: expo1: %.20f, err: %.20f, fac11: %.20f, facold: %.20f, fac: %.20f\n", threadId, expo1, err, fac11, facold, fac);
                printf("ThreadId %d - H new: prevH: %.20f, newH: %.20f\n", threadId, hnew);
            #endif

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
                for(int i=0; i<SYSTEM_SIZE; i++)
                    y0[i] = solution[i];

                // This step was accepted, so it was not rejected, so reject is
                // false. SCIENCE.
                reject = false;
            }

            // Final step size update!
            h = hnew;

            #ifdef DEBUG
                if(threadId == 0){
                    if(err > 1.){
                        printf("\n###### CHANGE: err: %.20f, h: %.20f\n\n", err, h);
                    }
                    else{
                        printf("\n###### ======:  err: %.20f, h: %.20f\n\n", err, h);
                    }
                }
            #endif
        }while(!last);

        // Aaaaand that's all, folks! Update system value (each thread its
        // result) and system time (do it just once) in the global memory :)
        for(int i=0; i<SYSTEM_SIZE; i++){
            devInitCond[i] = solution[i];
        }
        *devX0 = x0;
    } // Initial conditions for
}

int main(){
    int initCondsNumber = 1;
    Real* initCond = (Real*) malloc(initCondsNumber *
                                    SYSTEM_SIZE*sizeof(Real));

    Real globalRtoler[SYSTEM_SIZE];
    Real globalAtoler[SYSTEM_SIZE];

    for(int i=0; i<SYSTEM_SIZE; i++){
        globalAtoler[i] = 1e-12;
        globalRtoler[i] = 1e-6;
    }

    Real h = 0.001;
    Real safe=0.9;
    Real fac1=0.2;
    Real fac2=10.0;
    Real beta=0.04;
    Real uround=2.3e-16;

    initCond[0] = 1.;
    initCond[1] = 1.;

    Real x0 = -10.;
    Real xend = -5.;

    RK4Solve(&x0, xend, initCond, h, xend-x0, globalRtoler, globalAtoler,
             safe, fac1, fac2, beta, uround, initCondsNumber);

    printf("%.8f: [%.8f %.8f]\n", x0, initCond[0], initCond[1]);
}
