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
#include <time.h>

// #define SYSTEM_SIZE 2
#define MAXBLOCKSIZE 50
// typedef double Real;

#include "../Raytracer/Kernel/common.cu"

Real Delta(Real r, Real r2){
    return r2 - 2*r + __a2;
}

Real P(Real r, Real r2, Real b){
    return r2 + __a2 - __a*b;
}

Real R(Real r, Real r2, Real b, Real q){
    Real _P = P(r, r2, b);
    Real D = Delta(r, r2);

    return _P*_P - D*((b - __a)*(b - __a) + q);
}

Real dbR(Real r, Real r2, Real b){
    return (4*b - 4*__a)*r - 2*b*r2;
}

Real drR(Real r, Real r2, Real b, Real q){
    Real bMinusA = b-__a;
    return 4*r*(r2 - __a*b + __a2) - (q + bMinusA*bMinusA)*(2*r - 2);
}

Real Theta(Real sinT2, Real cosT2, Real b2, Real q){
    return q - cosT2*(b2/sinT2 - __a2);
}

Real dbTheta(Real sinT2, Real cosT2, Real b){
    return -(2*b*cosT2)/(sinT2);
}

Real dzTheta(Real sinT, Real sinT2, Real cosT, Real cosT2, Real b2){
    Real cosT3 = cosT2*cosT;
    Real sinT3 = sinT2*sinT;

    return 2*cosT*((b2/sinT2) - __a2)*sinT + (2*b2*cosT3)/(sinT3);
}

Real drDelta(Real r){
    return 2*r - 2;
}

Real rho(Real r2, Real cosT2){
    return sqrt(r2 + __a2*cosT2);
}

Real drRho(Real r, Real r2, Real cosT2, Real rho){
    return r / rho;
}

Real dzRho(Real r2, Real sinT, Real cosT, Real cosT2, Real rho){
    return -(__a2*cosT*sinT) / rho;
}

/**
* Computes the value of the threadId-th component of the function
* F(t) = (f1(t), ..., fn(t)) and stores it in the memory pointed by f
 * @param  int   threadId      Identifier of the calling thread.
 * @param  Real  x             Value of the time in which the system is solved
 * @param  Real* y             Initial conditions for the system: a pointer to
 *                             a vector whose lenght shall be the same as the
 *                             number of equations in the system.
 * @param  Real* f             Computed value of the function: a pointer to a
 *                             vector whose lenght shall be the same as the
 *                             number of equations in the system.
 * @param  Real* data          Additional data needed by the function, managed
 *                             by the caller.
 */
void computeComponent(Real x, Real* y, Real* f, Real* data){
    Real r, r2, theta, pR, pR2, pTheta, pTheta2, b, b2, q;
    Real sinT, cosT, sinT2, cosT2;
    Real _R, D, Z, DZplusR, rho1, rho2, rho3;

    // Parallelization of the retrieval of the input data (position of the ray,
    // momenta and constants), storing it as shared variables. Furthermore,
    // some really useful numbers are computed; namely: the sine and cosine of
    // theta (and their squares) and the square of the constant b.
    // Each thread retrieves its data and make the corresponding computations,
    // except for the thread 2: the corresponging value of this thread should
    // be ray's phi, but this value is not used in the system; as this thread
    // is free to do another calculation, it retrieves the constants b,q (not
    // directly associated with any thread) and compute b**2

    r = y[0];
    r2 = r*r;
    theta = y[1];
    sinT = sin(theta);
    cosT = cos(theta);
    sinT2 = sinT*sinT;
    cosT2 = cosT*cosT;
    b = data[0];
    q = data[1];
    b2 = b*b;
    pR = y[3];
    pTheta = y[4];

    // Parallelization of the computation of somec commonly used numbers, also
    // stored as shared variables; namely: R, D, Theta (that is called Z) and
    // rho (and its square and cube). These four numbers let one thread free:
    // it is used in the computation of the squares of the momenta: pR and
    // pTheta.
    _R = R(r, r2, b, q);
    D = Delta(r, r2);
    Z = Theta(sinT2, cosT2, b2, q);
    rho1 = rho(r2, cosT2);
    rho2 = rho1*rho1;
    rho3 = rho1*rho2;
    pR2 = pR*pR;
    pTheta2 = pTheta*pTheta;

    // Declaration of variables used in the actual computation: dR, dZ, dRho
    // and dD will store the derivatives of the corresponding functions (with
    // respect to the corresponding variable in each thread). The sumX values
    // are used as intermediate steps in the final computations, in order to
    // ease notation.
    Real dR, dZ, dRho, dD, sum1, sum2, sum3, sum4, sum5, sum6;

    // Actual computation: each thread computes its corresponding value in the
    // system; namely:
    //      Thread 0 -> r
    //      Thread 1 -> theta
    //      Thread 2 -> phi
    //      Thread 3 -> pR
    //      Thread 4 -> pTheta
    f[0] = D * pR / rho2;

    f[1] = pTheta / rho2;

    // Derivatives with respect to b
    dR = dbR(r, r2, b);
    dZ = dbTheta(sinT2, cosT2, b);

    f[2] = - (dR + D*dZ)/(2*D*rho2);

    // Derivatives with respect to r
    dRho = drRho(r, r2, cosT2, rho1);
    dD = drDelta(r);
    dR = drR(r, r2, b, q);

    DZplusR = D*Z + _R;

    sum1 = + dRho*pTheta2;
    sum2 = + D*pR2*dRho;
    sum3 = - (DZplusR*dRho / D);
    sum4 = - (dD*pR2);
    sum5 = + (dD*Z + dR) / D;
    sum6 = - (dD*DZplusR / (D*D));

    f[3] = (sum1 + sum2 + sum3)/rho3 +
                  (sum4 + sum5 + sum6)/(2*rho2);

    // Derivatives with respect to theta (called z here)
    dRho = dzRho(r2, sinT, cosT, cosT2, rho1);
    dZ = dzTheta(sinT, sinT2, cosT, cosT2, b2);

    sum1 = + dRho*pTheta2;
    sum2 = + D*pR2*dRho;
    sum3 = - (D*Z + _R)*dRho / D;
    sum4 = + dZ / (2*rho2);

    f[4] = (sum1 + sum2 + sum3)/rho3 + sum4;
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
 void RK4Solve(Real originalX0, Real xend, Real* devInitCond, Real h,
                          Real hmax, int conditionsNumber, Real* data){
    #ifdef DEBUG
        printf("ThreadId %d - INITS: x0=%.20f, xend=%.20f, y0=(%.20f, %.20f)\n", threadId, *((Real*)devX0), xend, ((Real*)devInitCond)[0], ((Real*)devInitCond)[1]);
    #endif

    // Shared array between the block threads to store intermediate
    // solutions.
    Real* solution;

    // Loop variable to manage the automatic step size detection.
    // TODO: Implement the hinit method
    Real hnew;

    // Check the direction of the integration: to the future or to the past
    // and get the absolute value of the maximum step size.
    Real integrationDirection = xend - originalX0 > 0. ? +1. : -1.;
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
    // Real* atoler = (Real*) globalAtoler;
    // Real* rtoler = (Real*) globalRtoler;

    Real x0;
    for(int condition = 0; condition < conditionsNumber; condition++){
        x0 = originalX0;
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
            computeComponent(x0, y0, k1, data);

            // K2 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*(1./5.)*k1[i];

            computeComponent(x0 + (1./5.)*h, y1, k2, data);

            // K3 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*((3./40.)*k1[i] +
                                   (9./40.)*k2[i]);

            computeComponent(x0 + (3./10.)*h, y1, k3, data);


            // K4 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*(  (44./45.)*k1[i]
                                   - (56./15.)*k2[i]
                                   + (32./9.)*k3[i]);

            computeComponent(x0 + (4./5.)*h, y1, k4, data);


            // K5 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*( (19372./6561.)*k1[i]
                                  - (25360./2187.)*k2[i]
                                  + (64448./6561.)*k3[i]
                                  - (212./729.)*k4[i]);

            computeComponent(x0 + (8./9.)*h, y1, k5, data);


            // K6 computation
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*((9017./3168.)*k1[i]
                                 - (355./33.)*k2[i]
                                 + (46732./5247.)*k3[i]
                                 + (49./176.)*k4[i]
                                 - (5103./18656.)*k5[i]);

            computeComponent(x0 + h, y1, k6, data);


            // K7 computation.
            for(int i=0; i<SYSTEM_SIZE; i++)
                y1[i] = y0[i] + h*((35./384.)*k1[i]
                                 + (500./1113.)*k3[i]
                                 + (125./192.)*k4[i]
                                 - (2187./6784.)*k5[i]
                                 + (11./84.)*k6[i]);

            computeComponent(x0 + h, y1, k7, data);


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
                 Real sk = atoli + rtoli*fmax(abs(y0[i]), abs(solution[i]));

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
            initCond[i] = solution[i];
        }
    } // Initial conditions for
}

int main(int argc, char* argv[]){
	fprintf(stderr,"Holi");
    Real initCond[MAXBLOCKSIZE * MAXBLOCKSIZE * SYSTEM_SIZE * sizeof(Real)];
    // Real globalRtoler[SYSTEM_SIZE];
    // Real globalAtoler[SYSTEM_SIZE];

    Real data[2];
    data[0] = -3.81716582717122587809;
    data[1] = 58.28301980847918173367;

    // for(int i=0; i<SYSTEM_SIZE; i++){
    //     globalAtoler[i] = 1e-12;
    // //     globalRtoler[i] = 1e-6;
    // }

    Real h = 0.001;
    // Real safe=0.9;
    // Real fac1=0.2;
    // Real fac2=10.0;
    // Real beta=0.04;
    // Real uround=2.3e-16;

    clock_t start, end;

    Real x0 = -10.;
    Real xend = -5.;
    int initCondsNumber;

    for(int blockSize = 1; blockSize <= MAXBLOCKSIZE; blockSize++) {
        initCondsNumber = blockSize*blockSize;

        for(int i=0; i<initCondsNumber*SYSTEM_SIZE; i+=SYSTEM_SIZE){
            initCond[i+0] = 20.;
            initCond[i+1] = 1.94134631163816862021;
            initCond[i+2] = 2.94610185334111251976;
            initCond[i+3] = 1.01597063355188099720;
            initCond[i+4] = -7.63433165434140548200;
        }

        start = clock();
        RK4Solve(x0, xend, initCond, h, xend-x0, initCondsNumber, data);
        end = clock();

        Real timeEx = (end-start)/(double)CLOCKS_PER_SEC;

        printf("%f\n", timeEx);
        fprintf(stderr, "%d: %f\n", blockSize, timeEx);
    }
}
