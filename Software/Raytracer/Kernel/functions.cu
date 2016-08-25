#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "Raytracer/Kernel/common.cu"

__device__ inline Real Delta(Real r, Real r2){
    return r2 - 2*r + __a2;
}

__device__ inline Real P(Real r, Real r2, Real b){
    return r2 + __a2 - __a*b;
}

__device__ inline Real R(Real r, Real r2, Real b, Real q){
    Real _P = P(r, r2, b);
    Real D = Delta(r, r2);

    return _P*_P - D*((b - __a)*(b - __a) + q);
}

__device__ inline Real dbR(Real r, Real r2, Real b){
    return (4*b - 4*__a)*r - 2*b*r2;
}

__device__ inline Real drR(Real r, Real r2, Real b, Real q){
    Real bMinusA = b-__a;
    return 4*r*(r2 - __a*b + __a2) - (q + bMinusA*bMinusA)*(2*r - 2);
}

__device__ inline Real Theta(Real sinT2Inv, Real cosT2, Real b2, Real q){
    return q - cosT2*(b2*sinT2Inv - __a2);
}

__device__ inline Real dbTheta(Real sinT2Inv, Real cosT2, Real b){
    return - 2 * b * cosT2 * sinT2Inv;
}

__device__ inline Real dzTheta(Real sinT, Real sinT2, Real sinT2Inv, Real cosT, Real cosT2, Real b2){
    Real cosT3 = cosT2*cosT;
    Real sinT3 = sinT2*sinT;

    return 2*cosT*((b2*sinT2Inv) - __a2)*sinT + (2*b2*cosT3)/(sinT3);
}

__device__ inline Real drDelta(Real r){
    return 2*r - 2;
}

__device__ inline Real rhoSquaredInv(Real r2, Real cosT2){
    return 1/(r2 + __a2*cosT2);
}

__device__ inline Real drRhoTimesRho(Real r, Real r2, Real cosT2){
    return r;
}

__device__ inline Real dzRhoTimesRho(Real r2, Real sinT, Real cosT,
                                     Real cosT2){
    return - __a2*cosT*sinT;
}

// /**
// * Computes the value of the threadId-th component of the function
// * F(t) = (f1(t), ..., fn(t)) and stores it in the memory pointed by f
//  * @param  int   threadId      Identifier of the calling thread.
//  * @param  Real  x             Value of the time in which the system is solved
//  * @param  Real* y             Initial conditions for the system: a pointer to
//  *                             a vector whose lenght shall be the same as the
//  *                             number of equations in the system.
//  * @param  Real* f             Computed value of the function: a pointer to a
//  *                             vector whose lenght shall be the same as the
//  *                             number of equations in the system.
//  * @param  Real* data          Additional data needed by the function, managed
//  *                             by the caller.
//  */
// __device__ void computeComponent(int threadId, Real x, Real* y, Real* f,
//                                  Real* data){
//     __shared__ Real r, r2, theta, pR, pR2, pTheta, pTheta2, b, b2, q;
//     __shared__ Real sinT, cosT, sinT2, cosT2;
//     __shared__ Real _R, D, Z, DZplusR, rho1, rho2, rho3;
//
//     // Parallelization of the retrieval of the input data (position of the ray,
//     // momenta and constants), storing it as shared variables. Furthermore,
//     // some really useful numbers are computed; namely: the sine and cosine of
//     // theta (and their squares) and the square of the constant b.
//     // Each thread retrieves its data and make the corresponding computations,
//     // except for the thread 2: the corresponging value of this thread should
//     // be ray's phi, but this value is not used in the system; as this thread
//     // is free to do another calculation, it retrieves the constants b,q (not
//     // directly associated with any thread) and compute b**2
//     switch(threadId){
//         case 0:
//             r = y[0];
//             r2 = r*r;
//             break;
//
//         case 1:
//             theta = y[1];
//             sinT = sin(theta);
//             cosT = cos(theta);
//             sinT2 = sinT*sinT;
//             cosT2 = cosT*cosT;
//             break;
//
//         case 2:
//             b = data[0];
//             q = data[1];
//             b2 = b*b;
//             break;
//
//         case 3:
//             pR = y[3];
//             break;
//
//         case 4:
//             pTheta = y[4];
//             break;
//     }
//     __syncthreads();
//
//     // Parallelization of the computation of somec commonly used numbers, also
//     // stored as shared variables; namely: R, D, Theta (that is called Z) and
//     // rho (and its square and cube). These four numbers let one thread free:
//     // it is used in the computation of the squares of the momenta: pR and
//     // pTheta.
//     switch(threadId){
//         case 0:
//             _R = R(r, r2, b, q);
//             break;
//
//         case 1:
//             D = Delta(r, r2);
//             break;
//
//         case 2:
//             Z = Theta(sinT2, cosT2, b2, q);
//             break;
//
//         case 3:
//             rho1 = rho(r2, cosT2);
//             rho2 = rho1*rho1;
//             rho3 = rho1*rho2;
//             break;
//
//         case 4:
//             pR2 = pR*pR;
//             pTheta2 = pTheta*pTheta;
//             break;
//     }
//     __syncthreads();
//
//     // Declaration of variables used in the actual computation: dR, dZ, dRho
//     // and dD will store the derivatives of the corresponding functions (with
//     // respect to the corresponding variable in each thread). The sumX values
//     // are used as intermediate steps in the final computations, in order to
//     // ease notation.
//     Real dR, dZ, dRho, dD, sum1, sum2, sum3, sum4, sum5, sum6;
//
//     // Actual computation: each thread computes its corresponding value in the
//     // system; namely:
//     //      Thread 0 -> r
//     //      Thread 1 -> theta
//     //      Thread 2 -> phi
//     //      Thread 3 -> pR
//     //      Thread 4 -> pTheta
//     switch(threadId) {
//             case 0:
//                 f[threadId] = D * pR / rho2;
//                 break;
//
//             case 1:
//                 f[threadId] = pTheta / rho2;
//                 break;
//
//             case 2:
//                 // Derivatives with respect to b
//                 dR = dbR(r, r2, b);
//                 dZ = dbTheta(sinT2, cosT2, b);
//
//                 f[threadId] = - (dR + D*dZ)/(2*D*rho2);
//                 break;
//
//             case 3:
//                 // Derivatives with respect to r
//                 dRho = drRho(r, r2, cosT2, rho1);
//                 dD = drDelta(r);
//                 dR = drR(r, r2, b, q);
//
//                 DZplusR = D*Z + _R;
//
//                 sum1 = + dRho*pTheta2;
//                 sum2 = + D*pR2*dRho;
//                 sum3 = - (DZplusR*dRho / D);
//                 sum4 = - (dD*pR2);
//                 sum5 = + (dD*Z + dR) / D;
//                 sum6 = - (dD*DZplusR / (D*D));
//
//                 f[threadId] = (sum1 + sum2 + sum3)/rho3 +
//                               (sum4 + sum5 + sum6)/(2*rho2);
//                 break;
//
//             case 4:
//                 // Derivatives with respect to theta (called z here)
//                 dRho = dzRho(r2, sinT, cosT, cosT2, rho1);
//                 dZ = dzTheta(sinT, sinT2, cosT, cosT2, b2);
//
//                 sum1 = + dRho*pTheta2;
//                 sum2 = + D*pR2*dRho;
//                 sum3 = - (D*Z + _R)*dRho / D;
//                 sum4 = + dZ / (2*rho2);
//
//                 f[threadId] = (sum1 + sum2 + sum3)/rho3 + sum4;
//                 break;
//     }
// }



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
__device__ void computeComponent(int threadId, Real x, Real* y, Real* f,
                                 Real* data){
    Real r, r2, theta, pR, pR2, pTheta, pTheta2, b, b2, q;
    Real sinT, cosT, sinT2, sinT2Inv, cosT2;
    Real _R, D, Dinv, Z, DZplusR, rho2Inv, twoRho2Inv, rho4Inv;

    // Retrieval of the input data (position of the ray, momenta and
    // constants).
    r = y[0];
    theta = y[1];
    pR = y[3];
    pTheta = y[4];

    // Computation of the square of r, widely used in the computations.
    r2 = r*r;

    // Sine and cosine of theta, as well as their squares.
    sincos(theta, &sinT, &cosT);
    sinT2 = sinT*sinT;
    sinT2Inv = 1/sinT2;
    cosT2 = cosT*cosT;

    // Retrieval of the constants data: b and q, along with the computation of
    // the square of b
    b = data[0];
    q = data[1];

    b2 = b*b;

    // Commonly used variables: R, D, Theta (that is called Z) and
    // rho (and its square and cube).
    _R = R(r, r2, b, q);
    D = Delta(r, r2);
    Dinv = 1/D;
    Z = Theta(sinT2Inv, cosT2, b2, q);

    rho2Inv = rhoSquaredInv(r2, cosT2);
    twoRho2Inv = rho2Inv/2;
    rho4Inv = rho2Inv*rho2Inv;

    // Squares of the momenta components
    pR2 = pR*pR;
    pTheta2 = pTheta*pTheta;

    // Declaration of variables used in the actual computation: dR, dZ, dRho
    // and dD will store the derivatives of the corresponding functions (with
    // respect to the corresponding variable in each thread). The sumX values
    // are used as intermediate steps in the final computations, in order to
    // ease notation.
    Real dR, dZ, dRhoTimesRho, dD, sum1, sum2, sum3, sum4, sum5, sum6;

    // *********************** EQUATION 1 *********************** //
    f[0] = D * pR * rho2Inv;

    // *********************** EQUATION 2 *********************** //
    f[1] = pTheta * rho2Inv;

    // *********************** EQUATION 3 *********************** //
    // Derivatives with respect to b
    dR = dbR(r, r2, b);
    dZ = dbTheta(sinT2Inv, cosT2, b);

    f[2] = - (dR + D*dZ)*Dinv*twoRho2Inv;

    // *********************** EQUATION 4 *********************** //
    // Derivatives with respect to r
    dRhoTimesRho = drRhoTimesRho(r, r2, cosT2);
    dD = drDelta(r);
    dR = drR(r, r2, b, q);

    DZplusR = D*Z + _R;

    sum1 = + pTheta2;
    sum2 = + D*pR2;
    sum3 = - (DZplusR * Dinv);
    sum4 = - (dD*pR2);
    sum5 = + (dD*Z + dR) * Dinv;
    sum6 = - (dD*DZplusR * Dinv * Dinv);

    f[3] = dRhoTimesRho*(sum1 + sum2 + sum3)*rho4Inv + (sum4 + sum5 + sum6)*twoRho2Inv;

    // *********************** EQUATION 5 *********************** //
    // Derivatives with respect to theta (called z here)
    dRhoTimesRho = dzRhoTimesRho(r2, sinT, cosT, cosT2);
    dZ = dzTheta(sinT, sinT2, sinT2Inv, cosT, cosT2, b2);

    sum1 = + pTheta2;
    sum2 = + D*pR2;
    sum3 = - DZplusR * Dinv;
    sum4 = + dZ * twoRho2Inv;

    f[4] = dRhoTimesRho*(sum1 + sum2 + sum3)*rho4Inv + sum4;
}

#endif // __FUNCTIONS__
