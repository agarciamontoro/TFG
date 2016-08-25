#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "Raytracer/Kernel/common.cu"

// __device__ Real Delta(Real r, Real r2){
//     return r2 - 2*r + __a2;
// }
//
// __device__ Real R(Real r, Real r2, Real b, Real q, Real D){
//     Real _P = r2 + __a2 - __a*b;
//     Real bMinusA = b - __a;
//
//     return _P*_P - D*(bMinusA*bMinusA + q);
// }
//
// __device__ Real dbR(Real r, Real r2, Real b){
//     return (4*b - 4*__a)*r - 2*b*r2;
// }
//
// __device__ Real drR(Real r, Real r2, Real b, Real q){
//     Real bMinusA = b - __a;
//     return 4*r*(r2 - __a*b + __a2) - (q + bMinusA*bMinusA)*(2*r - 2);
// }
//
// __device__ Real Theta(Real sinT2Inv, Real cosT2, Real b2, Real q){
//     return q - cosT2*(b2*sinT2Inv - __a2);
// }
//
// __device__ Real dbTheta(Real sinT2Inv, Real cosT2, Real b){
//     return - 2 * b * cosT2 * sinT2Inv;
// }
//
// __device__ Real dzTheta(Real sinT, Real sinT2, Real sinT2Inv, Real cosT, Real cosT2, Real b2){
//     Real cosT3 = cosT2*cosT;
//     Real sinT3 = sinT2*sinT;
//
//     return 2*cosT*((b2*sinT2Inv) - __a2)*sinT + (2*b2*cosT3)/(sinT3);
// }
//
// __device__ Real drDelta(Real r){
//     return 2*r - 2;
// }
//
// __device__ Real rhoSquaredInv(Real r2, Real cosT2){
//     return 1/(r2 + __a2*cosT2);
// }
//
// __device__ Real dzRhoTimesRho(Real r2, Real sinT, Real cosT,
//                                      Real cosT2){
//     return - __a2*cosT*sinT;
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
    // Variables to hold the position of the ray, its momenta and related
    // operations between them and the constant a, which is the spin of the
    // black hole.
    Real r, r2, twor, theta, pR, pR2, pTheta, pTheta2, b, twob, b2, q, bMinusA;

    // Variables to hold the sine and cosine of theta, along with some
    // operations with them
    Real sinT, cosT, sinT2, sinT2Inv, cosT2;

    // Variables to hold the value of the functions P, R, Delta (which is
    // called D), Theta (which is called Z) and rho, along with some operations
    // involving these values.
    Real P, R, D, Dinv, Z, DZplusR, rho2Inv, twoRho2Inv, rho4Inv;

    // Retrieval of the input data (position of the ray, momenta and
    // constants).
    r = y[0];
    theta = y[1];
    pR = y[3];
    pTheta = y[4];

    // Computation of the square of r, widely used in the computations.
    r2 = r*r;

    // Sine and cosine of theta, as well as their squares and inverses.
    sincos(theta, &sinT, &cosT);
    sinT2 = sinT*sinT;
    sinT2Inv = 1/sinT2;
    cosT2 = cosT*cosT;

    // Retrieval of the constants data: b and q, along with the computation of
    // the square of b and the number b - a, repeateadly used throughout the
    // computation
    b = data[0];
    q = data[1];

    b2 = b*b;
    bMinusA = b - __a;

    // Commonly used variables: R, D, Theta (that is called Z) and
    // rho (and its square and cube).
    D = r2 - 2*r + __a2;
    Dinv = 1/D;

    P = r2 - __a * bMinusA;
    R = P*P - D*(bMinusA*bMinusA + q);

    Z = q - cosT2*(b2*sinT2Inv - __a2);

    rho2Inv = 1/(r2 + __a2*cosT2);
    twoRho2Inv = rho2Inv/2;
    rho4Inv = rho2Inv*rho2Inv;

    // Squares of the momenta components
    pR2 = pR*pR;
    pTheta2 = pTheta*pTheta;

    // Double b and double r, that's it! :)
    twob = 2*b;
    twor = 2*r;

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
    dR = 4*bMinusA*r - twob*r2;
    dZ = - twob * cosT2 * sinT2Inv;

    f[2] = - (dR + D*dZ)*Dinv*twoRho2Inv;

    // *********************** EQUATION 4 *********************** //
    // Derivatives with respect to r
    dD = twor - 2;
    dR = 2*twor*(r2 - __a*bMinusA) - (q + bMinusA*bMinusA)*(twor - 2);

    DZplusR = D*Z + R;

    sum1 = + pTheta2;
    sum2 = + D*pR2;
    sum3 = - (DZplusR * Dinv);
    sum4 = - (dD*pR2);
    sum5 = + (dD*Z + dR) * Dinv;
    sum6 = - (dD*DZplusR * Dinv * Dinv);

    f[3] = r*(sum1 + sum2 + sum3)*rho4Inv + (sum4 + sum5 + sum6)*twoRho2Inv;

    // *********************** EQUATION 5 *********************** //
    // Derivatives with respect to theta (called z here)
    dRhoTimesRho = - __a2*cosT*sinT;

    Real cosT3 = cosT2*cosT;
    Real sinT3 = sinT2*sinT;

    dZ = 2*cosT*((b2*sinT2Inv) - __a2)*sinT + (2*b2*cosT3)/(sinT3);

    sum1 = + pTheta2;
    sum2 = + D*pR2;
    sum3 = - DZplusR * Dinv;
    sum4 = + dZ * twoRho2Inv;

    f[4] = dRhoTimesRho*(sum1 + sum2 + sum3)*rho4Inv + sum4;
}

#endif // __FUNCTIONS__
