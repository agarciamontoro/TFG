#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "Raytracer/Kernel/common.cu"

__device__ inline Real Delta(Real r, Real r2){
    return r2 - 2*r + __a2;
}

__device__ inline Real R(Real r, Real r2, Real b, Real q, Real D){
    Real _P = r2 + __a2 - __a*b;
    Real bMinusA = b - __a;

    return _P*_P - D*(bMinusA*bMinusA + q);
}

__device__ inline Real dbR(Real r, Real r2, Real b){
    return (4*b - 4*__a)*r - 2*b*r2;
}

__device__ inline Real drR(Real r, Real r2, Real b, Real q){
    Real bMinusA = b - __a;
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

__device__ inline Real dzRhoTimesRho(Real r2, Real sinT, Real cosT,
                                     Real cosT2){
    return - __a2*cosT*sinT;
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
    D = Delta(r, r2);
    _R = R(r, r2, b, q, D);
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
    dD = drDelta(r);
    dR = drR(r, r2, b, q);

    DZplusR = D*Z + _R;

    sum1 = + pTheta2;
    sum2 = + D*pR2;
    sum3 = - (DZplusR * Dinv);
    sum4 = - (dD*pR2);
    sum5 = + (dD*Z + dR) * Dinv;
    sum6 = - (dD*DZplusR * Dinv * Dinv);

    f[3] = r*(sum1 + sum2 + sum3)*rho4Inv + (sum4 + sum5 + sum6)*twoRho2Inv;

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
