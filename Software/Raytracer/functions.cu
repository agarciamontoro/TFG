#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "Raytracer/definitions.cu"
#include "Raytracer/numericalMethods.cu"

__device__ Real Delta(Real r){
    return r*r - 2*r + __a2;
}

__device__ Real P(Real r, Real b){
    return r*r + __a2 - __a*b;
}

__device__ Real R(Real r, Real b, Real q){
    Real _P = P(r, b);
    Real D = Delta(r);

    return _P*_P - D*((b - __a)*(b - __a) + q);
}

__device__ Real dbR(Real r, Real b){
    return (4*b - 4*__a)*r - 2*b*r*r;
}

__device__ Real drR(Real r, Real b, Real q){
    return 4*r*(r*r - __a*b + __a2) - (q + (b-__a)*(b-__a))*(2*r - 2);
}

__device__ Real Theta(Real theta, Real b, Real q){
    Real sinTheta = sin(theta);
    Real sin2 = sinTheta*sinTheta;

    Real cosTheta = cos(theta);
    Real cos2 = cosTheta*cosTheta;

    return q - cos2*(b*b/sin2 - __a2);
}

__device__ Real dbTheta(Real theta, Real b){
    Real cosT = cos(theta);
    Real sinT = sin(theta);

    return -(2*b*cosT*cosT)/(sinT*sinT);
}

__device__ Real dzTheta(Real theta, Real b){
    Real cosT = cos(theta);
    Real cosT3 = cosT*cosT*cosT;

    Real sinT = sin(theta);
    Real sinT2 = sinT*sinT;
    Real sinT3 = sinT2*sinT;

    Real b2 = b*b;

    return 2*cosT*((b2/sinT2) - __a2)*sinT + (2*b2*cosT3)/(sinT3);
}

__device__ Real drDelta(Real r){
    return 2*r - 2;
}

__device__ Real rho(Real r, Real theta){
    Real cosT = cos(theta);
    return sqrt(r*r + __a2*cosT*cosT);
}

__device__ Real drRho(Real r, Real theta){
    Real cosT = cos(theta);

    return r / sqrt(__a2*cosT*cosT + r*r);
}

__device__ Real dzRho(Real r, Real theta){
    Real cosT = cos(theta);
    Real sinT = sin(theta);

    return -(__a2*cosT*sinT)/sqrt(__a2*cosT*cosT + r*r);
}

// // NOTE: This is b_0(r) - b, not just b0(r)
// __device__ Real b0b(Real r, Parameters param){
//     Real b = param.b;
//     return -((r*r*r - 3.*(r*r) + __a2*r + __a2) / (__a*(r-1.))) - b;
// }
//
// __device__ Real q0(Real r){
//     Real r3 = r*r*r;
//     return -(r3*(r3 - 6.*(r*r) + 9.*r - 4.*__a2)) / (__a2*((r-1.)*(r-1.)));
// }
//
// __device__ OriginType getOriginType(Real pR, Real b, Real q){
//     Parameters param;
//
//     param.b = b;
//     param.q = q;
//
//     // Compute r0 such that b0(r0) = b
//     Real r0 = secant(-30., 30., b0b, param);
//
//     OriginType origin;
//
//     if(__b1 < b && b < __b2 && q < q0(r0)){
//         if(pR > 0)
//             origin = HORIZON;
//         else
//             origin = CELESTIAL_SPHERE;
//     }
//     else{
//         Real rUp1 = secant(-30., 30., R, param);
//
//         if(__camR < rUp1)
//             origin = HORIZON;
//         else
//             origin = CELESTIAL_SPHERE;
//     }
//
//     return origin;
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
    // Parameters param;

    const Real r = y[0];
    const Real theta = y[1];
    // const Real phi = y[2];
    const Real pR = y[3];
    const Real pTheta = y[4];
    const Real b = data[0];
    const Real q = data[1];

    Real _R, D, Z, rho1, rho2, rho3;

    _R = R(r, b, q);
    D = Delta(r);
    Z = Theta(theta, b, q);

    rho1 = rho(r, theta);
    rho2 = rho1*rho1;
    rho3 = rho1*rho2;

    Real dR, dZ, dRho, dD, sum1, sum2, sum3, sum4, sum5, sum6;

    switch(threadId) {
            case 0:
                f[threadId] = D * pR / rho2;
                break;

            case 1:
                f[threadId] = pTheta / rho2;
                break;

            case 2:
                dR = dbR(r, b);
                dZ = dbTheta(theta, b);

                f[threadId] = - (dR + D*dZ)/(2*D*rho2);
                break;

            case 3:
                dRho = drRho(r, theta);
                dD = drDelta(r);
                dR = drR(r, b, q);

                sum1 = + dRho*pTheta*pTheta / rho3;
                sum2 = + D*pR*pR*dRho / rho3;
                sum3 = - ((D*Z + _R)*dRho / (D*rho3));
                sum4 = - (dD*pR*pR / (2*rho2));
                sum5 = + (dD*Z + dR) / (2*D*rho2);
                sum6 = - (dD*(D*Z + _R) / (2*D*D*rho2));

                f[threadId] = (sum2+sum4) + (sum1) + (sum3+sum5+sum6);
                break;

            case 4:
                dRho = dzRho(r, theta);
                dZ = dzTheta(theta, b);

                sum1 = + dRho*pTheta*pTheta / rho3;
                sum2 = + D*pR*pR*dRho / rho3;
                sum3 = - (D*Z + _R)*dRho / (D*rho3);
                sum4 = + dZ / (2*rho2);

                f[threadId] = sum1 + sum2 + sum3 + sum4;
                break;
    }
}

#endif // __FUNCTIONS__
