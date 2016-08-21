#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "Raytracer/definitions.cu"
#include "Raytracer/numericalMethods.cu"

__device__ Real P(Real r, Real b){
    return r*r + __a2 - __a*b;
}

__device__ Real R(Real r, Parameters param){
    Real b = param.b;
    Real q = param.q;

    Real r2 = r*r;
    Real r4 = r2*r2;
    Real b2 = b*b;

    return r4 - q*r2 - b2*r2 + __a2*r2 + 2*q*r + 2*b2*r - 4*__a*b*r + 2*__a2*r - __a2*q;
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

__device__ Real Delta(Real r){
    return r*r - 2*r + __a2;
}

__device__ Real drDelta(Real r){
    return 2*r - 2;
}

__device__ Real rho(Real r, Real theta){
    Real cosT = cos(theta);
    return sqrt(r*2 + __a2*cosT*cosT);
}

__device__ Real drRho(Real r, Real theta){
    Real cosT = cos(theta);

    return r/sqrt(__a2*cosT*cosT + r*r);
}

__device__ Real dzRho(Real r, Real theta){
    Real cosT = cos(theta);
    Real sinT = sin(theta);

    return -(__a2*cosT*sinT)/sqrt(__a2*cosT*cosT + r*r);
}

__device__ Real eqPhi(Real b, Parameters param){
    Real _R = R(param.r, param);
    Real _Delta = Delta(param.r);
    Real _Theta = Theta(param.theta, param.b, param.q);
    Real _rho = rho(param.r, param.theta);

    return (_R+_Delta*_Theta)/(2*_Delta*_rho*_rho);
}

// NOTE: This is b_0(r) - b, not just b0(r)
__device__ Real b0b(Real r, Parameters param){
    Real b = param.b;
    return -((r*r*r - 3.*(r*r) + __a2*r + __a2) / (__a*(r-1.))) - b;
}

__device__ Real q0(Real r){
    Real r3 = r*r*r;
    return -(r3*(r3 - 6.*(r*r) + 9.*r - 4.*__a2)) / (__a2*((r-1.)*(r-1.)));
}

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
 * @param Real  x  Value of the time in which the system is solved
 * @param Real* y  Initial conditions for the system: a pointer to a vector
 *                 whose lenght shall be the same as the number of equations in
 *                 the system.
 * @param Real* f  Computed value of the function: a pointer to a vector whose
 *                 lenght shall be the same as the number of equations in the
 *                 system.
 */
__device__ void computeComponent(int threadId, Real x, Real* y, Real* f,
                                 Real* data){
    Parameters param;

    param.r = y[0];
    param.theta = y[1];
    param.phi = y[2];
    param.pR = y[3];
    param.pTheta = y[4];
    param.b = data[0];
    param.q = data[1];

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadId == 0){
    //     printf("CC[%.10f]: %.20f, %.20f, %.20f, %.20f, %.20f, %.20f, %.20f, %.20f, %.20f\n", x, __a, __a2, param.r, param.theta, param.phi, param.pR, param.pTheta, param.b, param.q);
    // }

    Real _R, D, Z, rho1, rho2, rho3;

    _R = R(param.r, param);
    D = Delta(param.r);
    Z = Theta(param.theta, param.b, param.q);

    rho1 = rho(param.r, param.theta);
    rho2 = rho1*rho1;
    rho3 = rho1*rho2;

    Real dR, dZ, dRho, dD, sum1, sum2, sum3, sum4, sum5, sum6;

    switch(threadId) {
            case 0:
                f[threadId] = D * param.pR / rho2;
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;

            case 1:
                f[threadId] = param.pTheta / rho2;
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;

            case 2:
                dR = dbR(param.r, param.b);
                dZ = dbTheta(param.theta, param.b);

                f[threadId] = - (dR + D*dZ)/(2*D*rho2);
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;

            case 3:
                dRho = drRho(param.r, param.theta);
                dD = drDelta(param.r);
                dR = drR(param.r, param.b, param.q);

                sum1 = + dRho*param.pTheta*param.pTheta / rho3;
                sum2 = + D*param.pR*param.pR*dRho / rho3;
                sum3 = - ((D*Z + _R)*dRho / (D*rho3));
                sum4 = - (dD*param.pR*param.pR / (2*rho2));
                sum5 = + (dD*Z + dR) / (2*D*rho2);
                sum6 = - (dD*(D*Z + _R) / (2*D*D*rho2));

                f[threadId] = sum1 + sum2 + sum3 + sum4 + sum5 + sum6;
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;

            case 4:
                dRho = dzRho(param.r, param.theta);
                dZ = dzTheta(param.theta, param.b);

                sum1 = + dRho*param.pTheta*param.pTheta / rho3;
                sum2 = + D*param.pR*param.pR*dRho / rho3;
                sum3 = - (D*Z + _R)*dRho / (D*rho3);
                sum4 = + dZ / (2*rho2);

                f[threadId] = sum1 + sum2 + sum3 + sum4;
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;
    }
}

#endif // __FUNCTIONS__
