#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "Raytracer/numericalMethods.cu"
#include "Raytracer/definitions.c"
#include "RK4/rk4_kernel_adapted.cu"

#define SYSTEM_SIZE {{ SYSTEM_SIZE }}
{{ DEBUG }}

#define J I
#define Pi M_PI

typedef enum origin{
    HORIZON,
    CELESTIAL_SPHERE
} OriginType;

__device__ Real __d;
__device__ Real __camR;
__device__ Real __camTheta;
__device__ Real __camPhi;
__device__ Real __camBeta;
__device__ Real __a;
__device__ Real __a2;
__device__ Real __b1;
__device__ Real __b2;
__device__ Real __ro;
__device__ Real __delta;
__device__ Real __pomega;
__device__ Real __alpha;
__device__ Real __omega;

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
    return -2*b*r*r + 4*b*r - 4*__a*r;
}

__device__ Real drR(Real r, Real b, Real q){
    return 4*r*(r*r - __a*b + __a2) - (q + (b-__a)*(b-__a))*(2*r - 2);
}

__device__ Real Theta(Real r, Real theta, Real b, Real q){
    Real sinTheta = sin(theta);
    Real sin2 = sinTheta*sinTheta;

    Real cosTheta = cos(theta);
    Real cos2 = cosTheta*cosTheta;

    return q - cos2*(b*b/sin2 - __a2);
}

__device__ Real dbTheta(Real theta, Real b){
    Real cosTheta = cos(theta);
    Real sinTheta = sin(theta);

    return -(2*b*cosTheta*cosTheta)/(sinTheta*sinTheta);
}

__device__ Real dzTheta(Real theta, Real b){
    Real cosT = cos(theta);
    Real cosT2 = cosT*cosT;

    Real sinT = sin(theta);
    Real sinT2 = sinT*sinT;
    Real sinT3 = sinT2*sinT;
    Real sinT4 = sinT2*sinT2;

    Real b2 = b*b;

    return -2*cosT*(__a2*sinT4 - b2*sinT2 - b2*cosT2)/sinT3;
}

__device__ Real Delta(Real r){
    return r*r - 2*r + __a2;
}

__device__ Real drDelta(Real r){
    return 2*r-2;
}

__device__ Real rho(Real r, Real theta){
    Real cosTheta = cos(theta);
    return sqrt(r*2 + __a2*cosTheta*cosTheta);
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

__device__ Real eqMomenta(Parameters param){
    Real r = param.r;
    Real pR = param.pR;
    Real pTheta = param.pTheta;

    Real _Delta = Delta(param.r);
    Real _rho = rho(param.r, param.theta);
    Real tworho2 = 2*_rho*_rho;
    Real _R = R(r, param);
    Real _Theta = Theta(r, param.theta, param.b, param.q);

    Real sol = -(_Delta*pR*pR/tworho2) - (pTheta*pTheta/tworho2) + ((_R+_Delta*_Theta)/(_Delta*tworho2));

    printf("%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", param.r, param.pR, param.pTheta, param.theta, param.phi, param.b, param.q);

    return sol;
}

__device__ Real eqMomentaTheta(Real theta, Parameters param){
    param.theta = theta;
    return eqMomenta(param);
}

__device__ Real eqMomentaR(Real r, Parameters param){
    param.r = r;
    return eqMomenta(param);
}

__device__ Real eqPhi(Real b, Parameters param){
    Real _R = R(param.r, param);
    Real _Delta = Delta(param.r);
    Real _Theta = Theta(param.r, param.theta, param.b, param.q);
    Real _rho = rho(param.r, param.theta);

    return (_R+_Delta*_Theta)/(2*_Delta*_rho*_rho);
}

__device__ void getCanonicalMomenta(Real rayTheta, Real rayPhi, Real* pR,
                                    Real* pTheta, Real* pPhi){
    // **************************** SET NORMAL **************************** //
    // Cartesian components of the unit vector N pointing in the direction of
    // the incoming ray
    Real Nx = sin(rayTheta) * cos(rayPhi);
    Real Ny = sin(rayTheta) * sin(rayPhi);
    Real Nz = cos(rayTheta);

    // ********************** SET DIRECTION OF MOTION ********************** //
    // Compute denominator, common to all the cartesian components
    Real den = 1. - __camBeta * Ny;

    // Compute factor common to nx and nz
    Real fac = -sqrt(1. - __camBeta*__camBeta);

    // Compute cartesian coordinates of the direction of motion. See(A.9)
    Real nY = (-Ny + __camBeta) / den;
    Real nX = fac * Nx / den;
    Real nZ = fac * Nz / den;

    // Convert the direction of motion to the FIDO's spherical orthonormal
    // basis. See (A.10)
    Real nR = nX;
    Real nTheta = -nZ;
    Real nPhi = nY;

    // *********************** SET CANONICAL MOMENTA *********************** //
    // Compute energy as measured by the FIDO. See (A.11)
    Real E = 1. / (__alpha + __omega * __pomega * nPhi);

    // Set conserved energy to unity. See (A.11)
    // Real pt = -1;

    // Compute the canonical momenta. See (A.11)
    *pR = E * __ro * nR / sqrt(__delta);
    *pTheta = E * __ro * nTheta;
    *pPhi = E * __pomega * nPhi;
}

__device__ void getConservedQuantities(Real pTheta, Real pPhi, Real* b,
                                       Real* q){
    // ********************* GET CONSERVED QUANTITIES ********************* //
    // Get conserved quantities. See (A.12)
    *b = pPhi;
    Real sinTheta = sin(__camTheta);
    *q = pTheta*pTheta + cos(__camTheta)*((*b)*(*b) / sinTheta*sinTheta - __a2);
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

__device__ OriginType getOriginType(Real pR, Real b, Real q){
    Parameters param;

    param.b = b;
    param.q = q;

    // Compute r0 such that b0(r0) = b
    Real r0 = secant(-30., 30., b0b, param);

    OriginType origin;

    if(__b1 < b && b < __b2 && q < q0(r0)){
        if(pR > 0)
            origin = HORIZON;
        else
            origin = CELESTIAL_SPHERE;
    }
    else{
        Real rUp1 = secant(-30., 30., R, param);

        if(__camR < rUp1)
            origin = HORIZON;
        else
            origin = CELESTIAL_SPHERE;
    }

    return origin;
}



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
__device__ void computeComponent(int threadId, Real x, Real* y, Real* f, Real b, Real q){
    Parameters param;

    Real _R, D, Z, rho1, rho2, rho3, rho4;

    switch(threadId) {
            case 0:
                param.r = y[0];
                param.b = b;
                Z = Theta(param.r, param.theta, param.b, param.q);
                break;

            case 1:
                param.theta = y[1];
                param.q = q;
                break;

            case 2:
                param.phi = y[2];
                D = Delta(param.r);
                break;

            case 3:
                param.pR = y[3];
                rho1 = rho(param.r, param.theta);
                rho2 = rho1*rho1;
                rho3 = rho1*rho2;
                rho4 = rho2*rho2;
                break;

            case 4:
                param.pTheta = y[4];
                _R = R(param.r, param);
                break;
    }
    __syncthreads();

    Real dR, dZ, dRho, dD, sum1, sum2, sum3, num, den;

    switch(threadId) {
            case 0:
                f[threadId] = D * param.pR / rho2;
                break;

            case 1:
                f[threadId] = param.pTheta / rho2;
                break;

            case 2:
                dR = dbR(param.r, param.b);
                dZ = dbTheta(param.theta, param.b);

                f[threadId] = - (dR + D*dZ)/(2*D*rho2);
                break;

            case 3:
                dRho = drRho(param.r, param.theta);
                dD = drDelta(param.r);
                dR = drR(param.r, param.b, param.q);

                sum1 = (dRho*2*D - dD*rho1)/(2*rho3);
                sum2 = (dRho*param.pTheta*param.pTheta)/(rho3);

                num = (dR + Z*dD)*D*rho2 - (_R + D*Z)*(dD*D*rho2 + 2*D*rho1*dRho);
                den = 2*D*D*rho4;
                sum3 = num/den;

                f[threadId] = sum1 + sum2 + sum3;
                break;

            case 4:
                dRho = dzRho(param.r, param.theta);
                dZ = dzTheta(param.theta, param.b);

                sum1 = (dRho*D*param.pR*param.pR)/(rho3);
                sum2 = (dRho*param.pTheta*param.pTheta)/(rho3);
                sum3 = (D*(dZ*rho1 - 2*Z*dRho) - 2*_R*dRho)/(2*D*rho3);

                f[threadId] = sum1 + sum2 + sum3;
                break;
    }
}


__global__ void rayTrace(void* devImage, Real imageRows, Real imageCols, Real pixelWidth, Real pixelHeight, Real d, Real camR, Real camTheta, Real camPhi, Real camBeta, Real a, Real b1,Real b2, Real ro, Real delta, Real pomega, Real alpha, Real omega){
    // Shared memory for the initial conditions of this thread
    __shared__ Real initCond[SYSTEM_SIZE];

    // Retrieve the ids of the thread in the block and of the block in the grid
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    // int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    if(threadId < SYSTEM_SIZE){
        Real* globalImage = (Real*) devImage;

        // Set global variables, common to all threads and constants
        // Camera constants
        __d = d;
        __camR = camR;
        __camTheta = camTheta;
        __camPhi = camPhi;
        __camBeta = camBeta;

        // Black hole constants
        __a = a;
        __a2 = a*a;
        __b1 = b1;
        __b2 = b2;

        // Kerr constants
        __ro = ro;
        __delta = delta;
        __pomega = pomega;
        __alpha = alpha;
        __omega = omega;

        // Compute pixel position in the physical space
        Real y = - (blockIdx.x - imageCols/2.) * pixelWidth;
        Real z =   (blockIdx.y - imageRows/2.) * pixelHeight;

        // Compute direction of the incoming ray in the camera's reference frame
        Real rayPhi = Pi + atan(y / d);
        Real rayTheta = Pi/2 + atan(z / sqrt(d*d + y*y));

        // Compute canonical momenta of the ray and the conserved quantites b and q
        Real pR, pTheta, pPhi, b, q;
        getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
        getConservedQuantities(pTheta, pPhi, &b, &q);

        // Check whether the ray comes from the horizon or from the celetial sphere
        int color = getOriginType(pR, b, q);


        __shared__ Real absoluteTol[SYSTEM_SIZE];
        __shared__ Real relativeTol[SYSTEM_SIZE];
        // Populate the initial conditions. Is this parallelization even necessary?
        switch(threadId){
            case 0:
                initCond[0] = __camR;
                relativeTol[0] = 1e-6;
                absoluteTol[0] = 1e-12;
                break;

            case 1:
                initCond[1] = rayTheta;
                relativeTol[1] = 1e-6;
                absoluteTol[1] = 1e-12;
                break;

            case 2:
                initCond[2] = rayPhi;
                relativeTol[2] = 1e-6;
                absoluteTol[2] = 1e-12;
                break;

            case 3:
                initCond[3] = pR;
                relativeTol[3] = 1e-6;
                absoluteTol[3] = 1e-12;
                break;

            case 4:
                initCond[4] = pTheta;
                relativeTol[4] = 1e-6;
                absoluteTol[4] = 1e-12;
                break;
        }
        __syncthreads();

        RK4Solve(computeComponent, b, q,
                 0., -0.1, initCond, -0.01, -0.1,
                 relativeTol, absoluteTol, 0.9, 0.2, 10.0, 0.04, 2.3e-16);


        globalImage[3*(blockIdx.x + blockIdx.y*gridDim.x) + 0] = color;
        globalImage[3*(blockIdx.x + blockIdx.y*gridDim.x) + 1] = color;
        globalImage[3*(blockIdx.x + blockIdx.y*gridDim.x) + 2] = color;
    }
}
