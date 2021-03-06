#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "Raytracer/Kernel/definitions.cu"
#include "Raytracer/Kernel/solver.cu"

#define Pi M_PI
#define SYSTEM_SIZE 5


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

    Real sinT = sin(__camTheta);
    Real sinT2 = sinT*sinT;

    Real cosT = cos(__camTheta);
    Real cosT2 = cosT*cosT;

    Real pTheta2 = pTheta*pTheta;
    Real b2 = pPhi*pPhi;

    *q = pTheta2 + cosT2*((b2/sinT2) - __a2);
}

__global__ void setInitialConditions(void* devInitCond,
                                     Real imageRows, Real imageCols,
                                     Real pixelWidth, Real pixelHeight){
    // Retrieve the id of the block in the grid
    int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    // Pointer for the initial conditions of this block
    Real* globalInitCond = (Real*) devInitCond;
    Real* initCond = globalInitCond + blockId*(SYSTEM_SIZE + 2);

    // Set global variables, common to all threads, and constants
    // Camera constants
    __d = d;
    __camR = camR;
    __camTheta = camTheta;
    __camPhi = camPhi;
    __camBeta = camBeta;

    // Black hole constants
    __b1 = b1;
    __b2 = b2;

    // Kerr constants
    __ro = ro;
    __delta = delta;
    __pomega = pomega;
    __alpha = alpha;
    __omega = omega;

    // Compute pixel position in the physical space
    Real x = - (blockIdx.x + 0.5 - imageCols/2) * pixelWidth;
    Real y = (blockIdx.y + 0.5 - imageRows/2) * pixelHeight;

    // Compute direction of the incoming ray in the camera's reference
    // frame
    Real rayPhi = Pi + atan(x / d);
    Real rayTheta = Pi/2 + atan(y / sqrt(d*d + x*x));

    // Compute canonical momenta of the ray and the conserved quantites b
    // and q
    Real pR, pTheta, pPhi, b, q;
    getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
    getConservedQuantities(pTheta, pPhi, &b, &q);

    if(blockIdx.x == 70 && blockIdx.y == 90){
        printf("pR = %.20f\npTheta = %.20f\npPhi = %.20f\nb = %.20f\nq = %.20f, rayTheta = %.20f\nrayPhi = %.20f\n", pR, pTheta, pPhi, b, q, rayTheta, rayPhi);
    }

    #ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0){
            printf("%.20f, %.20f\n", y, z);
            printf("INICIALES: theta = %.20f, phi = %.20f, pR = %.20f, pTheta = %.20f, pPhi = %.20f, b = %.20f, q = %.20f", rayTheta, rayPhi, pR, pTheta, pPhi, b, q);
        }
    #endif

    initCond[0] = __camR;
    initCond[1] = __camTheta;
    initCond[2] = __camPhi;
    initCond[3] = pR;
    initCond[4] = pTheta;
    initCond[5] = b;
    initCond[6] = q;
}

__global__ void rayTrace(Real xEnd, Real step, void* devRays){
    Real x = 0;

    while(x > xEnd){
        // Solve the system for
        RK4Solve(x, x+step, devRays, step/10., xEnd, devData, dataSize,
                 devStatus);

        detectCollisions(devRays, devStatus);
    }
}
