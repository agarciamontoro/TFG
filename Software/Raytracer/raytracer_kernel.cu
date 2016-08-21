#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "Raytracer/definitions.cu"

#define Pi M_PI
#define SYSTEM_SIZE 5

// Declaration of constants
__device__ Real __d;
__device__ Real __camR;
__device__ Real __camTheta;
__device__ Real __camPhi;
__device__ Real __camBeta;
__device__ Real __b1;
__device__ Real __b2;
__device__ Real __ro;
__device__ Real __delta;
__device__ Real __pomega;
__device__ Real __alpha;
__device__ Real __omega;

#include "Raytracer/functions.cu"

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

__global__ void setInitialConditions(void* devInitCond, Real imageRows, Real imageCols, Real pixelWidth, Real pixelHeight, Real d, Real camR, Real camTheta, Real camPhi, Real camBeta, Real a, Real b1, Real b2, Real ro, Real delta, Real pomega, Real alpha, Real omega){
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
    Real y = - (blockIdx.x - imageCols/2.) * pixelWidth;
    Real z =   (blockIdx.y - imageRows/2.) * pixelHeight;

    // Compute direction of the incoming ray in the camera's reference
    // frame
    Real rayPhi = Pi + atan(y / d);
    Real rayTheta = Pi/2 + atan(z / sqrt(d*d + y*y));

    // Compute canonical momenta of the ray and the conserved quantites b
    // and q
    Real pR, pTheta, pPhi, b, q;
    getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
    getConservedQuantities(pTheta, pPhi, &b, &q);

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
