#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "Raytracer/Kernel/common.cu"

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

__global__ void setInitialConditions(void* devInitCond,void* devConstants,
                                     Real imageRows, Real imageCols,
                                     Real pixelWidth, Real pixelHeight){
    // Retrieve the id of the block in the grid
    int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    // Pointer for the initial conditions of this ray (block)
    Real* globalInitCond = (Real*) devInitCond;
    Real* initCond = globalInitCond + blockId*SYSTEM_SIZE;

    // Pointer for the constants of this ray (block)
    Real* globalConstants = (Real*) devConstants;
    Real* constants = globalConstants + blockId*2;

    // Compute pixel position in the physical space
    Real x = - (blockIdx.x + 0.5 - imageCols/2) * pixelWidth;
    Real y = (blockIdx.y + 0.5 - imageRows/2) * pixelHeight;

    // Compute direction of the incoming ray in the camera's reference
    // frame
    Real rayPhi = Pi + atan(x / __d);
    Real rayTheta = Pi/2 + atan(y / sqrt(__d*__d + x*x));

    // Compute canonical momenta of the ray and the conserved quantites b
    // and q
    Real pR, pTheta, pPhi, b, q;
    getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
    getConservedQuantities(pTheta, pPhi, &b, &q);

    // if(blockIdx.x == 70 && blockIdx.y == 90){
    //     printf("pR = %.20f\npTheta = %.20f\npPhi = %.20f\nb = %.20f\nq = %.20f, rayTheta = %.20f\nrayPhi = %.20f\n", pR, pTheta, pPhi, b, q, rayTheta, rayPhi);
    // }

    #ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0){
            printf("%.20f, %.20f\n", x, y);
            printf("INICIALES: theta = %.20f, phi = %.20f, pR = %.20f, pTheta = %.20f, pPhi = %.20f, b = %.20f, q = %.20f", rayTheta, rayPhi, pR, pTheta, pPhi, b, q);
        }
    #endif

    // Save ray's initial conditions
    initCond[0] = __camR;
    initCond[1] = __camTheta;
    initCond[2] = __camPhi;
    initCond[3] = pR;
    initCond[4] = pTheta;

    // Save ray's constants
    constants[0] = b;
    constants[1] = q;
}

__device__ int detectCollisions(Real prevThetaCentered,
                                Real currentThetaCentered, Real prevR,
                                Real currentR){
    if (currentR <= horizonRadius){
        return HORIZON;
    }

    if(prevThetaCentered*currentThetaCentered < 0 &&
       prevR > innerDiskRadius && currentR > innerDiskRadius &&
       prevR < outerDiskRadius && currentR < outerDiskRadius){
        return DISK;
    }

    return SPHERE;
}



#include "Raytracer/Kernel/solver.cu"

__global__ void kernel(Real x0, Real xend, void* devInitCond, Real h,
                       Real hmax, void* devData, int dataSize,
                       void* devStatus, Real resolution){

    // Retrieve the ids of the thread in the block and of the block in the grid
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    if(threadId < SYSTEM_SIZE){

        // Array of status flags: at the output, the (x,y)-th element will be 0
        // if any error ocurred (namely, the step size was made too small) and
        // 1 if the computation succeded
        int* globalStatus = (int*) devStatus;
        globalStatus += blockId;
        int status = *globalStatus;

        // Retrieve the position where the initial conditions this block will
        // work with are.
        // Each block, absolutely identified in the grid by blockId, works with
        // only one initial condition (that has N elements, as N equations are
        // in the system). Then, the position of where these initial conditions
        // are stored in the serialized vector can be computed as blockId * N.
        Real* globalInitCond = (Real*) devInitCond;
        globalInitCond += blockId * SYSTEM_SIZE;

        // Pointer to the additional data array used by computeComponent
        Real* globalData = (Real*) devData;
        globalData += blockId * dataSize;

        // Shared arrays to store the initial conditions and the additional
        // data
        __shared__ Real initCond[SYSTEM_SIZE],
                        data[SYSTEM_SIZE];

        initCond[threadId] = globalInitCond[threadId];
        data[threadId] = globalData[threadId];

        __syncthreads();

        // Initialize previous theta and r to the initial conditions
        Real prevThetaCentered, prevR, currentThetaCentered, currentR;

        prevR = initCond[0];
        prevThetaCentered = initCond[1] - HALF_PI;

        // Local variable to know the status of the
        bool success;

        Real x = x0;

        while(status == SPHERE && x > xend){
            if(blockIdx.x == 15 && blockIdx.y == 15 && threadId == 0)
                printf("[%.10f, %.10f]\n", x, x+resolution);
            RK4Solve(x, x + resolution, initCond, &h, resolution, data, &success, threadId, blockId);
            __syncthreads();
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadId == 0)
            //     printf("%.10f\n", h);

            if(success){
                currentR = initCond[0];
                currentThetaCentered = initCond[1] - HALF_PI;

                status = detectCollisions(prevThetaCentered,
                                          currentThetaCentered,
                                          prevR, currentR);

                if(status == DISK){
                    // bisect(threadId, initCond, data, h);
                }
            }
            else{
                status = HORIZON;
            }

            prevR = currentR;
            prevThetaCentered = currentThetaCentered;

            x += resolution;
            // __syncthreads();

        } // While globalStatus == SPHERE and x > xend

        if(threadId == 0)
            *globalStatus = status;

        globalInitCond[threadId] = initCond[threadId];

    } // If threadId < SYSTEM_SIZE
}
