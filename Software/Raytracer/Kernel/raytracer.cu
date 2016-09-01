#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "Raytracer/Kernel/common.cu"
#include "Raytracer/Kernel/solvers.cu"

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
                                     Real pixelWidth, Real pixelHeight){
    // Compute pixel's row and col of this thread
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < IMG_ROWS && col < IMG_COLS){
        // Compute pixel unique identifier for this thread
        int pixel = row*IMG_COLS + col;

        // Pointer for the initial conditions of this ray (block)
        Real* globalInitCond = (Real*) devInitCond;
        Real* initCond = globalInitCond + pixel*SYSTEM_SIZE;

        // Pointer for the constants of this ray (block)
        Real* globalConstants = (Real*) devConstants;
        Real* constants = globalConstants + pixel*2;

        // Compute pixel position in the physical space
        Real x = - (col + 0.5 - IMG_COLS/2) * pixelWidth;
        Real y = (row + 0.5 - IMG_ROWS/2) * pixelHeight;

        // Compute direction of the incoming ray in the camera's reference
        // frame
        Real rayPhi = Pi + atan(x / __d);
        Real rayTheta = Pi/2 + atan(y / sqrt(__d*__d + x*x));

        // Compute canonical momenta of the ray and the conserved quantites b
        // and q
        Real pR, pTheta, pPhi, b, q;
        getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
        getConservedQuantities(pTheta, pPhi, &b, &q);

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
}

__global__ void kernel(Real x0, Real xend, void* devInitCond, Real h,
                       Real hmax, void* devData, int dataSize,
                       void* devStatus, Real resolutionOrig){
    // Compute pixel's row and col of this thread
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Only the threads that have a proper pixel shall compute its ray equation
    if(row < IMG_ROWS && col < IMG_COLS){
        // Compute pixel unique identifier for this thread
        int pixel = row*IMG_COLS + col;

        // Array of status flags: at the output, the (x,y)-th element will be 0
        // if any error ocurred (namely, the step size was made too small) and
        // 1 if the computation succeded
        int* globalStatus = (int*) devStatus;
        globalStatus += pixel;
        int status = *globalStatus;

        // Retrieve the position where the initial conditions this block will
        // work with are.
        // Each block, absolutely identified in the grid by blockId, works with
        // only one initial condition (that has N elements, as N equations are
        // in the system). Then, the position of where these initial conditions
        // are stored in the serialized vector can be computed as blockId * N.
        Real* globalInitCond = (Real*) devInitCond;
        globalInitCond += pixel * SYSTEM_SIZE;

        // Pointer to the additional data array used by computeComponent
        Real* globalData = (Real*) devData;
        globalData += pixel * dataSize;

        // Local arrays to store the initial conditions and the additional
        // data
        Real initCond[SYSTEM_SIZE], data[DATA_SIZE];

        // Retrieve the data from global to local memory :)
        memcpy(initCond, globalInitCond, sizeof(Real)*SYSTEM_SIZE);
        memcpy(data, globalData, sizeof(Real)*DATA_SIZE);

        // Variables to keep track of the current r and the previous and
        // current theta
        Real currentR;
        int prevThetaSign, currentThetaSign;

        // Initialize previous theta to the initial conditions
        prevThetaSign = sign(initCond[1] - HALF_PI);

        // Current time
        Real x = x0;

        // Local variable to know the status of the ray
        SolverStatus solverStatus;

        // Auxiliar array used to pass a copy of the data to bisect.
        // Bisect changes the data it receives, and we want to change them only
        // when the result of the bisect tells us the ray has collided with the
        // disk.
        // Hence: if we have to call bisect, we put a copy of the current data
        // into dataCopy, which we pass to bisect; then, only if the ray has
        // collided with the disk, we transfer again the data from copyData to
        // initCond.
        Real copyData[SYSTEM_SIZE];

        // Local variable to know how many iterations spent the solver in the
        // current step.
        int iterations = 0;

        // Local variable to know how many iterations spent the bisect in the
        // current step.
        int bisectIter;

        // This variable belongs to the solver logic, not the raytracer logic.
        // It is used inside the solver to automatically compute the steps
        // size. Without keeping track of this variable here, the solver would
        // reset it each time the method is called. As we want the solver to
        // think it is continuosly computing the evolution of the ray (the
        // resolution variable is invisible to the solver), it is mandatory to
        // manage the facold variable from here.
        float facold = 1.0e-4;

        // Size of the interval in whose extrems we will check whether the ray
        // has crossed theta = pi/2
        Real resolution = -1.0;

        // MAIN LOOP. Each iteration has the following phases:
        //   -> 0. Check that the ray has not collided with the disk or with
        //      the horizon and that the current time has not exceeded the
        //      final time.
        //   -> 1. Advance the ray a time of `resolution`, calling the main
        //      RK45 solver.
        //   -> 2. Test whether the ray has collided with the horizon.
        //          2.1 If the answer to the 2. test is positive: test whether
        //          the current theta has crossed theta = pi/2, and call bisect
        //          in case it did, updating its status accordingly (set it to
        //          DISK if the ray collided with the horizon).
        //          2.2. If the answer to the 2. test is negative: update the
        //          status of the ray to HORIZON.
        while(status == SPHERE && x > xend){
            // PHASE 1: Advance time an amount of `resolution`. The solver
            // itself updates the current time x with the final time reached
            solverStatus = SolverRK45(&x, x + resolution, initCond, &h,
                                      resolution, data, &iterations, &facold);

            // PHASE 2: Check whether the ray has collided with the horizon
            if(solverStatus == SOLVER_SUCCESS){
                // PHASE 2.1: Check if theta has crossed pi/2

                // Update current theta
                currentThetaSign = sign(initCond[1] - HALF_PI);

                // Check whether the ray has crossed theta = pi/2
                if(prevThetaSign != currentThetaSign){
                    // Copy the current ray state to the auxiliar array
                    memcpy(copyData, initCond, sizeof(Real)*SYSTEM_SIZE);

                    // Call bisect in order to find the exact spot where theta
                    // = pi/2
                    bisectIter = bisect(copyData, data, resolution, x);

                    // Safe guard: if bisect failed, put the status to HORIZON
                    if(bisectIter == -1){
                        status = HORIZON;
                        break;
                    }

                    // Retrieve the current r
                    currentR = copyData[0];

                    // Finally, check whether the current r is inside the disk,
                    // updating the status and copying back the data in the
                    // case it is
                    if(innerDiskRadius<currentR && currentR<outerDiskRadius){
                        status = DISK;
                        memcpy(initCond, copyData, sizeof(Real)*SYSTEM_SIZE);
                    }
                }
            }
            else{
                // PHASE 2.2: The ray has collided with the horizon
                status = HORIZON;
            }

            // Update the previous variables for the next step computation
            prevThetaSign = currentThetaSign;

        } // While globalStatus == SPHERE and x > xend

        // Once the loop is finished (the ray has been computed until the final
        // time or it has collided with the disk/horizon), update the global
        // status variable
        *globalStatus = status;

        // And, finally, update the current ray state in global memory :)
        memcpy(globalInitCond, initCond, sizeof(Real)*SYSTEM_SIZE);

    } // If row < IMG_ROWS and col < IMG_COLS
}
