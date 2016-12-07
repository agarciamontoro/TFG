#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
// #include <string.h>

#include "common.c"
#include "solver.c"

#define Pi M_PI
#define SYSTEM_SIZE 5

#define DATA_SIZE 2

void getCanonicalMomenta(Real rayTheta, Real rayPhi, Real* pR, Real* pTheta,
                         Real* pPhi){
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

void getConservedQuantities(Real pTheta, Real pPhi, Real* b, Real* q){
    // ********************* GET CONSERVED QUANTITIES ********************* //
    // Compute axial angular momentum. See (A.12).
    *b = pPhi;

    // Compute Carter constant. See (A.12).
    Real sinT = sin(__camTheta);
    Real sinT2 = sinT*sinT;

    Real cosT = cos(__camTheta);
    Real cosT2 = cosT*cosT;

    Real pTheta2 = pTheta*pTheta;
    Real b2 = pPhi*pPhi;

    *q = pTheta2 + cosT2*((b2/sinT2) - __a2);
}

void setInitialConditions(Real* globalInitCond,Real* globalConstants,
                          Real imageRows, Real imageCols, Real pixelWidth,
                          Real pixelHeight){
    for(int row = 0; row < imageRows; row++){
        for(int col = 0; col < imageCols; col++){
            // Compute pixel unique identifier for this thread
            int pixel = row*imageCols + col;

            // Compute the position in the global array to store the initial
            // conditions of this ray
            Real* initCond = globalInitCond + pixel*SYSTEM_SIZE;

            // Compute the position in the global array to store the constants of
            // this ray
            Real* constants = globalConstants + pixel*2;

            // Compute pixel position in physical units
            Real x = - (col + 0.5 - imageCols/2) * pixelWidth;
            Real y = (row + 0.5 - imageRows/2) * pixelHeight;

            // Compute direction of the incoming ray in the camera's reference
            // frame
            Real rayPhi = Pi + atan(x / __d);
            Real rayTheta = Pi/2 + atan(y / sqrt(__d*__d + x*x));

            // Compute canonical momenta of the ray and the conserved quantites b
            // and q
            Real pR, pTheta, pPhi, b, q;
            getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
            getConservedQuantities(pTheta, pPhi, &b, &q);

            // Save ray's initial conditions in the global array
            initCond[0] = __camR;
            initCond[1] = __camTheta;
            initCond[2] = __camPhi;
            initCond[3] = pR;
            initCond[4] = pTheta;

            // Save ray's constants in the global array
            constants[0] = b;
            constants[1] = q;
        }
    }
}

void kernel(Real x0, Real xend, Real* devInitCond, Real h, Real hmax,
            Real* devData, int dataSize, int* devStatus,
            Real resolution, int imageRows, int imageCols){
    for(int row = 0; row < imageRows; row++){
        for(int col = 0; col < imageCols; col++){
            // Compute pixel unique identifier for this thread
            int pixel = row*imageCols + col;

            // Array of status flags: at the output, the (x,y)-th element will be
            // set to SPHERE, HORIZON or disk, showing the final state of the ray.
            int* globalStatus = (int*) devStatus;
            globalStatus += pixel;
            int status = *globalStatus;

            // Integrate the ray only if it's still in the sphere. If it has
            // collided either with the disk or within the horizon, it is not
            // necessary to integrate it anymore.
            if(status == SPHERE){
                // Retrieve the position where the initial conditions this block
                // will work with are.
                // Each block, absolutely identified in the grid by blockId, works
                // with only one initial condition (that has N elements, as N
                // equations are in the system). Then, the position of where these
                // initial conditions are stored in the serialized vector can be
                // computed as blockId * N.
                Real* globalInitCond = (Real*) devInitCond;
                globalInitCond += pixel * SYSTEM_SIZE;

                // Pointer to the additional data array used by computeComponent
                Real* globalData = (Real*) devData;
                globalData += pixel * DATA_SIZE;

                // Local arrays to store the initial conditions and the additional
                // data
                Real initCond[SYSTEM_SIZE], data[DATA_SIZE];

                // Retrieve the data from global to local memory :)
                memcpy(initCond, globalInitCond, sizeof(Real)*SYSTEM_SIZE);
                memcpy(data, globalData, sizeof(Real)*DATA_SIZE);

                // Current time
                Real x = x0;

                // Local variable to know how many iterations spent the solver in
                // the current step.
                int iterations = 0;

                // MAIN ROUTINE. Integrate the ray from x to xend, checking disk
                // collisions on the go with the following algorithm:
                //   -> 0. Check that the ray has not collided with the disk or
                //   with the horizon and that the current time has not exceeded
                //   the final time.
                //   -> 1. Advance the ray a step, calling the main RK45 solver.
                //   -> 2. Test whether the ray has collided with the horizon.
                //          2.1 If the answer to the 2. test is negative: test
                //          whether the current theta has crossed theta = pi/2,
                //          and call bisect in case it did, updating its status
                //          accordingly (set it to DISK if the ray collided with
                //          the horizon).
                //          2.2. If the answer to the 2. test is positive: update
                //          the status of the ray to HORIZON.
                status = SolverRK45(&x, xend, initCond, h, xend - x, data,
                                    &iterations);

                // Update the global status variable with the new computed status
                *globalStatus = status;

                // And, finally, update the current ray state in global memory :)
                memcpy(globalInitCond, initCond, sizeof(Real)*SYSTEM_SIZE);
            } // If status == SPHERE
        }
    }
}

double measureKernelTime(int imageRows, int imageCols){
    Real* initCond = (Real*) malloc(imageCols * imageRows * SYSTEM_SIZE *
                                    sizeof(Real));
    Real* constants = (Real*) malloc(imageCols * imageRows * DATA_SIZE *
                                     sizeof(Real));
    int* status = (int*) malloc(imageCols * imageRows * sizeof(int));

    for (size_t i = 0; i < imageRows; i++) {
        for (size_t j = 0; j < imageCols; j++) {
            status[i*imageCols + j] = SPHERE;
        }
    }

    // Sensor physical size
    Real W = 2;
    Real H = 2;

    Real pixelWidth = W / (Real) imageCols;
    Real pixelHeight = H / (Real) imageRows;

    setInitialConditions(initCond, constants, imageRows, imageCols, pixelWidth,
                         pixelHeight);

    Real x0 = 0.;
    Real xend = -150.;
    Real h = -0.001;
    Real resolution = -1;

    clock_t start, end;

    start = clock();
    kernel(x0, xend, initCond, h, xend-x0, constants, DATA_SIZE, status,
           resolution, imageRows, imageCols);
    end = clock();

    FILE *f = fopen("file.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (size_t i = 0; i < imageRows; i++) {
        for (size_t j = 0; j < imageCols; j++) {
            fprintf(f, "%d,", status[i*imageCols + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);

    free(status);
    free(constants);
    free(initCond);

    Real timeExec = (end-start)/(double)CLOCKS_PER_SEC;
    return timeExec;
}
