#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "common.c"
#include "solver.c"

#define Pi M_PI
#define SYSTEM_SIZE 5

#define IMG_COLS 500
#define IMG_ROWS 500
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

void setInitialConditions(Real* globalInitCond,Real* globalConstants,
                          Real imageRows, Real imageCols, Real pixelWidth,
                          Real pixelHeight){
    for(int row = 0; row < imageRows; row++){
        for(int col = 0; col < imageCols; col++){
            // Retrieve the id of the block in the grid
            int blockId =  col  + row  * imageCols;

            // Pointer for the initial conditions of this ray (block)
            Real* initCond = globalInitCond + blockId*SYSTEM_SIZE;

            // Pointer for the constants of this ray (block)
            Real* constants = globalConstants + blockId*2;

            // Compute pixel position in the physical space
            Real x = - (col + 0.5 - imageCols/2) * pixelWidth;
            Real y = (row + 0.5 - imageRows/2) * pixelHeight;

            // Compute direction of the incoming ray in the camera's reference
            // frame
            Real rayPhi = Pi + atan(x / __d);
            Real rayTheta = Pi/2 + atan(y / sqrt(__d*__d + x*x));

            // Compute canonical momenta of the ray and the conserved quantites
            // b and q
            Real pR, pTheta, pPhi, b, q;
            getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
            getConservedQuantities(pTheta, pPhi, &b, &q);

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
}

int detectCollisions(Real prevThetaCentered, Real currentThetaCentered,
                     Real prevR, Real currentR){
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

void kernel(Real x0, Real xend, Real* devInitCond, Real h, Real hmax,
            Real* devData, int dataSize, int* devStatus,
            Real resolution){
    for(int row = 0; row < IMG_ROWS; row++){
        for(int col = 0; col < IMG_COLS; col++){
            // Retrieve the ids of the thread in the block and of the block in
            // the grid
            int blockId =  col  + row  * IMG_COLS;

            // Array of status flags: at the output, the (x,y)-th element will
            // be 0 if any error ocurred (namely, the step size was made too
            // small) and 1 if the computation succeded
            int* globalStatus = devStatus;
            globalStatus += blockId;
            int status = *globalStatus;

            // Retrieve the position where the initial conditions this block
            // will work with are.
            // Each block, absolutely identified in the grid by blockId, works
            // with only one initial condition (that has N elements, as N
            // equations are in the system). Then, the position of where these
            // initial conditions are stored in the serialized vector can be
            // computed as blockId * N.
            Real* globalInitCond = devInitCond;
            globalInitCond += blockId * SYSTEM_SIZE;

            // Pointer to the additional data array used by computeComponent
            Real* globalData = (Real*) devData;
            globalData += blockId * dataSize;

            // Shared arrays to store the initial conditions and the additional
            // data
            Real initCond[SYSTEM_SIZE],
                 data[SYSTEM_SIZE];

            for(int thread = 0; thread < SYSTEM_SIZE; thread++){
                initCond[thread] = globalInitCond[thread];
                data[thread] = globalData[thread];
            }

            // Initialize previous theta and r to the initial conditions
            Real prevThetaCentered, prevR, currentThetaCentered, currentR;

            prevR = initCond[0];
            prevThetaCentered = initCond[1] - HALF_PI;

            // Local variable to know the status of the
            bool success;

            Real x = x0;

            while(status == SPHERE && x > xend){
                RK4Solve(x, x + resolution, initCond, &h, resolution, data, &success, blockId);

                if(success){
                    currentR = initCond[0];
                    currentThetaCentered = initCond[1] - HALF_PI;

                    status = detectCollisions(prevThetaCentered,
                                              currentThetaCentered,
                                              prevR, currentR);

                    if(status == DISK){
                        bisect(initCond, data, h);
                    }
                }
                else{
                    status = HORIZON;
                }

                prevR = currentR;
                prevThetaCentered = currentThetaCentered;

                x += resolution;

            } // While globalStatus == SPHERE and x > xend

            *globalStatus = status;

            for(int thread = 0; thread < SYSTEM_SIZE; thread++){
                globalInitCond[thread] = initCond[thread];
            }
        }
    }
}

int main(int argc, char* argv[]){
    if(argc < 2){
      fprintf(stderr, "Oops, where's the output file path? I can't see it!\n");
      exit(1);
    }

    FILE *output = fopen(argv[1], "w");

    if (output == NULL) {
      fprintf(stderr, "Can't open output file %s!\n", argv[1]);
      exit(1);
    }

    fprintf(output, "Number of pixels, Computation time\n");

    Real* initCond = (Real*) malloc(IMG_COLS * IMG_ROWS * SYSTEM_SIZE * sizeof(Real));
    Real* constants = (Real*) malloc(IMG_COLS * IMG_ROWS * DATA_SIZE * sizeof(Real));
    int* status = (int*) malloc(IMG_COLS * IMG_ROWS * sizeof(int));

    // Sensor physical size
    Real W = 2;
    Real H = 2;

    Real pixelWidth = W / (Real) IMG_COLS;
    Real pixelHeight = H / (Real) IMG_ROWS;

    setInitialConditions(initCond, constants, IMG_ROWS, IMG_COLS, pixelWidth,
                         pixelHeight);

    Real x0 = 0.;
    Real xend = -90.;
    Real h = -0.001;
    Real resolution = -1;

    clock_t start, end;
    start = clock();

    kernel(x0, xend, initCond, h, xend-x0, constants, DATA_SIZE, status,
           resolution);

    end = clock();

    Real timeExec = (end-start)/(double)CLOCKS_PER_SEC;

    // printf("%f\n", timeExec);
    fprintf(output, "%d, %.10f\n", IMG_COLS*IMG_ROWS, timeExec);

    // Test to see the images
    for(int row = 0; row < IMG_ROWS; row++){
        for(int col = 0; col < IMG_COLS; col++){
            int pixel = row*IMG_COLS + col;
            printf("%d, ", status[pixel]);
        }
        printf("\n");
    }

    free(status);
    free(constants);
    free(initCond);
}
