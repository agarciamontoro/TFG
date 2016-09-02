#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "Raytracer/Kernel/common.cu"

__global__ void generate_image(void* devRayCoordinates, void* devStatus,
                               Real* result_image){

    // Compute pixel's row and col of this thread
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int pixel = row*IMG_COLS + col;
   
    int* globalStatus = (int*) devStatus;
    globalStatus += pixel;
    int status = *globalStatus;


    Real* globalRaycoords = (Real*) devRayCoordinates;
    globalRaycoords += pixel * SYSTEM_SIZE;

    Real rayCoords[SYSTEM_SIZE];
    memcpy(rayCoords, globalRaycoords, sizeof(Real)*SYSTEM_SIZE);

    if( status == 1 ){

    
    Real r     = (rayCoords[0] - innerDiskRadius) / outerDiskRadius;
    Real phi   = rayCoords[2];

    Real x = r * sin(phi);
    Real y = r * cos(phi);

    result_image[pixel] = x;

    }

}
