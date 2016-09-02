#ifndef __IMAGE_TRANSFORMATION__
#define __IMAGE_TRANSFORMATION__

#include "Raytracer/Kernel/common.cu"

#define TXT_COLS 2363
#define TXT_ROWS 500

__global__ void generate_image(void* devRayCoordinates, void* devStatus,
                               void* devTexture, void* devImage){
    // Compute pixel's row and col of this thread
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < IMG_ROWS && col < IMG_COLS){
        // Compute pixel unique identifier
        int pixel = row*IMG_COLS + col;

        // Retrieve status of the current pixel
        int* globalStatus = (int*) devStatus;
        globalStatus += pixel;
        int status = *globalStatus;

        // Locate the coordinates of the current ray
        Real* globalRaycoords = (Real*) devRayCoordinates;
        globalRaycoords += pixel * SYSTEM_SIZE;

        // Retrieve image and texture pointers
        Real* texture = (Real*) devTexture;
        Real* image = (Real*) devImage;

        // Locate the image pixel that corresponds to this thread
        image += pixel * 3;

        // Copy the coordinates of the current ray to local memory
        // FIXME: Is this efficient? We only access once to the memory.
        Real rayCoords[SYSTEM_SIZE];
        memcpy(rayCoords, globalRaycoords, sizeof(Real)*SYSTEM_SIZE);

        // Variables to hold the ray coordinates
        Real r, phi;

        // Variables to hold the texel coordinates
        int x, y, texel;

        switch(status){
            case DISK:
                r = (rayCoords[0] - innerDiskRadius) / (outerDiskRadius - innerDiskRadius);
                phi = rayCoords[2];

                x = round((sin(phi) + 1)/2 * TXT_COLS);
                y = round(r * TXT_ROWS);

                texel = y*TXT_COLS + x;
                texture += texel * 3;

                memcpy(image, texture, 3*sizeof(Real));

                break;

            case SPHERE:
                image[0] = 1;
                image[1] = 1;
                image[2] = 1;
                break;

            case HORIZON:
                image[0] = 0;
                image[1] = 0;
                image[2] = 0;
                break;
        }
    }

}

#endif // __IMAGE_TRANSFORMATION__
