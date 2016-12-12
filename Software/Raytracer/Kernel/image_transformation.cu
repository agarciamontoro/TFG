#ifndef __IMAGE_TRANSFORMATION__
#define __IMAGE_TRANSFORMATION__

#include "Raytracer/Kernel/common.cu"

#define TXT_COLS 2363
#define TXT_ROWS 500

__global__ void generate_image(void* devRayCoordinates, void* devStatus,
                               void* devDiskTexture,
                               int diskRows, int diskCols,
                               void* devSphereTexture,
                               int sphereRows, int sphereCols,
                               void* devImage){
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
        Real* diskTexture = (Real*) devDiskTexture;
        Real* sphereTexture = (Real*) devSphereTexture;
        Real* image = (Real*) devImage;

        // Locate the image pixel that corresponds to this thread
        image += pixel * 3;

        // Copy the coordinates of the current ray to local memory
        // FIXME: Is this efficient? We only access once to the memory.
        Real rayCoords[SYSTEM_SIZE];
        memcpy(rayCoords, globalRaycoords, sizeof(Real)*SYSTEM_SIZE);

        // Variables to hold the ray coordinates
        Real r, theta, phi;

        r = rayCoords[0];
        theta = rayCoords[1];
        phi = rayCoords[2];

        // Variables to hold the texel coordinates
        int u, v, texel;

        Real rNormalized;

        switch(status){
            case DISK:
                rNormalized = (r - innerDiskRadius) / (outerDiskRadius - innerDiskRadius);

                u = round((sin(phi) + 1)/2 * diskCols);
                v = round(rNormalized * diskRows);

                texel = v*diskCols + u;
                diskTexture += texel * 3;


                // Real numBueno = fabs((phi + Pi) / (2*Pi));
                //
                //
                // if(fmod(numBueno, 2*Pi/100) < Pi/100){
                //     if(fmod(rNormalized, 0.2) < 0.1){
                //         image[0] = 1;
                //     }
                //     else{
                //         image[0] = 0;
                //     }
                // }
                // else{
                //     if(fmod(rNormalized, 0.2) < 0.1){
                //         image[0] = 0;
                //     }
                //     else{
                //         image[0] = 1;
                //     }
                // }

                unsigned p1 = 4.0 * rNormalized;
                unsigned p2 = floor(fmod(phi+2*Pi, 2*Pi) * 4.0 / (2*Pi));

                image[1] = image[2] = 0;

                if((p1 ^ p2) & 1)
                    image[0] = 1;
                else{
                    image[0] = 1;
                    image[1] = image[2] = 1;
                }



                // memcpy(image, diskTexture, 3*sizeof(Real));

                break;

            case SPHERE:
                phi = fmod(phi, 2*Pi);
                theta = fmod(theta, Pi);

                u = round(sphereCols * phi / (2*Pi));
                v = round(sphereRows * theta / Pi);

                texel = v*sphereCols + u;
                sphereTexture += texel * 3;

                memcpy(image, sphereTexture, 3*sizeof(Real));
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
