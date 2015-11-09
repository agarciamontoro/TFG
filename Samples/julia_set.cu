#include <stdio.h>

/**
 * Auxiliar struct for storing an image, typedef'd to ppm_image
 **/
typedef struct {
    int width;           //Width of the image (in px)
    int height;          //Height of the image (in px)
    unsigned char *data; //Pointer to the actual data
    size_t size;         //Pixel size (in bytes)
} ppm_image;

/**
 * Auxiliar function to save a ppm_image to a *.ppm file
 **/
size_t ppm_save(ppm_image *img, FILE *outfile) {
    size_t n = 0;
    n += fprintf(outfile, "P6\n# THIS IS A COMMENT\n%d %d\n%d\n",
                 img->width, img->height, 0xFF);
    n += fwrite(img->data, 1, img->width * img->height * 3, outfile);
    return n;
}

/**
 * Data type: complex number.
 * All of its functions -and constructor- are thought to be executed in
 * the GPU. As they are declared as __device__, even the constructor, only
 * functions executed in the GPU can use this type of data.
 *
 * The definition of the struct does not need to be declared with __device__:
 * this cuComplex declaration just declares a new type of data. As the compiler
 * only needs to know that the memory allocation -the construction of the
 * struct- and the operations regarding complex numbers -product and sum-
 * need to be done in the GPU, only this functions need to have the qualifier.
 **/
struct cuComplex{
    float r;
    float i;

    // Constructs a new complex number given its real and imaginary parts
    __device__ cuComplex( float a, float b ):
        r(a), i(b)
        {}

    // Returns the norm, squared, of the complex number.
    __device__ float magnitude2(){
        return r*r + i*i;
    }

    // Returns the product of two complex numbers.
    __device__ cuComplex operator*( const cuComplex& a ){
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    // Returns the sum of two complex numbers.
    __device__ cuComplex operator+( const cuComplex& a ){
        return cuComplex(r+a.r, i+a.i);
    }
};

/**
 * Returns whether an (x,y) point in the image belongs to Julia set.
 *
 * The function maps the (x,y) image point to a point in the complex
 * unit disk (if scale=1.0). Then, it computes a fixed number of
 * iterations of the Julia sequence: Z_{n+1} = Z_n^2 + C, where C is
 * a constant complex number.
 * If this iteration diverges; i.e., the norm of the result of the
 * iterations is greater than a fixed threshold, then the point is not
 * in the Julia set and the function returns 0.
 * In any other case, the point belongs to the Julia set and the function
 * returns 1.
 *
 * That was the first behaviour. Now, the function returns the number of
 * iterations used to know whether a point belongs to the set, always
 * normalized to 1. If the whole loop is done and the point has not
 * overpassed the threshold, then the point is considered to be in th set
 * and 1.0 is returned.
 *
 * This function is executed in the GPU and not callable from the host.
 * Obviously, all the variables used in it -including the complex numbers-
 * are stored and computed in the GPU.
 **/
__device__ float julia( int x, int y, int DIM, float scale ){
    // Interval homomorphism: [0,DIM] |-> [-scale,scale]
    float jx = scale * (float) (DIM/2 - x)/(DIM/2);
    float jy = scale * (float) (DIM/2 - y)/(DIM/2);

    int num_iter = 500;

    cuComplex c(-0.8, 0.156);
    //cuComplex c(0.285, -0.01);
    cuComplex a(jx, jy);

    int i=0;
    for (i = 0; i < num_iter; i++) {
        a = a*a + c;
        if(a.magnitude2() > 1000){
            return (float)i/num_iter; //Previously: return 0;
        }
    }

    return 1;
}

/**
 * Function executed in the GPU as many times as pixels have the final image.
 *
 * The GPU block coordinates are used as the image point coordinates. The
 * function calls julia() with its block coordinates to know how close is the
 * point to belong to the set. This number in [0,1] is used to colour the pixel.
 *
 * The first argument, ptr, is a pointer to the GPU-stored image data.
 **/
__global__ void kernel( unsigned char* ptr, int DIM, float scale ){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < DIM && y < DIM){
        // Offset in the ptr data. Usual serialization of a 2D array.
        int offset = x + y * DIM ;

        float juliaValue = julia( x,y,DIM,scale );
        ptr[offset*3 + 0] = juliaValue*255;
        ptr[offset*3 + 1] = 0;
        ptr[offset*3 + 2] = 0;
    }
}
/**
 * This program generates the fractal associated to the Julia set and saves it
 * into a ppm file.
 **/
int main( int argc, char** argv ){
    // Default values
    int DIM = 1024;
    float scale = 1.5;

    // The program can be called with two arguments: the image dimension and the
    // scale zoom; i.e., the radius of the complex disk shown in the image.
    // If there are less arguments than expected, the default values are used;
    // if there are more arguments than expected, they are ignored.
    if(argc == 2){
        DIM = atoi(argv[1]);
    }
    else if(argc > 2){
        DIM = atoi(argv[1]);
        scale = atof(argv[2]);
    }

    printf("Fractal image valuse:\n");
    printf("\tDIM\t= %d\n", DIM);
    printf("\tScale\t= %.1f\n", scale);

    // Declaration of the local (CPU-stored) image data.
    ppm_image bitmap;
    bitmap.width  = DIM;
    bitmap.height = DIM;
    bitmap.size   = 3*sizeof(unsigned char);
    bitmap.data   = (unsigned char*) malloc(DIM*DIM*3*bitmap.size);

    // Declaration of the remote (GPU-stored) image data.
    unsigned char *dev_bitmap;

    // Memory allocation in the VRAM.
    // Why do we need a pointer to our pointer and not just a pointer? http://i.imgur.com/UmpOi.gif
    // In C, we can only pass data by value or by semi-reference. Semi-reference
    // is used when two-sided data handling is needed -i.e., when the calling
    // environment and the called function should change the data- and it is done
    // through a pointer to the actual data. In this case, the actual data is a pointer.
    // so we need the address of this variable -i.e., a pointer to our pointer-.
    // But, why cudaMalloc couldn't behave like malloc and just returns the
    // pointer value? This is basically a design decision: the API designers decided to let
    // cudaMalloc return an error code, so every other assignment should be done through
    // semi-reference in the function arguments. Therefore, we need to pass the address
    // of any variable that has to be modified => we need a pointer to the pointer that will
    // point to the memory. http://www.reactiongifs.com/r/mgc.gif
    cudaMalloc((void**)&dev_bitmap, DIM*DIM*bitmap.size );

    // Grid size declaration. dim3 is basically a uni-dimensional vector with three integer
    // elements that defines the grid size. It accepts 3D grids, but if we want 1D or 2D
    // grids, it is only necessary to specify the number of desired sizes. The other ones
    // default to 1.
    int threads_per_block = 32;
    int blocks_per_grid = (DIM + (threads_per_block-1)) / threads_per_block;

    printf("Grid configuration:\n");
    printf("\tThreads per block\t: %d\n", threads_per_block);
    printf("\tBlocks per grid\t\t: %d\n", blocks_per_grid);
    printf("\tThreads x blocks\t: %d\n", threads_per_block*blocks_per_grid);

    dim3 block(threads_per_block, threads_per_block);
    dim3 grid(blocks_per_grid, blocks_per_grid);

    // The kernel call is straightforward. The numbers in the angle brackets are the folowing:
    // <<<GRID STRUCTURE, BLOCK STRUCTURE>>>
    // Then, we are defining a grid with a two-dimensional structure with size DIMxDIM and only
    // one thread per block.
    // All the computations and memory management are done in the GPU.
    kernel<<<grid,block>>>( dev_bitmap, DIM, scale );

    // We need to retrieve the remote (VRAM) data to the local (RAM) memory. Then, we copy
    // the memory from the pointer that points to the VRAM, dev_bitmap (allocated with cudaMalloc),
    // to the pointer that points to the RAM, bitmap.data (allocated with malloc).
    cudaMemcpy( bitmap.data, dev_bitmap, DIM*DIM*bitmap.size, cudaMemcpyDeviceToHost );

    // Saves the generated fractal to a ppm image.
    FILE *fp;
    fp=fopen("julia_set_image.ppm", "w");
    ppm_save(&bitmap, fp);
    fclose(fp);

    // All -VRAM and RAM- memory manually allocated need to be freed :)
    free(bitmap.data);
    cudaFree( dev_bitmap );

    printf("Done! Enjoy :)\n");

    return 0;
}
