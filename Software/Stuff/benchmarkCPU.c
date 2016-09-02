#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

#include "../RaytracerCPU/raytracer.c"

int main(int argc, char* argv[]){
    if(argc < 5){
      fprintf(stderr, "Usage: ./benchmarkCPU outputPath minSide maxSide step\n");
      exit(1);
    }

    // Retrieve output file path and limits in the benchmark loop
    char* outputPath = argv[1];
    int minSide = atoi(argv[2]);
    int maxSide = atoi(argv[3]);
    int step = atoi(argv[4]);

    // Open a file
    FILE *output = fopen(outputPath, "w");

    // Check the file opening was successful
    if (output == NULL) {
      fprintf(stderr, "Can't open output file %s!\n", outputPath);
      exit(1);
    }

    // Print CSV header into the output file
    fprintf(output, "Number of pixels, CPU time\n");

    // Run the benchmark!
    Real timeExec;
    for(int side = minSide; side < maxSide; side += step){
        // Call the kernel and get the time spent in its computation
        timeExec = measureKernelTime(side, side);

        // Print the retrieved info both to the standard output and to the
        // output file
        fprintf(output, "%d, %.10f\n", side*side, timeExec);
        printf("Side = %d. Time = %.5f\n", side, timeExec);
    }
}
