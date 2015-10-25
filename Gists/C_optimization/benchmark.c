#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Matrix_mult.h"
#include "Matrix_mult_opt.h"


/**
 * Aux function for substracting timespec structures
*/
struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

int main(int argc, char** argv){
    // Matrices sizes
    const int INIT_SIZE = 10;
    const int SIZE_STEP = 10;
    const int MAX_SIZE = 1000;

    // Variables for measuring time
    struct timespec t_start, t_end, times[2];

    // The program expects the user to pass the L1 cache line size as
    // its first argument. Needed in the second benchmarked function.
    int cache_line_size = atoi(argv[1]);


    // Matrices declaration
    double **src1 = NULL, **src2 = NULL, **dest = NULL;

    // Loop variables
    int i,j,k,SIZE;

    for(SIZE = INIT_SIZE; SIZE < MAX_SIZE; SIZE += SIZE_STEP) {
        // Memory allocation
        src1 = (double**)malloc(SIZE*sizeof(double*));
        src2 = (double**)malloc(SIZE*sizeof(double*));
        dest = (double**)malloc(SIZE*sizeof(double*));

        for(i = 0; i < SIZE; i++) {
            src1[i] = (double*)malloc(SIZE*sizeof(double));
            src2[i] = (double*)malloc(SIZE*sizeof(double));
            dest[i] = (double*)malloc(SIZE*sizeof(double));

            // Initialization with stupid values
            for(j = 0; j < SIZE; j++) {
                src1[i][j] = i;
                src2[i][j] = j;
            }
        }

        // First benchmarking
        clock_gettime(CLOCK_REALTIME, &t_start);
            matrix_mult(src1,src2,dest,SIZE);
        clock_gettime(CLOCK_REALTIME, &t_end);

        times[0] = diff(t_start, t_end);

        // Second benchmarking
        clock_gettime(CLOCK_REALTIME, &t_start);
            matrix_mult_opt(src1,src2,dest,SIZE,cache_line_size);
        clock_gettime(CLOCK_REALTIME, &t_end);

        times[1] = diff(t_start, t_end);

        // Information output
        printf("%d\t%d.%d\t%d.%d\n", SIZE,
            times[0].tv_sec, times[0].tv_nsec,
            times[1].tv_sec, times[1].tv_nsec);

        // Memory allocated needs to be cleaned
        for (size_t i = 0; i < SIZE; i++) {
            free(src1[i]);
            free(src2[i]);
            free(dest[i]);
        }

        free(src1); free(src2); free(dest);
    }
}
