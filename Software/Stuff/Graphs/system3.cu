#ifndef __FUNCTIONS__
#define __FUNCTIONS__

typedef double Real;
#define SYSTEM_SIZE 3

/**
* Computes the value of the threadId-th component of the function
* F(t) = (f1(t), ..., fn(t)) and stores it in the memory pointed by f
 * @param  int   threadId      Identifier of the calling thread.
 * @param  Real  x             Value of the time in which the system is solved
 * @param  Real* y             Initial conditions for the system: a pointer to
 *                             a vector whose lenght shall be the same as the
 *                             number of equations in the system.
 * @param  Real* f             Computed value of the function: a pointer to a
 *                             vector whose lenght shall be the same as the
 *                             number of equations in the system.
 * @param  Real* data          Additional data needed by the function, managed
 *                             by the caller.
 */
__device__ void computeComponent(int threadId, Real x, Real* y, Real* f,
                                 Real* data){
    switch(threadId) {
            case 0:
                f[threadId] = -y[1] + y[2];
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;

            case 1:
                f[threadId] = 5*y[0] + y[2];
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;

            case 2:
                f[threadId] = -y[1]*y[2];
                // printf("Solution[%d] = %.20f\n", threadId, f[threadId]);
                break;
    }
}

#endif // __FUNCTIONS__
