#include <stdio.h>

#define SYSTEM_SIZE {{ SYSTEM_SIZE }}

typedef {{ Real }} Real;

/**
 * Computes the value of the threadId-th component of the function
 * F(t) = (f1(t), ..., fn(t)) and stores it in the memory pointed by f
 * @param Real  t  Value of the time in which the system is solved
 * @param Real* y  Initial conditions for the system: a vector whose lenght
 *                  shall be the same as the number of equations in the system
 * @param Real* f  Computed value of the function: a vector whose lenght
 *                  shall be the same as the number of equations in the system
 */
__device__ void computeComponent(int threadId, Real x, Real* y, Real* f){
    // Jinja template that renders to a switch in which every thread computes
    // a different equation and stores it in the corresponding position in f
    switch(threadId) {
        {% for i, function in SYSTEM_FUNCTIONS %}
            case {{ i }}:
                f[threadId] = {{ function }};
                break;
        {% endfor %}
    }
    printf("ThreadId %d - COMPU: %.5f, %.5f, %.5f, %5f\n", threadId, x, y[0], y[1], f[threadId]);
}

/**
 * Computes a step of the Runge Kutta 4 algorithm, storing the results in the
 * GPU array pointed by devInitCond.
 * @param {[type]} Real x0           Value of the time in which the system is
 * solved
 * @param {[type]} void  *devInitCond Pointer to a GPU array with the initial
 * conditions, also used as output for the evolution of the system.
 * @param {[type]} Real dx           Step size.
 * @param {[type]} Real tolerance    Error tolerance, used in the adaptative
 *                      step size computation.
 * @return Real The new step size.
 */
__global__ void RK4Solve(void* devX0, void *devInitCond, void* devStep, Real tolerance){
    // Retrieve the identifiers of the thread in the block and of the block in
    // the grid
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    // Assure the running thread is a useful thread :)
    if(threadId < SYSTEM_SIZE){
        // Arrays to store fourth and fifth order solutions.
        __shared__ Real rk4[SYSTEM_SIZE], rk5[SYSTEM_SIZE];

        // First try of the step size
        Real* globalStep = (Real*)devStep;
        Real dx = *globalStep;

        // Time
        Real* globalX0 = (Real*)devX0;
        Real x0 = *globalX0;

        // Retrieve the initial conditions this block will work with
        Real* globalInitCond = (Real*)devInitCond + blockId*SYSTEM_SIZE;

        // Get the initial condition this thread will work with
        Real y0 = globalInitCond[threadId];

        // Auxiliar computation arrays
        __shared__ Real k1[SYSTEM_SIZE],
                        k2[SYSTEM_SIZE],
                        k3[SYSTEM_SIZE],
                        k4[SYSTEM_SIZE],
                        k5[SYSTEM_SIZE],
                        k6[SYSTEM_SIZE];

        // New value of the system
        __shared__ Real y1[SYSTEM_SIZE];

        // Local errors
        __shared__ Real errors[SYSTEM_SIZE];
        Real delta, R = 0.0;
        Real err;

        do{
            // K1 computation
            y1[threadId] = y0;
            __syncthreads();
            computeComponent(threadId, x0, y1, k1);
            __syncthreads();

            // K2 computation
            y1[threadId] = y0 + dx*(1./4.)*k1[threadId];
            __syncthreads();
            computeComponent(threadId, x0 + (1./4.)*dx, y1, k2);
            __syncthreads();

            // K3 computation
            y1[threadId] = y0 + dx*((3./32.)*k1[threadId] +
                                    (9./32.)*k2[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (3./8.)*dx, y1, k3);
            __syncthreads();

            // K4 computation
            y1[threadId] = y0 + dx*(  (1932./2197.)*k1[threadId]
                                    - (7200./2197.)*k2[threadId]
                                    + (7296./2197.)*k3[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (12./13.)*dx, y1, k4);
            __syncthreads();

            // K5 computation
            y1[threadId] = y0 + dx*( (439./216.)*k1[threadId]
                                    - 8.*k2[threadId]
                                    + (3680./513.)*k3[threadId]
                                    - (845./4104.)*k4[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + dx, y1, k5);
            __syncthreads();

            // K6 computation
            y1[threadId] = y0 + dx*(-(8./27.)*k1[threadId]
                                    + 2.*k2[threadId]
                                    - (3544./2565.)*k3[threadId]
                                    + (1859./4104.)*k4[threadId]
                                    - (11./40.)*k5[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (1/2)*dx, y1, k6);
            __syncthreads();

            // Compute fourth and fifth order solutions
            rk4[threadId] = y0 + dx*( (25./216.)*k1[threadId]
                                    + (1408./2565.)*k3[threadId]
                                    + (2197./4101.)*k4[threadId]
                                    - (1./5.)*k5[threadId]);

            rk5[threadId] = y0 + dx*( (16./135.)*k1[threadId]
                                    + (6656./12825.)*k3[threadId]
                                    + (28561./56430.)*k4[threadId]
                                    - (9./50.)*k5[threadId]
                                    + (2./55.)*k6[threadId]);

            // Real sc = tolerance*(1 + fmax(y0, rk5[threadId]));
            //
            // // Retrieve the local errors
            // Real quotient = (rk5[threadId] - rk4[threadId])/tolerance;
            // errors[threadId] = quotient*quotient;
            // __syncthreads();
            //
            // printf("ThreadId %d - QUOTI: %.20f, %.20f\n", threadId, quotient, tolerance);
            //
            // printf("ThreadId %d - K1234: K1:%.7f, K2:%.7f, K3:%.7f, K4:%.7f, K5:%.7f, K6:%.7f\n", threadId, k1[threadId], k2[threadId], k3[threadId], k4[threadId], k5[threadId], k6[threadId]);
            // printf("ThreadId %d - RK4 5: %.20f, %.20f\n", threadId, rk4[threadId], rk5[threadId]);
            // printf("ThreadId %d - ERROR: %.20f\n", threadId, errors[threadId]);
            //
            // // Compute the distance between both solutions with the usual
            // // reduction technique, storing it in errors[0]. Note that the
            // // number of threads has to be a power of 2.
            // for(int s=(blockDim.x*blockDim.y)/2; s>0; s>>=1){
            //     if (threadId < s) {
            //         printf("ThreadId %d - SUMMS: S: %d, error: (%.10f, %.10f)\n", threadId, s, errors[threadId], errors[threadId+s]);
            //         errors[threadId] = errors[threadId] + errors[threadId + s];
            //     }
            //
            //     __syncthreads();
            // }
            //
            //
            // if(threadId == 0)
            //     printf("ThreadId %d - SUMMS: GLOBAL ERROR: %.20f\n", threadId, errors[0]);
            //
            // err = sqrt(errors[0]/SYSTEM_SIZE);
            //
            // #define FACMAX 1.5
            // #define FACMIN 0.1
            // #define FAC 0.8
            // dx *= fmin(FACMAX, fmax(FACMIN, FAC*pow(1./err, 0.2)));

            // // Update the step
            // R = sqrt(errors[0])/dx;
            // if(R > tolerance){
            //     delta = pow((Real)0.84*(tolerance/R), (Real)0.25);
            //     dx *= delta;
            // }

            err = 0.;

            if(threadId == 0){
                if(err > 1.){
                    printf("\n###### CHANGE: err: %.20f, dx: %.20f\n\n", err, dx);
                }
                else{
                    printf("\n###### ======:  err: %.20f, dx: %.20f\n\n", err, dx);
                }
            }
        }while(err > 1.);

        // Update system value in the global memory.
        globalInitCond[threadId] = rk5[threadId];

        // Update global step and time. Do it just once.
        if(threadId == 0){
            *globalStep = dx;
            *globalX0 += dx;
        }

    } // If threadId < SYSTEM_SIZE
}
