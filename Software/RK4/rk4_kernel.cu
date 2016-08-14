#include <stdio.h>

#define SYSTEM_SIZE {{ SYSTEM_SIZE }}
{{ DEBUG }}

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
}

/**
 * Computes a step of the Runge Kutta 4 algorithm, storing the results in the
 * GPU array pointed by devInitCond.
 * @param {[type]} Real x0           Value of the time in which the system is
 * solved
 * @param {[type]} void  *devInitCond Pointer to a GPU array with the initial
 * conditions, also used as output for the evolution of the system.
 * @param {[type]} Real h           Step size.
 * @param {[type]} Real tolerance    Error tolerance, used in the adaptative
 *                      step size computation.
 * @return Real The new step size.
 */
 __global__ void RK4Solve(void* devX0, Real xend, void *devInitCond, Real h,
                          Real hmax, void* globalRtoler, void* globalAtoler, Real safe, Real fac1, Real fac2, Real beta,
                          Real uround){

    // Retrieve the ids of the thread in the block and of the block in the grid
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    #ifdef DEBUG
        printf("ThreadId %d - INITS: x0=%.20f, xend=%.20f, y0=(%.20f, %.20f)\n", threadId, *((Real*)devX0), xend, ((Real*)devInitCond)[0], ((Real*)devInitCond)[1]);
    #endif

    // Assure the running thread is a useful thread :)
    if(threadId < SYSTEM_SIZE){
        // Arrays to store intermediate solutions.
        __shared__ Real solution[SYSTEM_SIZE];

        // First try of the step size
        // TODO: Implement the hinit method
        Real hnew;

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
                        k6[SYSTEM_SIZE],
                        k7[SYSTEM_SIZE];
        __shared__ Real y1[SYSTEM_SIZE];

        // Local errors
        __shared__ Real errors[SYSTEM_SIZE];
        Real err;

        // Initial values for the loop variables
        Real facold = 1.0E-4;
        Real expo1 = 0.2 - beta * 0.75;
        Real facc1 = 1.0 / fac1;
        Real facc2 = 1.0 / fac2;

        // Error tolerances
        Real* atoler = (Real*) globalAtoler;
        Real* rtoler = (Real*) globalRtoler;
        Real atoli = atoler[threadId];
        Real rtoli = rtoler[threadId];

        // More stuff
        bool last  = false;
        Real fac11, fac;
        Real sqr;

        Real reject = false;


        do{
            // TODO: Check that the step size is not too small

            if ((x0 + 1.01*h - xend) > 0.0){
              h = xend - x0;
              last = true;
            }

            // K1 computation
            y1[threadId] = y0;
            __syncthreads();
            computeComponent(threadId, x0, y1, k1);
            __syncthreads();

            // K2 computation
            y1[threadId] = y0 + h*(1./5.)*k1[threadId];
            __syncthreads();
            computeComponent(threadId, x0 + (1./5.)*h, y1, k2);
            __syncthreads();

            // K3 computation
            y1[threadId] = y0 + h*((3./40.)*k1[threadId] +
                                    (9./40.)*k2[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (3./10.)*h, y1, k3);
            __syncthreads();

            // K4 computation
            y1[threadId] = y0 + h*(  (44./45.)*k1[threadId]
                                    - (56./15.)*k2[threadId]
                                    + (32./9.)*k3[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (4./5.)*h, y1, k4);
            __syncthreads();

            // K5 computation
            y1[threadId] = y0 + h*( (19372./6561.)*k1[threadId]
                                    - (25360./2187.)*k2[threadId]
                                    + (64448./6561.)*k3[threadId]
                                    - (212./729.)*k4[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (8./9.)*h, y1, k5);
            __syncthreads();

            // K6 computation
            y1[threadId] = y0 + h*((9017./3168.)*k1[threadId]
                                    - (355./33.)*k2[threadId]
                                    + (46732./5247.)*k3[threadId]
                                    + (49./176.)*k4[threadId]
                                    - (5103./18656.)*k5[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + h, y1, k6);
            __syncthreads();

            // K7 computation. Maybe store it in K2 and get rid of K7? :D
            y1[threadId] = y0 + h*((35./384.)*k1[threadId]
                                    + (500./1113.)*k3[threadId]
                                    + (125./192.)*k4[threadId]
                                    - (2187./6784.)*k5[threadId]
                                    + (11./84.)*k6[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + h, y1, k7);
            __syncthreads();

            // Compute solution and local error
            solution[threadId] = y1[threadId];
            errors[threadId] = h*((71./57600.)*k1[threadId]
                                - (71./16695.)*k3[threadId]
                                + (71./1920.)*k4[threadId]
                                - (17253./339200.)*k5[threadId]
                                + (22./525.)*k6[threadId]
                                - (1./40.)*k7[threadId]);

            // Compute scale factor
            Real sk = atoli + rtoli*fmax(abs(y0), abs(solution[threadId]));

            // Compute the summands of the total error
            sqr = (errors[threadId])/sk;
            errors[threadId] = sqr*sqr;
            __syncthreads();

            // Add the local errors with the usual reduction technique, storing
            // it in errors[0]. Note that the number of threads has to be a
            // power of 2.
            for(int s=(blockDim.x*blockDim.y)/2; s>0; s>>=1){
                if (threadId < s) {
                    errors[threadId] = errors[threadId] + errors[threadId + s];
                }

                __syncthreads();
            }

            // Compute the total error
            err = sqrt(errors[0]/(Real)SYSTEM_SIZE);

            #ifdef DEBUG
                printf("ThreadId %d - SOLUS: %.20f, %.20f\n", threadId, solution[threadId], errors[threadId]);
                printf("ThreadId %d - QUOTI: %.20f, %.20f\n", threadId, sqr, sk);
                printf("ThreadId %d - K1234: K1:%.7f, K2:%.7f, K3:%.7f, K4:%.7f, K5:%.7f, K6:%.7f\n", threadId, k1[threadId], k2[threadId], k3[threadId], k4[threadId], k5[threadId], k6[threadId]);
                printf("ThreadId %d - ERROR: %.20f\n", threadId, errors[threadId]);
            #endif

            /* computation of hnew */
            fac11 = pow (err, expo1);
            /* Lund-stabilization */
            fac = fac11 / pow(facold,beta);
            /* we require fac1 <= hnew/h <= fac2 */
            fac = fmax(facc2, fmin(facc1, fac/safe));
            hnew = h / fac;

            #ifdef DEBUG
                printf("ThreadId %d - PREVH: fac11:%.20f, fac:%.20f, fa11/facold**beta:%.20f, facc2:%.20f, fac2:%.20f\n", threadId, fac11, fac, fac11/pow(facold, beta), facc2, fac2);
                printf("ThreadId %d - H NEW: %.20f, %.20f\n", threadId, hnew);
            #endif

            // STEP REJECTED
            if( err > 1.){
                hnew = h / fmin(facc1, fac11/safe);
                reject = true;
            }
            // STEP ACCEPTED
            else{
                // TODO: Stiffness detection

                facold = fmax(err, 1.0e-4);
                x0 += h;

                if (hnew > hmax)
                    hnew = hmax;

                if (reject)
                    hnew = fmin(fabs(hnew), fabs(h));

                y0 = solution[threadId];

                reject = false;
            }

            h = hnew;

            #ifdef DEBUG
                if(threadId == 0){
                    if(err > 1.){
                        printf("\n###### CHANGE: err: %.20f, h: %.20f\n\n", err, h);
                    }
                    else{
                        printf("\n###### ======:  err: %.20f, h: %.20f\n\n", err, h);
                    }
                }
            #endif
        }while(!last);

        // Update system value in the global memory.
        globalInitCond[threadId] = solution[threadId];

        // Update global step and time. Do it just once.
        if(threadId == 0){
            *globalX0 = x0;
        }

    } // If threadId < SYSTEM_SIZE
}
