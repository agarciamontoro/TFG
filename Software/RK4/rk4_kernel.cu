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
__global__ void RK4Solve(Real x0, void *devInitCond, void* step, Real tolerance){
    // Retrieve the identifiers of the thread in the block and of the block in
    // the grid
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    // Assure the running thread is a useful thread :)
    if(threadId < SYSTEM_SIZE){
        // Arrays to store fourth and fifth order solutions.
        __shared__ Real rk4[SYSTEM_SIZE], rk5[SYSTEM_SIZE];

        // First try of the step size
        Real* globalStep = (Real*)step;
        Real dx = *globalStep;

        // Boolean to know if the step has to be computed again (when the error
        // exceeds the tolerance)
        bool repeatStep;

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
        __shared__ Real R, delta;

        do{
            // Reset while-condition
            repeatStep = false;

            // K1 computation
            y1[threadId] = y0;
            __syncthreads();
            computeComponent(threadId, x0, y1, k1);
            __syncthreads();

            // K2 computation
            y1[threadId] = y0 + dx*(1/4)*k1[threadId];
            __syncthreads();
            computeComponent(threadId, x0 + (1/4)*dx, y1, k2);
            __syncthreads();

            // K3 computation
            y1[threadId] = y0 + dx*((3/32)*k1[threadId] +
                                    (9/32)*k2[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (3/8)*dx, y1, k3);
            __syncthreads();

            // K4 computation
            y1[threadId] = y0 + dx*(  (1932/2197)*k1[threadId]
                                    - (7200/2197)*k2[threadId]
                                    + (7296/2197)*k3[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (12/13)*dx, y1, k4);
            __syncthreads();

            // K5 computation
            y1[threadId] = y0 + dx*( (439/216)*k1[threadId]
                                    - 8*k2[threadId]
                                    + (3680/513)*k3[threadId]
                                    - (845/4104)*k4[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + dx, y1, k5);
            __syncthreads();

            // K6 computation
            y1[threadId] = y0 + dx*(-(8/27)*k1[threadId]
                                    + 2*k2[threadId]
                                    - (3544/2565)*k3[threadId]
                                    + (1859/4104)*k4[threadId]
                                    - (11/40)*k5[threadId]);
            __syncthreads();
            computeComponent(threadId, x0 + (1/2)*dx, y1, k6);
            __syncthreads();

            // Compute fourth and fifth order solutions
            rk4[threadId] = y0 + dx*( (25/216)*k1[threadId]
                                    + (1408/2565)*k3[threadId]
                                    + (2197/4101)*k4[threadId]
                                    - (1/5)*k5[threadId]);

            rk5[threadId] = y0 + dx*( (16/135)*k1[threadId]
                                    + (6656/12825)*k3[threadId]
                                    + (28561/56430)*k4[threadId]
                                    - (9/50)*k5[threadId]
                                    + (2/55)*k6[threadId]);

            // Retrieve the local errors
            Real diff = rk5[threadId] - rk4[threadId];
            errors[threadId] = diff*diff;
            __syncthreads();

            // Compute the distance between both solutions with the usual
            // reduction technique, storing it in errors[0]. Note that the
            // number of threads has to be a power of 2.
            for(int s=(blockDim.x*blockDim.y)/2; s>0; s>>=1){
                if (threadId < s) {
                    errors[threadId] = errors[threadId] + errors[threadId + s];
                }

                __syncthreads();
            }

            if(threadId == 0){
                errors[0] = sqrt(errors[0]);
                R = errors[0]/dx;
                delta = pow((Real)0.84*(tolerance/R), (Real)0.25);
            }
            __syncthreads();

            // Update the step
            dx *= delta;

            // Repeat the step if the error exceeds the tolerance
            if(R > tolerance){
                repeatStep = true;
            }

        }while(repeatStep);

        // Update system value in the global memory.
        globalInitCond[threadId] = rk5[threadId];

        // Update global step. Do it just once.
        if(threadId == 0)
            *globalStep = dx;

    } // If threadId < SYSTEM_SIZE
}
