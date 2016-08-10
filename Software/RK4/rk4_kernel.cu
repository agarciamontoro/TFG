#define SYSTEM_SIZE {{ SYSTEM_SIZE }}

/**
 * Returns the block identifier in a 2D grid
 * @return int Block identifier in which the running thread resides
 */
__device__ inline int getBlockId(){
   return blockIdx.x + blockIdx.y * gridDim.x;
}

/**
 * Returns the thread identifier in a 2D block
 * @return int Thread identifier local to the block
 */
__device__ inline int getThreadId(){
    return (threadIdx.y * blockDim.x) + threadIdx.x;
}

/**
 * Computes the value of the threadId-th component of the function
 * F(t) = (f1(t), ..., fn(t)) and stores it in the memory pointed by f
 * @param float  t  Value of the time in which the system is solved
 * @param float* y  Initial conditions for the system: a vector whose lenght
 *                  shall be the same as the number of equations in the system
 * @param float* f  Computed value of the function: a vector whose lenght
 *                  shall be the same as the number of equations in the system
 */
__device__ void computeComponent(int threadId, float x, float* y, float* f){
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
 * @param {[type]} float x0           Value of the time in which the system is
 * solved
 * @param {[type]} void  *devInitCond Pointer to a GPU array with the initial
 * conditions, also used as output for the evolution of the system.
 * @param {[type]} float dx           Step size.
 */
__global__ void RK4Solve(float x0, void *devInitCond, float dx){
    // Retrieve the identifiers of the thread in the block and of the block in
    // the grid
    int threadId = getThreadId();
    int blockId = getBlockId();

    // Retrieve the initial conditions this block will work with
    float* globalInitCond = (float*)devInitCond + blockId*SYSTEM_SIZE;

    // Copy the initial conditions to shared memory (as there are as many
    // initial conditions as equations on the system, each thread can copy one
    // of them).
    __shared__ float y0[SYSTEM_SIZE];
    y0[threadId] = globalInitCond[threadId];

    __syncthreads();

    // Auxiliar computation arrays
    __shared__ float k1[SYSTEM_SIZE],
                     k2[SYSTEM_SIZE],
                     k3[SYSTEM_SIZE],
                     k4[SYSTEM_SIZE];

    // New value of the system
    __shared__ float y1[SYSTEM_SIZE];

    // K1 computation
    computeComponent(threadId, x0, y0, k1);
    __syncthreads();

    // K2 computation
    y1[threadId] = y0[threadId] + 0.5*dx*k1[threadId];
    __syncthreads();

    computeComponent(threadId, x0 + 0.5*dx, y1, k2);
    __syncthreads();

    // K3 computation
    y1[threadId] = y0[threadId] + 0.5*dx*k2[threadId];
    __syncthreads();

    computeComponent(threadId, x0 + 0.5*dx, y1, k3);
    __syncthreads();

    // K4 computation
    y1[threadId] = y0[threadId] + dx*k3[threadId];
    __syncthreads();

    computeComponent(threadId, x0 + dx, y1, k4);
    __syncthreads();

    // Update system value in the global memory
    globalInitCond[threadId] += dx*(k1[threadId] +
                                    2*(k2[threadId]+k3[threadId]) +
                                    k4[threadId])/6;
}
