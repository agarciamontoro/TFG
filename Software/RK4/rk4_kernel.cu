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
 * @param float  t  Value of the time
 * @param float* y  Initial conditions
 * @param float* f  Computed value of the function
 */
__device__ void computeComponent(int threadId, float x, float* y, float* f){
    switch(threadId) {
        {% for i, function in SYSTEM_FUNCTIONS %}
            case {{ i }}:
                f[threadId] = {{ function }};
                break;
        {% endfor %}
    }
}

__global__ void RK4Solve(float x0, void *devInitCond, float dx){
    // Retrieve the identifier of the thread in the block
    int threadId = getThreadId();
    int blockId = getBlockId();

    // Pointers to the initial conditions
    float* globalInitCond = (float*)devInitCond;
    float* y0 = globalInitCond + blockId*SYSTEM_SIZE;


    // Auxiliar computations
    __shared__ float k1[SYSTEM_SIZE],
                     k2[SYSTEM_SIZE],
                     k3[SYSTEM_SIZE],
                     k4[SYSTEM_SIZE];

    // New value
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

    // Update system value
    y0[threadId] += dx*(k1[threadId] + 2*(k2[threadId]+k3[threadId]) +
                        k4[threadId])/6;
}
