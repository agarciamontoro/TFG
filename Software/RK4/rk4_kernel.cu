#define CONSTANT_A      {{ CONSTANT_A }}
#define SYSTEM_SIZE     {{ SYSTEM_SIZE }}

// Global pointer to the system coefficients
__device__ float* globalSystemCoeffs;

// Auxiliar array to store the shared computations
__shared__ float newValue[SYSTEM_SIZE];

/**
 * Returns the block identifier in a 2D grid
 * @return int Block identifier in which the running thread resides
 */
__device__ int getBlockId(){
   return blockIdx.x + blockIdx.y * gridDim.x;
}

/**
 * Returns the thread identifier in a 2D block
 * @return int Thread identifier local to the block
 */
__device__ int getThreadId(){
    return (threadIdx.y * blockDim.x) + threadIdx.x;
}

/**
 * Returns the thread identifier in a 2D grid with 2D blocks
 * @return int Global identifier of the running thread
 */
__device__ int getGlobalId(){
   int blockId = getBlockId();
   int threadId = blockId * (blockDim.x * blockDim.y) +
                 getThreadId();

   return threadId;
}

/**
 * Computes the value of the threadId-th component of the function
 * F(t) = (f1(t), ..., fn(t)) and stores it in the memory pointed by f
 * @param float  t  Value of the time
 * @param float* y  Initial conditions
 * @param float* f  Computed value of the function
 */
__device__ void computeComponent(int comp, float t, float* y, float* f){
    int row = comp*SYSTEM_SIZE;

    f[comp] = 0;

    for(int i=0; i<SYSTEM_SIZE; i++){
        f[comp] += globalSystemCoeffs[row + i] * y[i];
    }

    __syncthreads();
}

__device__ void computeK1(int threadId, float t, float* y, float step, float* k1){
    computeComponent(threadId, t, y, k1);
}

__device__ void computeK2(int threadId, float t, float* y, float step, float* k1, float* k2){
    newValue[threadId] = y[threadId] + step*k1[threadId]/2.0;
    __syncthreads();

    computeComponent(threadId, t + step/2.0, newValue, k2);
}

__device__ void computeK3(int threadId, float t, float* y, float step, float* k2, float* k3){
    newValue[threadId] = y[threadId] + step*k2[threadId]/2.0;
    __syncthreads();

    computeComponent(threadId, t + step/2.0, newValue, k3);
}

__device__ void computeK4(int threadId, float t, float* y, float step, float* k3, float* k4){
    newValue[threadId] = y[threadId] + step*k3[threadId];
    __syncthreads();

    computeComponent(threadId, t + step, newValue, k4);
}

__global__ void rungeKutta4(void *devInitCond, void *devSystemCoeffs,
                            float step){
    // Pointers to the positions and accelerations
    float* globalInitCond = (float*)devInitCond;
    globalSystemCoeffs = (float*)devSystemCoeffs;

    // Retrieve the identifier of the thread in the block and the block
    // identifier, as well as the global identifier.
    int threadId = getThreadId();
    int blockId = getBlockId();

    // Initial conditions used by the block
    // TODO: Copy to shared memory
    float* initConditions = globalInitCond + blockId*(SYSTEM_SIZE + 1);
    float t = initConditions[0];
    float* y = initConditions + 1;

    // Auxiliar computations
    __shared__ float k1[SYSTEM_SIZE],
                     k2[SYSTEM_SIZE],
                     k3[SYSTEM_SIZE],
                     k4[SYSTEM_SIZE];

    computeK1(threadId, t, y, step, k1);
    computeK2(threadId, t, y, step, k1, k2);
    computeK3(threadId, t, y, step, k2, k3);
    computeK4(threadId, t, y, step, k3, k4);

    // Update time (do it just once)
    if(threadId == 0)
        initConditions[0] += step;

    // Update system value
    initConditions[threadId + 1] += step*(k1[threadId] +
                                          2*(k2[threadId]+k3[threadId]) +
                                          k4[threadId])/6;
}
