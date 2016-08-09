#define CONSTANT_A      {{ CONSTANT_A }}
#define SYSTEM_SIZE     {{ SYSTEM_SIZE }}

/**
 * Returns the thread identifier in a 2D grid with 2D blocks
 * @return int Global identifier of the running thread
 */
__device__ int getGlobalId(){
   int blockId = blockIdx.x + blockIdx.y * gridDim.x;
   int threadId = blockId * (blockDim.x * blockDim.y) +
                 (threadIdx.y * blockDim.x) + threadIdx.x;

   return threadId;
}

// TODO: The y'' = -ay system is hard-coded, generalize to accept any system
// of ODEs, accepting just a matrix A of coefficients: y' = Ay, where y and y'
// are vectors of SYSTEM_SIZE length.
/**
 * Computes the value of the function F(t) = (f1(t), ..., fn(t)) and stores it
 * in the memory pointed by f
 * @param float  t  Value of the time
 * @param float* y  Initial conditions
 * @param float* f  Computed value of the function
 */
__device__ void computeF(float t, float* y, float* f){
    f[0] = y[1];
    f[1] = -CONSTANT_A * y[0];
}

__device__ void computeK1(float t, float* y, float step, float* k1){
    computeF(t, y, k1);
}

__device__ void computeK2(float t, float* y, float step, float* k1, float* k2){
    float newY[SYSTEM_SIZE];

    for(int i=0; i<SYSTEM_SIZE; i++){
        newY[i] = y[i] + step*k1[i]/2.0;
    }

    computeF(t + step/2.0, newY, k2);
}

__device__ void computeK3(float t, float* y, float step, float* k2, float* k3){
    float newY[SYSTEM_SIZE];

    for(int i=0; i<SYSTEM_SIZE; i++){
        newY[i] = y[i] + step*k2[i]/2.0;
    }

    computeF(t + step/2.0, newY, k3);
}

__device__ void computeK4(float t, float* y, float step, float* k3, float* k4){
    float newY[SYSTEM_SIZE];

    for(int i=0; i<SYSTEM_SIZE; i++){
        newY[i] = y[i] + step*k3[i];
    }

    computeF(t + step, newY, k4);
}

__global__ void rungeKutta4(void *devInitCond, float step){
    // Pointers to the positions and accelerations
    float* globalInitCond = (float*)devInitCond;

    // Retrieve the global identifier of the thread
	int gtid = getGlobalId();

    // Initial conditions for the thread
    float* initConditions = globalInitCond + gtid*(SYSTEM_SIZE + 1);
    float t = initConditions[0];
    float* y = initConditions + 1;

    // Auxiliar computations
    float k1[SYSTEM_SIZE], k2[SYSTEM_SIZE], k3[SYSTEM_SIZE], k4[SYSTEM_SIZE];
    computeK1(t, y, step, k1);
    computeK2(t, y, step, k1, k2);
    computeK3(t, y, step, k2, k3);
    computeK4(t, y, step, k3, k4);

    // Update state
    initConditions[0] += step;
    for(int i=0; i<SYSTEM_SIZE; i++){
        initConditions[i+1] += step*(k1[i] + 2*(k2[i]+k3[i]) + k4[i])/6;
    }
}
