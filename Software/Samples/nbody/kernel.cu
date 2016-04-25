#define EPS2 {{ EPS2 }}
#define NUM_BODIES {{ NUM_BODIES }}
#define TILE_SIZE {{ TILE_SIZE }}
#define G_CONST {{ G_CONST }}
#define ep 0.67f						// 0.5f

/**
 * Computes the interaction between two bodies given their positions and the
 * acceleration of the first one
 * @param  bi            First body positions and mass
 * @param  bj            Second body positions and mass
 * @param  ai            Acceleration of the first body
 * @return               Updated acceleration for the first body
 */
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj){
    float3 r;

    // Position vector from bi to bj
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // Squared distance between bi and bj
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;

    // Inverse of the cube of the distance
    float distCube = distSqr * distSqr * distSqr;

    if (distCube < 1.0f) return make_float3(0., 0., 0.);

    float invDistCube = 1.0f/sqrtf(distCube);

    // Mass product to get final factor
    float s = G_CONST * bj.w * invDistCube * ep;

    // New acceleration computation
    float3 ai;
    ai.x = r.x * s;
    ai.y = r.y * s;
    ai.z = r.z * s;

    return ai;
}

/**
 * Computes the acceleration induced by all the bodies in a tile
 * to the body which description is passed as myPosition
 * @param  {[type]} float4 myPosition    Position and mass of the body
 * @param  {[type]} float3 accel         Current acceleration of the body
 * @param  {[type]} float4* shPosition   Descriptions of all the bodies
 *                          			 in the tile
 * @return {[type]}        Updated acceleration for the body
 */
// __device__ float3 tile_calculation(float4 myPosition, float3 accel, float4* shPosition){
//     int i;
//     // Shared memory declared and populated on the kernel
//     // extern __shared__ float4[] shPosition;
//
//     for (i = 0; i < TILE_SIZE; i++) {
//         accel += bodyBodyInteraction(myPosition, shPosition[i]);
//     }
//
//     return accel;
// }

__global__ void galaxyKernel(void *devPos, void *devVel, float step){
    // Declaration of shared memory to compute the tiles in a block
    __shared__ float4 shPosition[TILE_SIZE];

    // Pointers to the positions and accelerations
    float4* globalPos = (float4*)devPos;
    float4* globalVel = (float4*)devVel;

    // The body represented by this thread is the global identifier of the
    // thread
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // Position of the body represented by the thread
    float4 myPosition, myVelocity;
    myPosition = globalPos[gtid];
    myVelocity = globalVel[gtid];

    // Loop setup
    float3 myAcceleration = {0.0f, 0.0f, 0.0f};
    float3 currAcceleration;
    int i, tile;

    // for (i = 0, tile = 0; i < NUM_BODIES; i += TILE_SIZE, tile++) {
    //     int idx = tile * blockDim.x + threadIdx.x;
    //
    //     shPosition[threadIdx.x] = globalPos[idx];
    //     __syncthreads();
    //
    //     myAcceleration = tile_calculation(myPosition, myAcceleration, shPosition);
    //     __syncthreads();
    // }
    float4 otherBody;
    for(i = 0; i < NUM_BODIES; i++){
        otherBody = globalPos[i];
        currAcceleration = bodyBodyInteraction(myPosition, otherBody);

        myAcceleration.x += currAcceleration.x;
        myAcceleration.y += currAcceleration.y;
        myAcceleration.z += currAcceleration.z;
    }


    // Update velocity with updated acceleration
    myVelocity.x += myAcceleration.x * step;
    myVelocity.y += myAcceleration.y * step;
    myVelocity.z += myAcceleration.z * step;

    // Update position with updated velocity
    myPosition.x += myVelocity.x * step + (myAcceleration.x * step * step) / 2;
    myPosition.y += myVelocity.y * step + (myAcceleration.y * step * step) / 2;
    myPosition.z += myVelocity.z * step + (myAcceleration.z * step * step) / 2;

    __syncthreads();

    // Update global array
    globalVel[gtid] = myVelocity;
    globalPos[gtid] = myPosition;
}


// // original plumer softener is 0.025. here the value is square of it.
// #define softeningSquared {{ softeningSquared }}
//
// __global__ void galaxyKernel(float * pdata, float step, int bodies)
// {
//
//     // Body index
//     unsigned int body = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
//
//     // index for global memory
//     unsigned int posLoc = body * 4;
//     unsigned int velLoc = posLoc * 2;
//
//     // position (last frame)
//     float px = pdata[posLoc + 0];
//     float py = pdata[posLoc + 1];
//     float pz = pdata[posLoc + 2];
//
//     // velocity (last frame)
//     float vx = pdata[velLoc + 0];
//     float vy = pdata[velLoc + 1];
//     float vz = pdata[velLoc + 2];
//
//     // update gravity (accumulation): naive big loop
//     float3 acc = {0.0f, 0.0f, 0.0f};
//     float3 r;
//     float distSqr, distCube, s;
//
//     unsigned int ni = 0;
//
//     for (int i = 0; i < bodies; i++)
//     {
//
//         ni = i * 4;
//
//         r.x = pdata[ni + 0] - px;
//         r.y = pdata[ni + 1] - py;
//         r.z = pdata[ni + 2] - pz;
//
//         distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
//         distSqr += softeningSquared;
//
//         float dist = sqrtf(distSqr);
//         distCube = dist * dist * dist;
//
//         // Maybe we could move masses to constant or texture memory?
//         s = pdata[ni + 3] / distCube;
//
//         acc.x += r.x * s;
//         acc.y += r.y * s;
//         acc.z += r.z * s;
//
//     }
//
//     // update velocity with above acc
//     vx += acc.x * step;
//     vy += acc.y * step;
//     vz += acc.z * step;
//
//     // update position
//     px += vx * step;
//     py += vy * step;
//     pz += vz * step;
//
//     // thread synch
//     __syncthreads();
//
//     // update global memory with update value (position, velocity)
//     pdata[posLoc + 0] = px;
//     pdata[posLoc + 1] = py;
//     pdata[posLoc + 2] = pz;
//     pdata[velLoc + 0] = vx;
//     pdata[velLoc + 1] = vy;
//     pdata[velLoc + 2] = vz;
// }
