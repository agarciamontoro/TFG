// original plumer softener is 0.025. here the value is square of it.
#define softeningSquared {{ softeningSquared }}

__global__ void galaxyKernel(float * pdata, float step, int bodies)
{

    // Body index
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockDim.x * gridDim.x + x;

    // index for global memory
    unsigned int posLoc = x * 4;
    unsigned int velLoc = y * 4;

    // position (last frame)
    float px = pdata[posLoc + 0];
    float py = pdata[posLoc + 1];
    float pz = pdata[posLoc + 2];

    // velocity (last frame)
    float vx = pdata[velLoc + 0];
    float vy = pdata[velLoc + 1];
    float vz = pdata[velLoc + 2];

    // update gravity (accumulation): naive big loop
    float3 acc = {0.0f, 0.0f, 0.0f};
    float3 r;
    float distSqr, distCube, s;

    unsigned int ni = 0;

    for (int i = 0; i < bodies; i++)
    {

        ni = i * 4;

        r.x = pdata[ni + 0] - px;
        r.y = pdata[ni + 1] - py;
        r.z = pdata[ni + 2] - pz;

        distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
        distSqr += softeningSquared;

        float dist = sqrtf(distSqr);
        distCube = dist * dist * dist;

        // Maybe we could move masses to constant or texture memory?
        s = pdata[ni + 3] / distCube;

        acc.x += r.x * s;
        acc.y += r.y * s;
        acc.z += r.z * s;

    }

    // update velocity with above acc
    vx += acc.x * step;
    vy += acc.y * step;
    vz += acc.z * step;

    // update position
    px += vx * step;
    py += vy * step;
    pz += vz * step;

    // thread synch
    __syncthreads();

    // update global memory with update value (position, velocity)
    pdata[posLoc + 0] = px;
    pdata[posLoc + 1] = py;
    pdata[posLoc + 2] = pz;
    pdata[velLoc + 0] = vx;
    pdata[velLoc + 1] = vy;
    pdata[velLoc + 2] = vz;
}
