__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
float3 r;
// r_ij  [3 FLOPS]
r.x = bj.x - bi.x;
r.y = bj.y - bi.y;
r.z = bj.z - bi.z;
// distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
// invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
float distSixth = distSqr * distSqr * distSqr;
float invDistCube = 1.0f/sqrtf(distSixth);
// s = m_j * invDistCube [1 FLOP]
float s = bj.w * invDistCube;
// a_i =  a_i + s * r_ij [6 FLOPS]
ai.x += r.x * s;
ai.y += r.y * s;
ai.z += r.z * s;
return ai;
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
