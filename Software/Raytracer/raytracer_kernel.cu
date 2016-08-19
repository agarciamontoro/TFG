#include <stdio.h>
#include <math.h>
// #include <cuComplex.h>
#include <assert.h>

#define SYSTEM_SIZE {{ SYSTEM_SIZE }}
{{ DEBUG }}

#define J I
#define Pi M_PI

#define CELESTIAL_SPHERE 1
#define HORIZON 0

typedef {{ Real }} Real;
// typedef cuDoubleComplex Complex;

// __device__ Real cabs(const Complex& z){
//     return cuCabs(z);
// }
//
// __device__ Real carg(const Complex& z){
//     return atan2(cuCreal(z), cuCimag(z));
// }
//
// __device__ Complex cuCpow(const Complex& z, const int &n){
//     Real real = pow(cabs(z), n) * cos(n * carg(z));
//     Real imag = pow(cabs(z), n) * sin(n * carg(z));
//
//     return make_cuDoubleComplex(real, imag);
// }

// NOTE: This is b_0(r) - b.
__device__ Real b0b(Real r, Real a, Real b, Real useless){
    Real a2 = a*a;
    return(-((r*r*r - 3.*(r*r) + a2*r + a2) / (a*(r-1.))) - b);
}

__device__ Real q0(Real r, Real a){
    Real r3 = r*r*r;
    Real a2 = a*a;
    return -(r3*(r3 - 6.*(r*r) + 9.*r - 4.*a2)) / (a2*((r-1.)*(r-1.)));
}

__device__ Real R(Real r, Real a, Real b, Real q){
    Real r2 = r*r;
    Real r4 = r2*r2;
    Real a2 = a*a;
    Real b2 = b*b;
    return(r4 -q*r2 - b2*r2 + a2*r2 + 2*q*r + 2*b2*r - 4*a*b*r + 2*a2*r - a2*q);
}

__device__ Real secant( Real xA, Real xB, Real(*f)(Real, Real, Real, Real),
                        Real a, Real b, Real q )
{
    Real e = 1.0e-10;
    Real fA, fB;
    Real d;
    int i;
    int limit = 10000;

    fA=(*f)(xA, a, b, q);

    for (i=0; i<limit; i++) {
        fB=(*f)(xB, a, b, q);
        d = (xB - xA) / (fB - fA) * fB;
        if (fabs(d) < e)
            break;
        xA = xB;
        fA = fB;
        xB -= d;
    }

    if (i==limit) {
        printf("Function is not converging near (%7.4f,%7.4f).\n", xA, xB);
        return xB;
    }

    return xB;
}

__global__ void rayTrace(void* devImage, Real imageRows, Real imageCols, Real pixelWidth, Real pixelHeight, Real d, Real camR, Real camTheta, Real camPhi, Real camBeta, Real a, Real b1,Real b2, Real ro, Real delta, Real pomega, Real alpha, Real omega){
    // Retrieve the ids of the thread in the block and of the block in the grid
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId =  blockIdx.x  + blockIdx.y  * gridDim.x;

    Real* globalImage = (Real*) devImage;

    // Compute the squares once and for all
    Real a2 = a*a;

    // Compute pixel position in the physical space
    Real y = - (blockIdx.x - imageCols/2.) * pixelWidth;
    Real z =   (blockIdx.y - imageRows/2.) * pixelHeight;

    // Compute direction of the incoming ray in the camera's reference frame
    Real rayPhi = Pi + atan(y / d);
    Real rayTheta = Pi/2 + atan(z / sqrt(d*d + y*y));

    // **************************** SET NORMAL **************************** //
    // Cartesian components of the unit vector N pointing in the direction of
    // the incoming ray
    Real Nx = sin(rayTheta) * cos(rayPhi);
    Real Ny = sin(rayTheta) * sin(rayPhi);
    Real Nz = cos(rayTheta);

    // ********************** SET DIRECTION OF MOTION ********************** //
    // Compute denominator, common to all the cartesian components
    Real den = 1. - camBeta * Ny;

    // Compute factor common to nx and nz
    Real fac = -sqrt(1. - camBeta*camBeta);

    // Compute cartesian coordinates of the direction of motion. See(A.9)
    Real nY = (-Ny + camBeta) / den;
    Real nX = fac * Nx / den;
    Real nZ = fac * Nz / den;

    // Convert the direction of motion to the FIDO's spherical orthonormal
    // basis. See (A.10)
    Real nR = nX;
    Real nTheta = -nZ;
    Real nPhi = nY;

    // *********************** SET CANONICAL MOMENTA *********************** //
    // Compute energy as measured by the FIDO. See (A.11)
    Real E = 1. / (alpha + omega * pomega * nPhi);

    // Set conserved energy to unity. See (A.11)
    Real pt = -1;

    // Compute the canonical momenta. See (A.11)
    Real pR = E * ro * nR / sqrt(delta);
    Real pTheta = E * ro * nTheta;
    Real pPhi = E * pomega * nPhi;

    // ********************* SET CONSERVED QUANTITIES ********************* //
    // Set conserved quantities. See (A.12)
    Real b = pPhi;
    Real sinTheta = sin(camTheta);
    Real q = pTheta*pTheta + cos(camTheta)*(b*b / sinTheta*sinTheta - a2);

    // Compute r0 such that b0(r0) = b
    Real r0 = secant(-15., 15., b0b, a, b, 0);

    int color = 0.5;

    if(b1 < b && b < b2 && q < q0(r0, a)){
        if(pR > 0)
            color = HORIZON;
        else
            color = CELESTIAL_SPHERE;
    }
    else{

        Real rUp1 = secant(-30., 30., R, a, b, q);

        if(camR < rUp1)
            color = HORIZON;
        else
            color = CELESTIAL_SPHERE;
    }

    globalImage[3*(blockIdx.x + blockIdx.y*gridDim.x) + 0] = color;
    globalImage[3*(blockIdx.x + blockIdx.y*gridDim.x) + 1] = color;
    globalImage[3*(blockIdx.x + blockIdx.y*gridDim.x) + 2] = color;
}





// Compute r0 such that b0(r0) = b. The computation of this number involves
// complex numbers (there is a square root of a negative number).
// Nevertheless, the imaginary parts cancel each other when substracting
// the final terms. In order not to get np.sqrt errors because of the
// negative argument, a conversion to complex is forced summing a null
// imaginary part in the argument of sqrt (see the + 0J below, in the
// innerSqrt assignation). After the final computation is done, the real
// part is retrieved (the imaginary part can be considered null).
//
// // Simplify notation by computing this factor before
// fac = -9. + 3.*a2 + 3.*a*b;
// Real fac3 = fac*fac*fac;
//
// // Compute the square root of a negative number, by creating a complex with
// // real part zero and imaginary part the square root of the absolute value
// // of the number
// Real radicand = (54. - 54.*a2)*(54. - 54.*a2) + 4.*fac3;
//
// if(radicand < 0){
//     Complex innerSqrt = make_cuDoubleComplex(0., sqrt(-radicand));
//     Complex summand = make_cuDoubleComplex(54. - 54.*a2, 0.);
//
//     // Simplify notation by computing this cubic root
//     Complex base = cuCadd(innerSqrt, summand);
//     Complex cubicRoot = cuCpow(base, 1./3.);
//
//     // Finish the computation with the above computed numbers
//     Real cubicTwo = 1.2599210498948732; // pow(2, 1./3.);
//     Complex num1 = make_cuDoubleComplex(cubicTwo*fac, 0.);
//     Complex den1 = make_cuDoubleComplex(3*cuCreal(cubicRoot),
//                                         3*cuCimag(cubicRoot));
//     Complex den2 = make_cuDoubleComplex(3*cubicTwo, 0.);
//
//     Complex one = make_cuDoubleComplex(1., 0.);
//
//     Complex r0_c = cuCsub(one,
//                           cuCadd(cuCdiv(num1,den1),
//                                  cuCdiv(cubicRoot,den2)
//                                 )
//                          );
//
//     // Retrieve the real part and make sure the imaginary part is (nearly) zero
//     Real r0 = cuCreal(r0_c);
//     assert(abs(cuCimag(r0_c)) < 1e-9);
//
//     if(blockIdx.x==170 && blockIdx.y==184){
//         printf("r_0 = %.10f\n", r0);
//     }
// }
