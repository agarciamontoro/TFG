#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "Raytracer/Kernel/common.cu"
#include "Raytracer/Kernel/solvers.cu"
#include "Raytracer/Kernel/image_transformation.cu"

#define Pi M_PI
#define SYSTEM_SIZE 5

/**
 * Given the ray's incoming direction on the camera's local sky, rayTheta and
 * rayPhi, this function computes its canonical momenta. See Thorne's paper,
 * equation (A.11), for more information.
 * Note that the computation of this quantities depends on the constants
 * __camBeta (speed of the camera) and __ro, __alpha, __omega, __pomega and
 * __ro  (Kerr metric constants), that are defined in the common.cu template.
 * @param[in]       Real  rayTheta      Polar angle, or inclination, of the
 *                        ray's incoming direction on the camera's local sky.
 * @param[in]       Real  rayPhi        Azimuthal angle, or azimuth, of the
 *                        ray's incoming direction on the camera's local sky.
 * @param[out]      Real* pR            Computed covariant coordinate r of the
 *                        ray's 4-momentum.
 * @param[out]      Real* pTheta        Computed covariant coordinate theta of
 *                        the ray's 4-momentum.
 * @param[out]      Real* pPhi          Computed covariant coordinate phi of
 *                        the ray's 4-momentum.
 */
__device__ void getCanonicalMomenta(Real rayTheta, Real rayPhi, Real* pR,
                                    Real* pTheta, Real* pPhi){
    // **************************** SET NORMAL **************************** //
    // Cartesian components of the unit vector N pointing in the direction of
    // the incoming ray
    Real Nx = sin(rayTheta) * cos(rayPhi);
    Real Ny = sin(rayTheta) * sin(rayPhi);
    Real Nz = cos(rayTheta);

    // ********************** SET DIRECTION OF MOTION ********************** //
    // Compute denominator, common to all the cartesian components
    Real den = 1. - __camBeta * Ny;

    // Compute factor common to nx and nz
    Real fac = -sqrt(1. - __camBeta*__camBeta);

    // Compute cartesian coordinates of the direction of motion. See(A.9)
    Real nY = (-Ny + __camBeta) / den;
    Real nX = fac * Nx / den;
    Real nZ = fac * Nz / den;

    // Convert the direction of motion to the FIDO's spherical orthonormal
    // basis. See (A.10)
    Real nR = nX;
    Real nTheta = -nZ;
    Real nPhi = nY;

    // *********************** SET CANONICAL MOMENTA *********************** //
    // Compute energy as measured by the FIDO. See (A.11)
    Real E = 1. / (__alpha + __omega * __pomega * nPhi);

    // Set conserved energy to unity. See (A.11)
    // Real pt = -1;

    // Compute the canonical momenta. See (A.11)
    *pR = E * __ro * nR / sqrt(__delta);
    *pTheta = E * __ro * nTheta;
    *pPhi = E * __pomega * nPhi;
}

/**
 * Given the ray's canonical momenta, this function computes its constants b
 * (the axial angular momentum) and q (Carter constant). See Thorne's paper,
 * equation (A.12), for more information.
 * Note that the computation of this quantities depends on the constant
 * __camTheta, which is the inclination of the camera with respect to the black
 * hole, and that is defined in the common.cu template
 * @param[in]       Real  pTheta        Covariant coordinate theta of the ray's
 *                        4-momentum.
 * @param[in]       Real  pPhi          Covariant coordinate phi of the ray's
 *                        4-momentum.
 * @param[out]      Real* b             Computed axial angular momentum.
 * @param[out]      Real* q             Computed Carter constant.
 */
__device__ void getConservedQuantities(Real pTheta, Real pPhi, Real* b,
                                       Real* q){
    // ********************* GET CONSERVED QUANTITIES ********************* //
    // Compute axial angular momentum. See (A.12).
    *b = pPhi;

    // Compute Carter constant. See (A.12).
    Real sinT = sin(__camTheta);
    Real sinT2 = sinT*sinT;

    Real cosT = cos(__camTheta);
    Real cosT2 = cosT*cosT;

    Real pTheta2 = pTheta*pTheta;
    Real b2 = pPhi*pPhi;

    *q = pTheta2 + cosT2*((b2/sinT2) - __a2);
}

/**
 * CUDA kernel that computes the initial conditions (r, theta, phi, pR, pPhi)
 * and the constants (b, q) of every ray in the simulation.
 *
 * This method depends on the shape of the CUDA grid: it is expected to be a 2D
 * matrix with at least IMG_ROWS threads in the Y direction and IMG_COLS
 * threads in the X direction. Every pixel of the camera is assigned to a
 * single thread that computes the initial conditions and constants of its
 * corresponding ray, following a pinhole camera model.
 *
 * Each thread that executes this method implements the following algorithm:
 * 		1. Compute the pixel physical coordinates, considering the center of
 * 		the sensor as the origin and computing the physical position using the
 * 		width and height of each pixel.
 * 		2. Compute the ray's incoming direction, theta and phi, on the camera's
 * 		local sky, following the pinhole camera model defined by the sensor
 * 		shape and the focal distance __d.
 * 		3. Compute the canonical momenta pR, pTheta and pPhi with the method
 * 		`getCanonicalMomenta`.
 * 		4. Compute the ray's constants b and q with the method.
 * 		`getConservedQuantities`.
 * 		5. Fill the pixel's corresponding entry in the global array pointed by
 * 		devInitCond with the initial conditions: __camR, __camTheta, __camPhi,
 * 		pR and pTheta, where the three first components are constants that
 * 		define the position of the focal point on the black hole coordinate
 * 		system.
 * 		6. Fill the pixel's corresponding entry in the global array pointed by
 * 		devConstants with the computed constants: b and q.
 *
 * @param[out]     void* devInitCond  Device pointer to a serialized 2D matrix
 *                       where each entry corresponds to a single pixel in the
 *                       camera sensor. If the sensor has R rows and C columns,
 *                       the vector pointed by devInitCond contains R*C
 *                       entries, where each entry is a 5-tuple prepared to
 *                       receive the initial conditions of a ray: (r, theta,
 *                       phi, pR, pPhi). At the end of this kernel, the array
 *                       pointed by devInitCond is filled with the initial
 *                       conditions of every ray.
 * @param[out]     void* devConstants  Device pointer to a serialized 2D matrix
 *                       where each entry corresponds to a single pixel in the
 *                       camera sensor. If the sensor has R rows and C columns,
 *                       the vector pointed by devConstants contains R*C
 *                       entries, where each entry is a 2-tuple prepared to
 *                       receive the constants of a ray: (b, q). At the end of
 *                       this kernel, the array pointed by devConstants is
 *                       filled with the computed constants of every ray.
 * @param[in]      Real  pixelWidth   Width, in physical units, of the camera's
 *                       pixels.
 * @param[in]      Real  pixelHeight  Height, in physical units, of the
 *                       camera's pixels.
 */
__global__ void setInitialConditions(void* devInitCond,void* devConstants,
                                     Real pixelWidth, Real pixelHeight){
    // Each pixel is assigned to a single thread thorugh the grid and block
    // configuration, both of them being 2D matrices:
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // The blocks have always a multiple of 32 threads, configured in a 2D
    // shape. As it is possible that there are more threads than pixels, we
    // have to make sure that only the threads that have an assigned pixel are
    // running.
    if(row < IMG_ROWS && col < IMG_COLS){
        // Compute pixel unique identifier for this thread
        int pixel = row*IMG_COLS + col;

        // Compute the position in the global array to store the initial
        // conditions of this ray
        Real* globalInitCond = (Real*) devInitCond;
        Real* initCond = globalInitCond + pixel*SYSTEM_SIZE;

        // Compute the position in the global array to store the constants of
        // this ray
        Real* globalConstants = (Real*) devConstants;
        Real* constants = globalConstants + pixel*2;

        // Compute pixel position in physical units
        Real x = - (col + 0.5 - IMG_COLS/2) * pixelWidth;
        Real y = (row + 0.5 - IMG_ROWS/2) * pixelHeight;

        // Compute direction of the incoming ray in the camera's reference
        // frame
        Real rayPhi = Pi + atan(x / __d);
        Real rayTheta = Pi/2 + atan(y / sqrt(__d*__d + x*x));

        // Compute canonical momenta of the ray and the conserved quantites b
        // and q
        Real pR, pTheta, pPhi, b, q;
        getCanonicalMomenta(rayTheta, rayPhi, &pR, &pTheta, &pPhi);
        getConservedQuantities(pTheta, pPhi, &b, &q);

        // Save ray's initial conditions in the global array
        initCond[0] = __camR;
        initCond[1] = __camTheta;
        initCond[2] = __camPhi;
        initCond[3] = pR;
        initCond[4] = pTheta;

        // Save ray's constants in the global array
        constants[0] = b;
        constants[1] = q;
    }
}

/**
 * CUDA kernel that integrates a set of photons backwards in time from x0 to
 * xend, storing the final results of their position and canonical momenta on
 * the array pointed by devInitCond.
 *
 * This method depends on the shape of the CUDA grid: it is expected to be a 2D
 * matrix with at least IMG_ROWS threads in the Y direction and IMG_COLS
 * threads in the X direction. Every ray is assigned to a single thread, which
 * computes its final state solving the ODE system defined by the relativistic
 * spacetime.
 *
 * Each thread that executes this method implements the following algorithm:
 * 		1. Copy the initial conditions and constants of the ray from its
 * 		corresponding position at the global array devInitCond and devData into
 * 		local memory.
 * 		2. Integrate the ray's equations defined in Thorne's paper, (A.15).
 * 		This is done while continuosly checking whether the ray has collided
 * 		with disk or horizon.
 * 		3. Overwrite the conditions at devInitCond to the new computed ones.
 * 		Fill the ray's final status (no collision, collision with the disk or
 * 		collision with the horizon) in the devStatus array.
 *
 * @param[in]       Real  x0             Start of the integration interval
 *                        [x_0, x_{end}]. It is usually zero.
 * @param[in]       Real  xend           End of the integration interval
 *                        [x_0, x_{end}].
 * @param[in,out]   void* devInitCond    Device pointer to a serialized 2D
 *                        Real matrix where each entry corresponds to a single
 *                        pixel in the camera sensor; i.e., to a single ray. If
 *                        the sensor has R rows and C columns, the vector
 *                        pointed by  devInitCond contains R*C entries, where
 *                        each entry is a 5-tuple filled with the initial
 *                        conditions of the corresponding ray: (r, theta, phi,
 *                        pR, pPhi). At the end of this kernel, the array
 *                        pointed by devInitCond is overwritten with the final
 *                        state of each ray.
 * @param[in]       Real  h              Step size for the Runge-Kutta solver.
 * @param[in]       Real  hmax           Value of the maximum step size allowed
 *                        in the Runge-Kutta solver.
 * @param[in]       void* devData        Device pointer to a serialized 2D
 *                        Real matrix where each entry corresponds to a single
 *                        pixel in the camera sensor; i.e., to a single ray. If
 *                        the sensor has R rows and C columns, the vector
 *                        pointed by devData contains R*C entries, where each
 *                        entry is a 2-tuple filled with the constants of the
 *                        corresponding ray: (b, q).
 * @param[out]      void* devStatus      Device pointer to a serialized 2D
 *                        Int matrix where each entry corresponds to a single
 *                        pixel in the camera sensor; i.e., to a single ray. If
 *                        the sensor has R rows and C columns, the vector
 *                        pointed by devData contains R*C entries, where each
 *                        entry is an integer that will store the ray's status
 *                        at the end of the kernel
 * @param[in]       Real  resolutionOrig Amount of time in which the ray will
 *                        be integrated without checking collisions. The lower
 *                        this number is (in absolute value, as it should
 *                        always be negative), the more resolution you'll get
 *                        in the disk edges.
 */
__global__ void kernel(Real x0, Real xend, void* devInitCond, Real h,
                       Real hmax, void* devData, void* devStatus,
                       Real resolutionOrig){
    // Compute pixel's row and col of this thread
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Only the threads that have a proper pixel shall compute its ray equation
    if(row < IMG_ROWS && col < IMG_COLS){
        // Compute pixel unique identifier for this thread
        int pixel = row*IMG_COLS + col;

        // Array of status flags: at the output, the (x,y)-th element will be
        // set to SPHERE, HORIZON or disk, showing the final state of the ray.
        int* globalStatus = (int*) devStatus;
        globalStatus += pixel;
        int status = *globalStatus;

        // Integrate the ray only if it's still in the sphere. If it has
        // collided either with the disk or within the horizon, it is not
        // necessary to integrate it anymore.
        if(status == SPHERE){
            // Retrieve the position where the initial conditions this block
            // will work with are.
            // Each block, absolutely identified in the grid by blockId, works
            // with only one initial condition (that has N elements, as N
            // equations are in the system). Then, the position of where these
            // initial conditions are stored in the serialized vector can be
            // computed as blockId * N.
            Real* globalInitCond = (Real*) devInitCond;
            globalInitCond += pixel * SYSTEM_SIZE;

            // Pointer to the additional data array used by computeComponent
            Real* globalData = (Real*) devData;
            globalData += pixel * DATA_SIZE;

            // Local arrays to store the initial conditions and the additional
            // data
            Real initCond[SYSTEM_SIZE], data[DATA_SIZE];

            // Retrieve the data from global to local memory :)
            memcpy(initCond, globalInitCond, sizeof(Real)*SYSTEM_SIZE);
            memcpy(data, globalData, sizeof(Real)*DATA_SIZE);

            // Current time
            Real x = x0;

            // Local variable to know how many iterations spent the solver in
            // the current step.
            int iterations = 0;

            // MAIN ROUTINE. Integrate the ray from x to xend, checking disk
            // collisions on the go with the following algorithm:
            //   -> 0. Check that the ray has not collided with the disk or
            //   with the horizon and that the current time has not exceeded
            //   the final time.
            //   -> 1. Advance the ray a step, calling the main RK45 solver.
            //   -> 2. Test whether the ray has collided with the horizon.
            //          2.1 If the answer to the 2. test is negative: test
            //          whether the current theta has crossed theta = pi/2,
            //          and call bisect in case it did, updating its status
            //          accordingly (set it to DISK if the ray collided with
            //          the horizon).
            //          2.2. If the answer to the 2. test is positive: update
            //          the status of the ray to HORIZON.
            status = SolverRK45(&x, xend, initCond, h, xend - x, data,
                                &iterations);

            // Update the global status variable with the new computed status
            *globalStatus = status;

            // And, finally, update the current ray state in global memory :)
            memcpy(globalInitCond, initCond, sizeof(Real)*SYSTEM_SIZE);
        } // If status == SPHERE

    } // If row < IMG_ROWS and col < IMG_COLS
}
