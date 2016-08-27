#ifndef __DEFINITIONS__
#define __DEFINITIONS__

// Debug switch


// Declaration of the system size; i.e., the number of equations
#define SYSTEM_SIZE 5
#define DATA_SIZE 2

// Declaration of the image parameters: number of rows and columns, as well as
// the total amount of pixels.
#define IMG_ROWS 1000
#define IMG_COLS 1000
#define NUM_PIXELS 1000000

// Useful constant for collision detection
#define HALF_PI 1.57079632679489655799898173427209258079528808593750


// Bisect's constants
#define BISECT_TOL 0.000001
#define BISECT_MAX_ITER 100

// Butcher's tableau coefficients
#define A21 (1./5.)

#define A31 (3./40.)
#define A32 (9./40.)

#define A41 (44./45.)
#define A42 (- 56./15.)
#define A43 (32./9.)

#define A51 (19372./6561.)
#define A52 (- 25360./2187.)
#define A53 (64448./6561.)
#define A54 (- 212./729.)

#define A61 (9017./3168.)
#define A62 (- 355./33.)
#define A63 (46732./5247.)
#define A64 (49./176.)
#define A65 (- 5103./18656.)

#define A71 (35./384.)
#define A72 (0)
#define A73 (500./1113.)
#define A74 (125./192.)
#define A75 (- 2187./6784.)
#define A76 (11./84.)

#define C2 (1./5.)
#define C3 (3./10.)
#define C4 (4./5.)
#define C5 (8./9.)
#define C6 (1)
#define C7 (1)

#define E1 (71./57600.)
#define E2 (0)
#define E3 (- 71./16695.)
#define E4 (71./1920.)
#define E5 (- 17253./339200.)
#define E6 (22./525.)
#define E7 (- 1./40.)

// Black hole's spin and its square
#define __a  1e-08
#define __a2 __a * __a

// Camera constants
#define __d 3
#define __camR 30
#define __camTheta 1.511
#define __camPhi 0
#define __camBeta 0

// Black hole constants
#define __b1 -5.19615229843
#define __b2 5.19615231843

// Kerr constants
#define __ro 30.0
#define __delta 840.0
#define __pomega 29.9463819688
#define __alpha 0.966091783079
#define __omega 7.40740740741e-13

// RK45 parameters
#define rtoli 1e-06
#define atoli 1e-12
#define safe 0.9
#define safeInv 1.1111111111111112
#define fac1 0.2
#define fac1_inverse 5.0
#define fac2 10.0
#define fac2_inverse 0.1
#define beta 0.04
#define uround 2.3e-16

// Convention for ray's status
#define HORIZON 2
#define DISK 1
#define SPHERE 0
#define STRAIGHT 
#define STRAIGHT_TOL 0.001

#define horizonRadius 2.0
#define innerDiskRadius 9
#define outerDiskRadius 20

// Definition of the data type
typedef double Real;

typedef enum solverStatus{
    RK45_SUCCESS,
    RK45_FAILURE,
    RK45_STOP
} SolverStatus;

#endif // __DEFINITIONS__