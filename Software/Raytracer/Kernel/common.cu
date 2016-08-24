#ifndef __DEFINITIONS__
#define __DEFINITIONS__

// Debug switch


// Declaration of the system size; i.e., the number of equations
#define SYSTEM_SIZE 5

// Black hole's spin and its square
#define __a  0.999
#define __a2 __a * __a

// Camera constants
#define __d 3
#define __camR 74
#define __camTheta 1.511
#define __camPhi 0
#define __camBeta 0

// Black hole constants
#define __b1 -6.99833323454
#define __b2 2.07812987106

// Kerr constants
#define __ro 74.0000240824
#define __delta 5328.998001
#define __pomega 73.8746543387
#define __alpha 0.986393999977
#define __omega 4.92968044321e-06

// RK45 parameters
#define rtoli 1e-06
#define atoli 1e-12
#define safe 0.9
#define fac1 0.2
#define fac2 10.0
#define beta 0.04
#define uround 2.3e-16

// Convention for ray's status
#define HORIZON 2
#define DISK 1
#define SPHERE 0

#define horizonRadius 1.04471017781
#define innerDiskRadius 9
#define outerDiskRadius 20

// Definition of the data type
typedef double Real;

#endif // __DEFINITIONS__