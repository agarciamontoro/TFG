#ifndef __DEFINITIONS__
#define __DEFINITIONS__

// Debug switch


// Declaration of the system size; i.e., the number of equations
#define SYSTEM_SIZE 5

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
#define fac1 0.2
#define fac2 10.0
#define beta 0.04
#define uround 2.3e-16

// Convention for ray's status
#define HORIZON 2
#define DISK 1
#define SPHERE 0

#define horizonRadius 2.0
#define innerDiskRadius 9
#define outerDiskRadius 20

// Definition of the data type
typedef double Real;

#endif // __DEFINITIONS__