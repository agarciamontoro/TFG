#ifndef __DEFINITIONS__
#define __DEFINITIONS__

// Debug switch


// Declaration of the system size; i.e., the number of equations
#define SYSTEM_SIZE 5

// Definition of the black hole's spin and its square
#define __a  0.999
#define __a2 __a * __a

// Definition of the data type
typedef double Real;

// Struct to store the system parameters
typedef struct foo_param {
   Real r;
   Real theta;
   Real phi;
   Real pR;
   Real pTheta;
   Real b;
   Real q;
} Parameters;

// Enum to differentiate between points of origin
typedef enum origin{
    HORIZON,
    CELESTIAL_SPHERE
} OriginType;

#endif // __DEFINITIONS__