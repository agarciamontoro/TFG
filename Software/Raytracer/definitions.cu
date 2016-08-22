#ifndef __DEFINITIONS__
#define __DEFINITIONS__



#define SYSTEM_SIZE 5
#define DATA_SIZE 

#define __a  1e-05
#define __a2 __a * __a

typedef double Real;

typedef struct foo_param {
   Real r;
   Real theta;
   Real phi;
   Real pR;
   Real pTheta;
   Real b;
   Real q;
} Parameters;

typedef enum origin{
    HORIZON,
    CELESTIAL_SPHERE
} OriginType;

#endif // __DEFINITIONS__