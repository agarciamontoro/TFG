#ifndef __DEFINITIONS__
#define __DEFINITIONS__



#define SYSTEM_SIZE 5
#define DATA_SIZE 

#define __a  0.0001
#define __a2 __a * __a

typedef double Real;

typedef struct foo_param {
   Real r;
   Real theta;
   Real phi;
   Real b;
   Real q;
   Real pR;
   Real pTheta;
} Parameters;

typedef enum origin{
    HORIZON,
    CELESTIAL_SPHERE
} OriginType;

#endif // __DEFINITIONS__