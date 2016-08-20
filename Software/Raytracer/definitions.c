#ifndef __DEFINITIONS__
#define __DEFINITIONS__

typedef double Real;

extern __device__ Real __a;
extern __device__ Real __a2;

typedef struct foo_param {
   Real r;
   Real theta;
   Real phi;
   Real b;
   Real q;
   Real pR;
   Real pTheta;
} Parameters;

#endif
