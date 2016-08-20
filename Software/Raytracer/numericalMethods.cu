#include <math.h>
#include <stdio.h>
#include "definitions.c"

#define CON 1.4
#define CON2 (CON*CON)
#define INF 1.0e30
#define NTAB 10
#define SAFE 2.0

typedef double Real;

// Kindly borrowed from Rosetta code :)
__device__ Real secant( Real xA, Real xB, Real(*f)(Real, Parameters),
                        Parameters param)
{
    Real e = 1.0e-10;
    Real fA, fB;
    Real d;
    int i;
    int limit = 10000;

    fA=(*f)(xA, param);

    for (i=0; i<limit; i++) {
        fB=(*f)(xB, param);
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

// Kindly borrowed from Numerical Recipes in C :)
__device__ Real dfridr(Real (*func)(Real, Parameters), Real x, Parameters param, Real h, Real *err){
	int i, j;
	Real errt, fac, hh, ans;

	if (h == 0.0){
		printf("h must be nonzero in dfridr.\n");
		return -INF;
	}

	Real a[NTAB][NTAB];
	hh = h;

	a[1][1] = ((*func)(x+hh, param) - (*func)(x-hh, param)) / (2.0*hh);
	*err = INF;

	for(i = 2; i <= NTAB; i++){
		hh /= CON;
		a[1][i] = ((*func)(x+hh, param) - (*func)(x-hh, param)) / (2.0*hh);
		fac = CON2;

		for(j = 2; j <= i; j++){
			a[j][i] = (a[j-1][i]*fac-a[j-1][i-1]) / (fac-1.0);
			fac *= CON2;
			errt = fmax(abs(a[j][i]-a[j-1][i]), abs(a[j][i]-a[j-1][i-1]));

			if(errt <= *err){
				*err = errt;
				ans = a[j][i];
			}
		}

		if(fabs(a[i][i]-a[i-1][i-1]) >= SAFE*(*err)){
			return ans;
		}
	}

	return ans;
}
