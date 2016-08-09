// Cosas:
// nvar es el numbero de ecuaciones del sistema
// x0 es el valor del punto donde se esta calculando
// ( El tiempo en el que se esta haciendo el paso, vaya)
// dx es el paso
// y0 es un puntero a las condiciones iniciales
// tiene que tener dimension nvar obviamente
// Y finalmente
// rhs
// es un puntero a una fiuncion
// rhs( t0, condiciones iniciales, vector de ks) que evalua las ecuaciones


/* ********************************************************************* */
int RK4Solve (double *y0, int nvar, double x0,
              double dx, void (*rhs)(double, double *, double *))
/*
 *
 *
 *********************************************************************** */
{
  int    nv;
  double y1[ODE_NVAR_MAX];
  double k1[ODE_NVAR_MAX], k2[ODE_NVAR_MAX];
  double k3[ODE_NVAR_MAX], k4[ODE_NVAR_MAX];

/* -- step 1 -- */

  rhs (x0, y0, k1);
  for (nv = 0; nv < nvar; nv++) y1[nv] = y0[nv] + 0.5*dx*k1[nv];

/* -- step 2 -- */

  rhs (x0 + 0.5*dx, y1, k2);
  for (nv = 0; nv < nvar; nv++) y1[nv] = y0[nv] + 0.5*dx*k2[nv];

/* -- step 3 -- */

  rhs (x0 + 0.5*dx, y1, k3);
  for (nv = 0; nv < nvar; nv++) y1[nv] = y0[nv] + dx*k3[nv];

/* -- step 4 -- */

  rhs (x0 + dx, y1, k4);
  for (nv = 0; nv < nvar; nv++)
    y0[nv] += dx*(k1[nv] + 2.0*(k2[nv] + k3[nv]) + k4[nv])/6.0;

  return (0);
}
