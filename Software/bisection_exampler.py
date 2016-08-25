# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import numpy





mod = SourceModule("""

     #include <stdio.h>
     #include <math.h>


    __device__ void computeComponent(int threadId, double x, double* y, double* f,
                                  double* data){

        f[0] = y[1];
        f[1] = -y[0];

    }



    __global__ void bisect(double* p1, double step, double target, int n)
    {

        double* y_current = p1;
        double y_center[2];
        int i;

        memcpy( y_center, y_current, n*sizeof(double));


        while (abs(y_center[0]-target) > 0.001){

            step = step / 2;
            computeComponent( 0, 0, y_current, y_center, 0);

            for(i=0;i<n;i++){
                y_center[i] = y_current[i] + y_center[i]*step;
            }

            if( y_center[0] > target ){

                memcpy( y_current, y_center, n*sizeof(double));
            }

        }

        p1 = y_center;
    }

    """)

a = numpy.array([0.212212,-2.99248])
a = a.astype(numpy.float64)
a_gpu = gpuarray.to_gpu(a)
func = mod.get_function("bisect")
func( a_gpu,np.float64(0.1),np.float64(0.0),np.int32(2),block=(1,1,1),grid=(1,1,1))
result = a_gpu.get()
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
print(result)
