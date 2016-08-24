# Sample source code from the Tutorial Introduction in the documentation.

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import numpy
a = numpy.random.randn(4,4)
a = a.astype(numpy.float32)
a_gpu = gpuarray.to_gpu(a)


mod = SourceModule("""
    __global__ void doublify(int n,float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;

      float cosa = a[idx];
      for(int i=0;i<n;i++){
      cosa *= 1.001;
      }
      a[idx] = cosa;
    }
    """)

func = mod.get_function("doublify")

for _ in range(40000):
    func( numpy.int32(1),a_gpu, block=(4,4,1))
a_doubled = a_gpu.get()

print("original array:")
print(a)
print("doubled with kernel:")
print(a_doubled)


