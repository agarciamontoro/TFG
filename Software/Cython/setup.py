from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize('geodesic_integrator.pyx'),
            include_dirs=[numpy.get_include()]
            )


extensions = [
    Extension("geodesic_integrator", ["geodesic_integrator.pyx"],
        include_dirs = [numpy.get_include()]),
    Extension("metric", ["metric.pyx"],
        include_dirs = [numpy.get_include()]),
]


setup(
    name = "blackstar",
    ext_modules = cythonize(extensions),
)
