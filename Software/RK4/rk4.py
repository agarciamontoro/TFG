import os
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit


# Kindly borrowed from http://stackoverflow.com/a/14267825
def nextPowerOf2(x):
    return 1 << (x-1).bit_length()


class RK4Solver:
    """4th order Runge Kutta solver.

    This class numerically solves systems of the form :math:`y'(x) = F(x, y,
    y')``, where :math:`x \in \mathbb{R}`, :math:`y(x) = (y_0(x), y_1(x),
    \dots, y_{n-1}(x))\in\mathbb{R}^n` and :math:`F(x, y)` is a function in
    :math:`\mathbb{R}^n`.

    The solver works with an arbitrary shaped matrix of initial conditions,
    each of one with a lenght of :math:`n` elements. The computation is
    parallelized against these initial conditions and within the algorithm
    itself, using CUDA.

    Attributes:
        x0 (float): Current value of the independent variable of the system
            status.
        y0 (NumPy array): A :math:`(w, h, n)` shaped numpy array containing
            :math:`y_0`, the state of the system at :math:`x_0`,
            where :math:`w` and :math:`h` are the widht and height of the
            matrix and :math:`n` the number of initial conditions.
    """

    # TODO: Make tolerances a SYSTEM_SIZE-length array
    def __init__(self, x0, y0, dx, functionFile, relativeTol=1e-6, absoluteTol=1e-12,
                 safe=0.9, fac1=0.2, fac2=10.0, beta=0.04, uround=2.3e-16,
                 debug=False):
        """Builds the RungeKutta4 solver.

        Args:
            x0 (float): Independent variable value in the initial conditions.
            y0 (NumPy array): A :math:`(w, h, n)` shaped numpy array of initial
                conditions, where :math:`w` and :math:`h` are the widht and
                height of the matrix and :math:`n` the number of initial
                conditions, that shall be the same as the number of equations
                in the system. The solver forces the types of x0 and dx to be
                the same as the type of y0.
            dx (float): Inital step size provided to the automatic step size
                detector as a first try.
            systemFunctions: A list of :math:`n` strings containing the
                functions of the system, written in C.
            relativeTol (float): Relative tolerance for the local error
                estimation. Defaults to 1e-6.
            absoluteTol (float): Absolute tolerance for the local error
                estimation. Defaults to 1e-12.
            safe (float): Safe factor for the computation of the step size.
                Deafaults to 0.9.
            fac1 (float): Minimum factor difference between two successive step
                sizes.
            fac2 (float): Maximum factor difference between two successive step
                sizes.
            beta (float): Stabilized step size control factor. Defaults to
                0.04.
            uround (float): Rounding unit. Defaults to 2.3e-16.
            debug (bool): Switch to print out debug messages while running the
                CUDA kernel.


        Example:
            Let :math:`y''(x) = -25y(x)` be the equation to solve. If we make
            the usual variable change to get a first order system of ODEs;
            i.e., we call :math:`Y = \\begin{pmatrix} y_0 \\\\ y_1
            \\end{pmatrix}`, where :math:`y_0 = y` and :math:`y_1 = y'` then
            the system to solve is

            .. math::
                \\begin{eqnarray}
                    y_0'(x) =& y_1(x) \\\\
                    y_1'(x) =& -25y_0(x)
                \\end{eqnarray}

            The system function is then

            .. math::
                F(x, Y) = \\begin{cases}
                    y_1 \\\\
                    -25y_0
                \\end{cases}

            If we want to solve this system for two different initial
            conditions :math:`Y(x_0) = (1, 1)` and :math:`Y(x_0) = (2, 2)`,
            both of them with starting point :math:`x_0 = -1`, and a step size
            of :math:`dx=0.02` we can call the solver as follows:

            >>> x0 = -1
            >>> initCond = np.array([[[1, 1]], [[2, 2]]])
            >>> dx = 0.02
            >>> functions = ["y[1]",
            ...              "-25*y[0]"]
            >>> solver = RK4Solver(x0, initCond, dx, functions)

            .. note::
                Please note that the functions provided to the solver has to be
                written in plain C. You can use the following variables,
                already defined in the code for you:.

                - :code:`Real x`: A real containing the value of the
                  independent variable :math:`x`.
                - :code:`Real y[n]`: An array of :code:`n` elements containing
                  the value of the initial conditions :math:`Y(x) = (y_0(x),
                  \\dots, y_{n-1}(x))`.
        """

        # ======================= INITIAL CONDITIONS ======================= #

        self.setInitialConditions(x0, y0)

        # ============================ CONSTANTS ============================ #

        # Shape of the initial conditions matrix: width and height. This shape
        # will define the dimensions of the CUDA grid
        self.INIT_H = y0.shape[0]
        self.INIT_W = y0.shape[1]

        # Number of equations on the system
        self.SYSTEM_SIZE = y0.shape[2]

        # Number of threads in the block
        self.THREADS_NUM = nextPowerOf2(self.SYSTEM_SIZE)

        # Convert dx to the same type of y0
        self.step = np.array(dx).astype(self.type)

        # Convert tolerances to arrays and copy them to GPU
        self.relativeTol = np.repeat(relativeTol,
                                     self.SYSTEM_SIZE).astype(self.type)
        self.relativeTolGPU = gpuarray.to_gpu(self.relativeTol)

        self.absoluteTol = np.repeat(absoluteTol,
                                     self.SYSTEM_SIZE).astype(self.type)
        self.absoluteTolGPU = gpuarray.to_gpu(self.absoluteTol)

        # Algorithm parameters
        self.safe = np.array(safe).astype(self.type)
        self.fac1 = np.array(fac1).astype(self.type)
        self.fac2 = np.array(fac2).astype(self.type)
        self.beta = np.array(beta).astype(self.type)
        self.uround = np.array(uround).astype(self.type)

        self.functionFile = functionFile

        # Debug switch
        self.debug = debug


        # ==================== KERNEL TEMPLATE RENDERING ==================== #

        # We must construct a FileSystemLoader object to load templates off
        # the filesystem
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        templateLoader = jinja2.FileSystemLoader(searchpath=currentDirectory)

        # An environment provides the data necessary to read and
        # parse our templates.  We pass in the loader object here.
        templateEnv = jinja2.Environment(loader=templateLoader)

        # Read the template file using the environment object.
        # This also constructs our Template object.
        template = templateEnv.get_template("rk4_kernel.cu")

        codeType = "float" if self.type == np.float32 else "double"

        # Specify any input variables to the template as a dictionary.
        templateVars = {
            "SYSTEM_SIZE": self.SYSTEM_SIZE,
            "INCLUDES": '#include "%s"' % self.functionFile,
            # "Real": "typedef %s Real;" % codeType,
            "DEBUG": "#define DEBUG" if self.debug else ""
        }

        # Finally, process the template to produce our final text.
        kernel = template.render(templateVars)

        if(self.debug):
            kernelTmpFile = open("lastKernel.cu", "w")
            kernelTmpFile.write(kernel)
            kernelTmpFile.close()

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        mod = compiler.SourceModule(kernel)

        # Get the kernel function from the compiled module
        self.RK4Solve = mod.get_function("RK4Solve")

        # ========================== DATA TRANSFER ========================== #

        # Create two timers to measure the time
        self.start = driver.Event()
        self.end = driver.Event()

        self.totalTime = 0.

    def setInitialConditions(self, x0, y0):
        # Get precision: single or double
        self.type = np.float64
        assert(self.type == np.float32 or self.type == np.float64)

        # Convert x0 to the same type of y0
        self.x0 = np.array(x0).astype(self.type)

        # Transfer host (CPU) memory to device (GPU) memory
        # FIXME: Does this free the previous memory or no?
        self.y0GPU = y0

    def solve(self, xEnd):
        """Evolve the system between x0 and xEnd.

        Iteratively calls the DOPRI5 solver to evolve the system between x0 and
        xEnd, automatically adapting the step size and minimizing the local
        errors, that should be roughly below relativeTol*abs(y[i])+absoluteTol.

        Args:
            xEnd (float): End of the interval where the system will be evolved;
                i.e., the solver will take as initial conditions :math:`(x_0,
                y_0)` and will compute the value at :math:`(x_{end}, y_{end})`.

        Returns:
            NumPy array: A :math:`(w, h, n)` shaped numpy array containing
                :math:`y_{end}`, the state of the system at :math:`x_{end}`,
                where :math:`w` and :math:`h` are the widht and height of the
                matrix and :math:`n` the number of initial conditions.

        """

        self.start.record()  # start timing

        # Call the kernel on the card
        self.RK4Solve(
            # Inputs
            self.x0,
            np.array(xEnd).astype(self.type),
            self.y0GPU,
            self.step,
            np.array(xEnd-self.x0).astype(self.type),
            self.relativeTolGPU,
            self.absoluteTolGPU,
            self.safe,
            self.fac1,
            self.fac2,
            self.beta,
            self.uround,

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes one RK4 step for a single initial condition
            grid=(self.INIT_W, self.INIT_H, 1),
            # block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one equation
            block=(self.THREADS_NUM, 1, 1),
        )

        self.end.record()   # end timing
        self.end.synchronize()

        # Calculate the run length
        self.totalTime = self.totalTime + self.start.time_till(self.end)*1e-3

        # Update the new state of the system
        self.x0 = xEnd
        self.y0 = self.y0GPU.get()

        return(self.y0)
