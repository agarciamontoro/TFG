import os
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit


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
        x0: A real with the value of the time the system is being solved.
        y0: A numpy array storing the state of the system.
    """

    def __init__(self, x0, y0, dx, systemFunctions):
        """Builds the RungeKutta4 solver.

        Args:
            x0: A real with the value of the time the system will be solved.
            y0: A :math:`(w, h, n)` shaped numpy array of initial conditions,
                where :math:`w` and :math:`h` are the widht and height of the
                matrix and :math:`n` the number of initial conditions, that
                shall be the same as the number of equations in the system. The
                solver forces the types of x0 and dx to be the same as the type
                of y0.
            dx: A real containing the step size for the evolution of the
                system.
            systemFunctions: A list of :math:`n` strings containing the
                functions of the system, written in C.

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
                already defined in the code for you to freely use them.

                - :code:`Real x`: A real containing the value of the
                  independent variable :math:`x`.
                - :code:`Real y[n]`: An array of :code:`n` elements containing
                  the value of the initial conditions :math:`Y(x) = (y_0(x),
                  \\dots, y_{n-1}(x))`.
        """

        # ======================= INITIAL CONDITIONS ======================= #

        # Get precision: single or double
        self.type = y0.dtype
        assert(self.type == np.float32 or self.type == np.float64)

        # Convert x0 to the same type of y0
        self.x0 = np.array(x0).astype(self.type)
        self.y0 = y0

        # ============================ CONSTANTS ============================ #

        # Shape of the initial conditions matrix: width and height. This shape
        # will define the dimensions of the CUDA grid
        self.INIT_H = y0.shape[0]
        self.INIT_W = y0.shape[1]

        # Number of equations on the system
        self.SYSTEM_SIZE = y0.shape[2]

        # Convert dx to the same type of y0
        self.STEP_SIZE = np.array(dx).astype(self.type)

        # System function
        self.F = [(str(i), f) for i, f in enumerate(systemFunctions)]


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
            "SYSTEM_FUNCTIONS": self.F,
            "Real": codeType
        }

        # Finally, process the template to produce our final text.
        kernel = template.render(templateVars)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        mod = compiler.SourceModule(kernel)

        # Get the kernel function from the compiled module
        self.RK4Solve = mod.get_function("RK4Solve")

        # ========================== DATA TRANSFER ========================== #

        # Transfer host (CPU) memory to device (GPU) memory
        self.y0GPU = gpuarray.to_gpu(self.y0)

        # Create two timers to measure the time
        self.start = driver.Event()
        self.end = driver.Event()

        self.totalTime = 0.

    def solve(self):
        """Computes one step of the system evolution.

        The system is evolved a single step, using the initial conditions x0
        and y0 provided in the constructor. These variables are then updated to
        their new computed values.
        """

        self.start.record()  # start timing

        # Call the kernel on the card
        self.RK4Solve(
            # Inputs
            self.x0,
            self.y0GPU,
            self.STEP_SIZE,

            # Grid definition -> number of blocks x number of blocks.
            # Each block computes one RK4 step for a single initial condition
            grid=(self.INIT_W, self.INIT_H, 1),
            # block definition -> number of threads x number of threads
            # Each thread in the block computes one RK4 step for one equation
            block=(self.SYSTEM_SIZE, 1, 1),
        )

        self.end.record()    # end timing

        # calculate the run length
        self.end.synchronize()
        self.totalTime = self.totalTime + self.start.time_till(self.end)*1e-3

        # Update the time in which the system solution is computed
        self.x0 = self.x0 + self.STEP_SIZE

        # Return the new data
        y1 = self.y0GPU.get()
        return(y1)
