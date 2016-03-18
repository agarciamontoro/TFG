# TFG
General Relativity massively parallel raytracer - a free software alternative.

## Learning CUDA
### N-body simulation
The files [`Samples/nbody/nbody.py`](https://github.com/agarciamontoro/TFG/blob/master/Samples/nbody/nbody.py) and [`Samples/nbody/kernel.cu`](https://github.com/agarciamontoro/TFG/blob/master/Samples/nbody/kernel.cu) implement a simulation of an event that will take place in about 4 billion years: the Andromeda–Milky Way collision.

The code uses the power of the massively parallel structure of a GPU to compute a simulation on what is generally called an N-body problem.

The program assigns each body in the simulation (49152) to a thread in the GPU. Then, each one of these threads computes the force produced by all the remaining bodies onto itself -using the Newtonian laws-, updating its position and velocity when the acceleration is obtained.

The simulated collision can be sen in the following animation, where the blue bodies correspond to Andromeda and the green ones to our galaxy, the Milky Way.

<p align="center">
<img src="https://cloud.githubusercontent.com/assets/3924815/13893566/c74974c8-ed5f-11e5-9536-1533721596aa.gif" />
</p>

### Fractal generation
The file [`Samples/julia_set/julia_set.py`](https://github.com/agarciamontoro/TFG/blob/master/Samples/julia_set/julia_set.py) implements the generation of the fractal associated to the [Julia set](https://en.wikipedia.org/wiki/Julia_set).

For every pixel in the final image, the code launches a different thread that computes if the associated point in the complex plane belongs to the Julia set. The number of necessary iterations to decide whether it belongs to the set determine the intensity of the colour in the final image.

An example of the results that can be obtained with this code can be seen in the two following figures, which show the set for two different constants:

<p align="center">
<img src="https://cloud.githubusercontent.com/assets/3924815/11050110/6e21aad2-8743-11e5-8414-6eb5bd86e881.png" width="49%" alt="Julia set fractal"/> <img src="https://cloud.githubusercontent.com/assets/3924815/11050389/bacaeeb4-8745-11e5-8fa5-f45278f62731.png" width="49%" alt="Julia set fractal"/>
</p>
