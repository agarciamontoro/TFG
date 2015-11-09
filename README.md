# TFG
General Relativity massively parallel raytracer - a free software alternative.

## Learning CUDA
### Fractal generation
The file `Samples/julia_set.cu` implements the generation of the fractal associated to the [Julia set](https://en.wikipedia.org/wiki/Julia_set).

For every pixel in the final image, the code launches a different thread that computes if the associated point in the complex plane belongs to the Julia set. The number of necessary iterations to decide whether it belongs to the set determine the intensity of the colour in the final image.

An example of the results that can be obtained with this code can be seen in the two following figures, which show the set for two different constants:

![Julia set fractal](https://cloud.githubusercontent.com/assets/3924815/11050110/6e21aad2-8743-11e5-8414-6eb5bd86e881.png)

![Julia set fractal 2](https://cloud.githubusercontent.com/assets/3924815/11050389/bacaeeb4-8745-11e5-8fa5-f45278f62731.png)
