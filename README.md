##  Scalable Hyperparameter Optimization with Lazy Gaussian Processes ##

Original Paper: https://ieeexplore.ieee.org/abstract/document/8950672

This work optimizes a black-box function using 2 variants of Bayesian optimization:

    1) Classical Gaussian process

    2) A lazy Gaussian process with a positive integer parameter lag,
       When lag==1: the lazy Gaussian process is equivalent to the classical Gaussian process.


## How to run the script

The first step would be to run the script 'optimize_function.py' as 

python < pythonscript > --seed= < numseeds > --iter= < numiters >

e.g.: python optimize_function.py --seed=1 --iter=200

The function to be optimized should be defined in 'black_box_function' with
the bounded space of arguments called 'pbounds'. 

For illustration, we have defined a one dimensional negative levy function below:

def black_box_function(x):

    w_x = 1 + (x - 1)/4.0
    w_b = 1 + (x - 1)/4.0
    
    term1 = (math.sin(math.pi*w_x))**2
    
    term3 = (w_b-1)**2 * (1+(math.sin(2*math.pi*w_b))**2)
    
    y = term1 + term3
    
    return -y

and pbounds: { 'x': (-10, 10)}


A custom black_box_function with range of arguments (pbounds) can also be defined and run using script 'optimize_function.py'. 

A boolean flag 'lazy_gpr' is used to distinguish between classical and lazy Gaussian processes. When flag lazy_gpr is False, it denotes bayesian optimization with the classical Gaussian process.
and lazy_gpr is True, denotes bayesian optimization with the lazy Gaussian process (with parameter 'lag' that can be set). 

##    Optimization of black-box functions
In the 'experiments' folder, we have optimized the test function: 5-dimensional levy function.

how to run:

cd experiments;

python run_levy5D.py --seed= < numseeds > --iter= < numiters>, e.g. python run_levy5D.py --seed=1 --iter=200

## Hyperparameter optimization of neural networks 

In the 'experiments' folder, we have optimized the hyperparameters of neural networks. The black box function is the doing the training of
the neural networks. The arguments of the black-box function are treated as hyperparameters.

We have optimized the hyperparameters (learning_rate, momentum, weight_decay, keep_prob1, keep_prob2) for LeNet5 with MNIST dataset.

how to run:

cd experiments;

python run_mnist.py --seed= < numseeds > --iter= < numiters >, e.g. python run_mnist.py --seed=1 --iter=1000

We have optimized the hyperparameters (learning_rate, momentum, weight_decay) for ResNet32 with CIFAR10 datset.

how to run:

cd experiments;

python run_cifar10.py --seed= < numseeds > --iter = < numiters >, e.g. python run_cifar10.py --seed=1 --iter=200

The desired accuracy in the neural networks can be modeled by setting the parameter 'eps' in the script, 
e.g. inside run_cifar10.py:  eps=0.21 means reaching an accuracy of  0.79 (1 - 0.21). 
Once this accuracy is reached, the iteration loop will break.
