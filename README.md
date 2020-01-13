##  Scalable Hyperparameter Optimization with Lazy Gaussian Processes ##

This work optimizes a black box funtion using 2 variants of bayesian optimization:
    1) Classical gaussian process
    2) Lazy gaussian process with a positive integer parameter lag,
       When lag==1: lazy gaussian process is equivalent to the classical gaussian process.



## How to run the script

The first step would be to run the .py script called 'optimize_function.py'
python <pythonscript> --seed=<numseeds> --iter=<numiters>
ex: python optimize_function.py --seed=1 --iter=200


The function to be optimized should be defined in 'black_box_function' with
the bounded space of agruments 'pbounds'

For illustration, we have defined 1d levy function below:

w_x = 1 + (x - 1)/4.0
w_b = 1 + (x - 1)/4.0

term1 = (math.sin(math.pi*w_x))**2
term3 = (w_b-1)**2 * (1+(math.sin(2*math.pi*w_b))**2)
y = term1 + term3

return -y

and pbounds = { 'x': (-10, 10)}

A custom black box function with agruments can be defined and run with 'optimize_function.py'

The boolean flag lazy_gpr = False,  denotes bayesian optimization with classical gaussian process.
and lazy_gpr = True, denotes bayesian optimization with lazy gaussian process (with parameter lag that can be set)

##Optimization of black box functions
In the 'experiments' folder, we have optimized the test functions such as 5 dimensional levy function.

how to run:
cd experiments;
python run_levy5D.py --seed=<numseeds> --iter=<numiters>, e.g. python run_levy5D.py --seed=1 --iter=200

-----------------------Hyperparemeter optimization of neural networks ---------------------------

Also, we have optimized the hyperparameters of neural networks. The black box function is the training of
the neural networks. The agruments of the black box function are treated as hyperparameters.

We have optimized the hyperparameters (learning_rate, momentum, weight_decay, keep_prob1, keep_prob2) for LeNet5 with MNIST datset.

how to run:
cd experiments;
python run_mnist.py --seed=<numseeds> --iter=<numiters>, e.g. python run_mnist.py --seed=1 --iter=1000

We have optimized the hyperparameters (learning_rate, momentum, weight_deca)) for ResNet32 with CIFAR10 datset.
how to run:
cd experiments;
python run_cifar10.py --seed=<numseeds> --iter=<numiters>, e.g. python run_cifar10.py --seed=1 --iter=200

The desired accuracy can be modeled by setting the parameter eps in the script.
For e.g. inside run_cifar10.py:  eps=0.21 means reaching an accuracy of 0.79. One this desired accuracy is reached the
iteration loop will break

------------------------------------------------------------------------------------------------------------
## Results






