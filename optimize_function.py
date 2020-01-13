""" This script optimizes a black box funtion using 2 variants of bayesian optimization:
    1) Classical gaussian process
    2) Lazy gaussian process with a positive integer parameter lag,
       When lag==1: lazy gaussian process becomes classical gaussian process.

    Authors: Raju Ram <raju.ram@itwm.fraunhofer.de>
             Sabine Mueller <sabine.b.mueller@itwm.fraunhofer.de>
"""

import sys
sys.path.append("bayes_opt")

# Bayesian optimization using classical and lazy gaussian process
from lazy_gaussian_process.bayesian_optimization import BayesianOptimization

import numpy as np
import pickle
import math
import time
import argparse


#Define the black box funtion which is to be optimized
#def black_box_function(args..):
def black_box_function(x):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """

    #Define the function to be optimized below ..
    #For illustration, we have defined 1d levy function below:
    w_x = 1 + (x - 1)/4.0
    w_b = 1 + (x - 1)/4.0

    term1 = (math.sin(math.pi*w_x))**2
    term3 = (w_b-1)**2 * (1+(math.sin(2*math.pi*w_b))**2)
    y = term1 + term3

 #since we want to minimize the levy function, using minus(- sign means maximize the negative of the levy funtion
    return -y


# Define the bounded region of parameter space
#For illustration, we have defined bounded region of 1d levy function below:
pbounds = { 'x': (-10, 10)}


parser = argparse.ArgumentParser(description='black_box_function')
parser.add_argument('--seed', type=int, default=100, help='Number of seed points')
parser.add_argument('--iter', type=int, default=500, help='Number of iterations')

args = parser.parse_args()

#lazy_gpr is set to False when using classical bayesian optimization
#lazy_gpr is set to True when using bayesian optimization with lazy gaussian processes
lazy_gpr = False

for i in range(2):

    start=time.time()

    if(lazy_gpr):
        opt = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
            lazy_gpr=True,
            lag = 10000
        )
    else:
        opt = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
            lazy_gpr=False
        )

    opt.maximize(
        init_points=args.seed,
        n_iter=args.iter,
        samples=None,
        eps=0.0000001,
        solution=0,
        acq="ei", xi=0.01
    )
    end = time.time()
    passed = end-start
    print('target: ', opt.max['target'], ',  lazy_gpr: ', lazy_gpr)
    print('Time(sec.) : ' + str(passed) + "\n")
    lazy_gpr=True