import sys
sys.path.append("../bayes_opt")
sys.path.append("..")

# Bayesian optimization using classical and lazy gaussian process
from lazy_gaussian_process.bayesian_optimization import BayesianOptimization

import numpy as np
import pickle
import math
import time
import argparse


#5 dimensional levy function, which is used as a blackbox function in the optimization problem
def levy(x,y,z,a,b):

    w_x = 1 + (x - 1)/4.0
    w_y = 1 + (y - 1)/4.0
    w_z = 1 + (z - 1)/4.0
    w_a = 1 + (a - 1)/4.0
    w_b = 1 + (b - 1)/4.0

    term1 = (math.sin(math.pi*w_x))**2
    term3 = (w_b-1)**2 * (1+(math.sin(2*math.pi*w_b))**2)
    sum = 0
    sum += (w_x-1)**2 * (1+10*(math.sin(math.pi*w_x+1))**2)
    sum += (w_y-1)**2 * (1+10*(math.sin(math.pi*w_y+1))**2)
    sum += (w_z-1)**2 * (1+10*(math.sin(math.pi*w_z+1))**2)
    sum += (w_a-1)**2 * (1+10*(math.sin(math.pi*w_a+1))**2)
    y = term1 + sum + term3

    return -y


pbounds = {'x': (-10, 10), 'y': (-10, 10), 'z': (-10, 10), 'a': (-10, 10), 'b': (-10, 10)}

parser = argparse.ArgumentParser(description='Levy5')
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
            f=levy,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
            lazy_gpr=True,
            lag = 10000
        )
    else:
        opt = BayesianOptimization(
            f=levy,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
            lazy_gpr=False
        )

    opt.maximize(
        init_points=args.seed,
        n_iter=args.iter,
        samples=None,
        eps=1,
        solution=0,
        acq="ei", xi=0.01
    )
    end = time.time()
    passed = end-start
    print('target : ', opt.max['target'], ',  lazy_gpr: ', lazy_gpr)
    print('Time(sec.) : ' + str(passed) + "\n")
    lazy_gpr=True