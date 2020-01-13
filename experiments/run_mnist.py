import sys
sys.path.append("../bayes_opt")
sys.path.append("..")

# Bayesian optimization using classical and lazy_gpr gaussian process
from lazy_gaussian_process.bayesian_optimization import BayesianOptimization
import numpy as np

# import network_training
import pickle
import math
import time
import pandas as pd
import lenet5_network as lenet5
import argparse

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

#load the mnist dataset from the .csv files
def mnist_load_data():
    df_train = pd.read_csv('../dataset/train.csv')
    df_test = pd.read_csv('../dataset/test.csv')
    df_train.head()

    df_train = pd.get_dummies(df_train,columns=["label"])
    df_features = df_train.iloc[:, :-10].values
    df_label = df_train.iloc[:, -10:].values


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_label, test_size = 0.2, random_state = 1212)
    X_test, X_validation, y_test,y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=0)



    train_dataset, train_labels = reformat(X_train, y_train)
    valid_dataset, valid_labels = reformat(X_validation, y_validation)
    test_dataset , test_labels = reformat(X_test, y_test)
    df_test = df_test.values.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    # Pad images with 0s
    X_train      = np.pad(train_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_validation = np.pad(valid_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test       = np.pad(test_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    return X_train, X_validation, X_test, y_train, y_validation, y_test

X_train, X_validation, X_test, y_train, y_validation, y_test = mnist_load_data()

#The function which is to be optimized with the given hyperparameters weight_decay, momentum, learning_rate, keep_prob1, keep_prob2
def black_box_function(weight_decay, momentum, learning_rate, keep_prob1, keep_prob2):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """

    print ('Training MNIST dataset on Lenet5 with')
    print('(lr: ', learning_rate, ', momentum: ', momentum, ', wd: ', weight_decay, ', prob1: ', keep_prob1, ', prob2: ', keep_prob2, ')')

    #Training of LeNet5 with given hyperparameter, this function returns accuracy of LeNet5
    return lenet5.run_mnist(learning_rate, momentum, weight_decay, X_train, X_validation, X_test, y_train, y_validation, y_test, keep_prob1, keep_prob2)

# Bounded region of parameter space

# as per in the paper
pbounds = {'weight_decay': (0, 0.001), 'momentum': (0, 0.99), 'learning_rate': (0.0001, 0.1), 'keep_prob1':(0.01,1), 'keep_prob2':(0.01,1)}

# prev
#pbounds = {'weight_decay': (0, 0.01), 'momentum': (0, 0.99), 'learning_rate': (10e-4, 0.25), 'keep_prob1':(0.01,1), 'keep_prob2':(0.01,1)}

# most suited -> 28th Nov 2019
#pbounds = {'weight_decay': (0, 0.01), 'momentum': (0.5, 0.99), 'learning_rate': (10e-4, 0.01), 'keep_prob1':(0.70,1), 'keep_prob2':(0.70,1)}


parser = argparse.ArgumentParser(description='mnist')
parser.add_argument('--seed', type=int, default=1, help='Number of seed points')
parser.add_argument('--iter', type=int, default=100, help='Number of iterations')
parser.add_argument('--log_gpr', default='./logs_mnist_gpr', help='logfile')
parser.add_argument('--restart', type=bool, default=False, help='restart optimization')

args = parser.parse_args()

#lazy_gpr is set to False when using classical bayesian optimization
#lazy_gpr is set to True when using bayesian optimization with lazy gaussian processes
lazy_gpr = False
for i in range(2):

    start=time.time()
    logfile = args.log_gpr + '_lazy_gpr_' + str(lazy_gpr)

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

    if(args.restart):
        print('restart')
        load_logs(opt, logs=[args.log_gpr])

    def_logger = JSONLogger(path=logfile)
    opt.subscribe(Events.OPTMIZATION_STEP, def_logger)
    #print(args.iter)
    opt.maximize(
        init_points=args.seed,
        n_iter=args.iter,
        samples=None,
        eps=0.035,  # to reach an accuracy of 0.965
        solution=1,
        acq="ei",
        xi=0.01
    )
    end = time.time()
    passed = end-start
    print('target : ', opt.max['target'], ',  lazy_gpr: ', lazy_gpr)
    print('Time(sec.) : ' + str(passed) + "\n")
    lazy_gpr=True