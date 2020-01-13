import sys
sys.path.append("../bayes_opt")
sys.path.append("..")

# Bayesian optimization using classical and lazy gaussian process
from lazy_gaussian_process.bayesian_optimization import BayesianOptimization

import resnet32_network as resnet32
import time
# from bayes_opt import JSONLogger
# from bayes_opt.util import load_logs
# from bayes_opt import Events

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from tensorflow.keras.datasets import cifar10


image_size = 28
num_labels = 10
num_channels = 1 # grayscale


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#The function which is to be maximized with the given hyperparameters weight_decay, momentum, learning_rate
def black_box_function(weight_decay, momentum, learning_rate):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    print ("Training the CIFAR-10 dataset with")
    print('(lr: ', learning_rate, ', momentum: ', momentum, ', wd: ', weight_decay, ')')

    #Training of ResNet32 with given hyperparameter, this function returns accuracy of ResNet32
    return resnet32.train_resnet(x_train, y_train, x_test, y_test, epochs=10,
        learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Bounded region of parameter space
pbounds = {'weight_decay': (0, 0.001), 'momentum': (0, 0.99), 'learning_rate': (10e-4, 0.1)}

import argparse #   import argparse
parser = argparse.ArgumentParser(description='ResNet')
parser.add_argument('--seed', type=int, default=100, help='Number of seed points')
parser.add_argument('--iter', type=int, default=500, help='Number of iterations')
parser.add_argument('--log_gpr', default='./logs_cifar10.json', help='logfile')
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
        eps=0.21, # to reach an accuracy of 0.79
        solution=1,
        acq="ei",
        xi=0.01
    )
    end = time.time()
    passed = end-start
    print('target : ', opt.max['target'], ',  lazy_gpr: ', lazy_gpr)
    print('Time(sec.) : ' + str(passed) + "\n")
    lazy_gpr=True