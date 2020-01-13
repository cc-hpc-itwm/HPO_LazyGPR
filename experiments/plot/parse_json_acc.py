import sys
sys.path.append("../../bayes_opt")
sys.path.append("../../")

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import numpy as np

import matplotlib.pyplot as plt


import argparse #   import argparse
parser = argparse.ArgumentParser(description='ResNet')
parser.add_argument('--log1', default='./log1.json', help='Number of seed points')
parser.add_argument('--log2', default='./log2.json', help='Number of iterations')

args = parser.parse_args()

def parse_logs(logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        print(len(logs))
        logs = [logs]

    timings = []
    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    timings.append(iteration["target"])
                except KeyError:
                    pass

    return timings

# New optimizer is loaded with previously seen points
log1 = parse_logs(logs=[args.log1])

log1arr = np.asarray(log1)

acc = np.ndarray(shape=(len(log1arr)-1,1), dtype=float)
iterations = np.ndarray(shape=(len(log1arr)-1,1), dtype=int)
max_acc = -100
max_acc_opt = -100

print('Naive Cholesky')

for i in range(len(log1arr)-1):
    acc[i] = log1arr[i+1]
    if(acc[i]> max_acc):
        print(i, 'improved: ' , max_acc, '->', acc[i])
        max_acc = acc[i]
    iterations[i] = i+1
xs = [x[0] for x in iterations]
ys = [y[0] for y in acc]


print('Optimized Cholesky')
log2 = parse_logs(logs=[args.log2])

log2arr = np.asarray(log2)
acc_opt = np.ndarray(shape=(len(log2arr)-1,1), dtype=float)
iterations = np.ndarray(shape=(len(log2arr)-1,1), dtype=int)

for i in range(len(log2arr)-1):
    acc_opt[i] = log2arr[i+1]
    if(acc_opt[i]> max_acc_opt):
        print(i, 'improved: ' , max_acc_opt, '->', acc_opt[i])
        max_acc_opt = acc_opt[i]
    iterations[i] = i+1
xs = [x[0] for x in iterations]
ys = [y[0] for y in acc_opt]

