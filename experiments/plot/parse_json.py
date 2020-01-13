from bayes_opt import JSONLogger
from bayes_opt import Events
import numpy as np
from bayes_opt import BayesianOptimization
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
                    timings.append(iteration["datetime"])
                except KeyError:
                    pass

    return timings

# New optimizer is loaded with previously seen points
timings = parse_logs(logs=[args.log1])

myarray = np.asarray(timings)

timings_opt = np.ndarray(shape=(len(myarray)-1,1), dtype=float)
iterations = np.ndarray(shape=(len(myarray)-1,1), dtype=int)

for i in range(len(myarray)-1):
    entry = myarray[i+1]
    timings_opt[i] = entry['delta']#/8.0
    iterations[i] = i+1
xs = [x[0] for x in iterations]
ys = [y[0] for y in timings_opt]

print(np.mean(ys))

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(xs, ys, c='b', label='Original Cholesky',linewidth=1.0)

timings = parse_logs(logs=[args.log2])

myarray = np.asarray(timings)

timings_opt = np.ndarray(shape=(len(myarray)-1,1), dtype=float)
iterations = np.ndarray(shape=(len(myarray)-1,1), dtype=int)

for i in range(len(myarray)-1):
    entry = myarray[i+1]
    timings_opt[i] = entry['delta']
    iterations[i] = i+1
xs = [x[0] for x in iterations]
ys = [y[0] for y in timings_opt]

print(np.mean(ys))

ax.plot(xs, ys, c='r', label='Optimized Cholesky')
ax.set_xlabel('# Iterations')
ax.set_ylabel('Computational Overhead')
leg = plt.legend()

plt.savefig('timings_orig.png')