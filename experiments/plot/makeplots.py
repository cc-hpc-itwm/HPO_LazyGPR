from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import argrelextrema



def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

pbounds={'x': (-10, 10)}

def plot_gp_demo(optimizer, x, y, i):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        ' ',
        fontdict={'size':30}
    )

    gs = gridspec.GridSpec(1, 1)
    axis = plt.subplot(gs[0])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    n = 100
    Xtest = np.linspace(-15, 15, n).reshape(-1,1)
    mu2, sigma2 = posterior(optimizer, x_obs, y_obs, Xtest)
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Samples', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    f_post = np.random.normal(mu2.reshape(-1,1), sigma2.reshape(-1,1), size=(mu2.shape[0],3))
    axis.plot(Xtest, f_post)


    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.3, fc='blue', ec='None', label='95% confidence interval')

    axis.set_xlim((-10, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':15})
    axis.set_xlabel('x', fontdict={'size':15})

    axis.legend()
    #acq_par.legend()
    plt.savefig('1d-demo'+ str(i) + '.png')

def plot_gp(optimizer, x, y, i):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    #acq_par = plt.subplot(gs[2])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target', color='black')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Samples', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='blue', ec='None', label='95% confidence interval')

    axis.set_xlim((-10, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':15})
    axis.set_xlabel('x', fontdict={'size':15})

    utility_function = UtilityFunction(kind="ei", kappa=0, xi=1e-1)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility/(0.1*np.amax(utility)), label='Utility Function', color='red')
    acq.plot(x[np.argmax(utility)], np.max(utility/(0.1*np.amax(utility))), '*', markersize=15,
             label=u'Next Suggestion', markerfacecolor='blue', markeredgecolor='blue', markeredgewidth=1)
    acq.set_xlim((-10, 10))
    acq.set_ylim((0, np.max(utility/(0.1*np.amax(utility))) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':15})
    acq.set_xlabel('x', fontdict={'size':15})

    '''
    #parallel_util
    local_max = argrelextrema(utility, np.greater)
    acq_par.plot(x, utility/(0.1*np.amax(utility)), label='Utility Function', color='red')
    acq_par.plot(x[local_max], utility[local_max]/(0.1*np.amax(utility)), '*', markersize=15,
             label=u'Next Suggestion', markerfacecolor='blue', markeredgecolor='blue', markeredgewidth=1)
    acq_par.set_xlim((-10, 10))
    acq_par.set_ylim((0, np.max(utility/(0.1*np.amax(utility))) + 0.5))
    acq_par.set_ylabel('Utility', fontdict={'size':15})
    acq_par.set_xlabel('x', fontdict={'size':15})

    '''
    axis.legend()
    acq.legend()
    #acq_par.legend()
    plt.savefig('1d-func'+ str(i) + '.png')

def levy1d(x):
    w_x = 1 + (x - 1)/4.0
    term1 = (np.sin(math.pi*w_x))**2
    term3 = (w_x-1)**2 * (1+(np.sin(2*np.pi*w_x))**2)
    y = term1 + term3

    return -y

x = np.linspace(-10, 10, 10000).reshape(-1, 1)
y = levy1d(x)

optimizer = BayesianOptimization(levy1d, pbounds=pbounds, verbose=0, random_state=21)
optimizer.maximize(init_points=9, n_iter=1, eps=0, solution=0, acq="ei", xi=10e-1)
plot_gp_demo(optimizer, x, y, -1)

for i in range(5):
    optimizer.maximize(init_points=0, n_iter=1*i, eps=0, solution=0, acq="ei", xi=10e-1)
    plot_gp_demo(optimizer, x, y, i)


