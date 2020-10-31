#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:33:47 2020

bootstrap an array with a given metric

usage:
    array = np.array([1,4,2,3,4,2,4,5,1,4,2,5,7,9,1,3,6,2,3,9,10,2,3,1,4,7,6,1,3,1,2,0,4,3,0,0,0,1,3])
    from bootstrap.bootstrap import bootstrap
    res = bootstrap(array, num_iters=100, function=np.mean)
    print('Average:', res['average_metric'],
          'LowerBound', res['lower_confidence_bound'],
          'UpperBound', res['upper_confidence_bound'])
    import matplotlib.pyplot as plt
    
    plt.hist(array, bins=11, density=True, color='blue')
    plt.plot([res['average_metric'], res['average_metric']], [0, 1], color='black', label='Mean:{0}'.format(res['average_metric']))
    plt.plot([res['lower_confidence_bound'], res['lower_confidence_bound']], [0, 1], color='red', label='LowerBound:{0}'.format(np.round(res['lower_confidence_bound'], 2)))
    plt.plot([res['upper_confidence_bound'], res['upper_confidence_bound']], [0, 1], color='red', label='UpperBound:{0}'.format(np.round(res['upper_confidence_bound'], 2)))
    plt.legend()
    plt.show()

@author: George
"""

import numpy as np
import multiprocessing as mp

def _compute_metric(array, func, size):
    # metric for bootstrap function
    return func(np.random.choice(array, size=size, replace=True))

def bootstrap(array, num_iters=100, func=np.std, alpha=0.95, size_scaler=1, size=None):
    # entry function for obtaining metrics of an array with the alpha confidence interval, by resampling bootstrap method
    if size is None:
        size = int(np.prod(array.shape)) * int(size_scaler)
    
    targets = [(array, func, size) for i in range(num_iters)]
    with mp.Pool() as pool:
        result = pool.starmap(_compute_metric, targets)
    
    ave = np.mean(result)
    # get 2.5%th index and 97.5%th index as lower and upper bounds
    result = np.sort(result)
    lower_idx = int(num_iters*0.025)
    upper_idx = int(num_iters*0.975)
    lower_val = result[lower_idx]
    upper_val = result[upper_idx]
    
    return {'average_metric':ave, 'lower_confidence_bound':lower_val, 'upper_confidence_bound':upper_val, 'array':result}

if __name__ == '__main__':
    
    array = np.array([1,4,2,3,4,2,4,5,1,4,2,5,7,8,0,2,3,4,2,6,3,2,4,1,3,6,2,3,9,10,2,3,1,4,7,6,1,3,1,2,0,4,3,0,2,0,1,0,1,3])
    res = bootstrap(array, num_iters=100, func=np.mean, size_scaler = 5)
    print('Average:', res['average_metric'],
          'LowerBound', res['lower_confidence_bound'],
          'UpperBound', res['upper_confidence_bound'])
    import matplotlib.pyplot as plt
    
    height, _ = np.histogram(array, bins=11, density=True)
    height = np.max(height)
    plt.hist(array, bins=11, density=True, color='blue')
    plt.plot([res['average_metric'], res['average_metric']], [0, height], color='black', label='Mean:{0}'.format(np.round(res['average_metric'], 2)))
    plt.plot([res['lower_confidence_bound'], res['lower_confidence_bound']], [0, height], color='red', label='LowerBound:{0}'.format(np.round(res['lower_confidence_bound'], 2)))
    plt.plot([res['upper_confidence_bound'], res['upper_confidence_bound']], [0, height], color='red', label='UpperBound:{0}'.format(np.round(res['upper_confidence_bound'], 2)))
    plt.legend()
    plt.show()