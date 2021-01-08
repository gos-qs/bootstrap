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

def _compute_metric(array:np.ndarray or list, func, size:int) -> float:
    # metric for bootstrap function
    return func(np.random.choice(array, size=size, replace=True))

def bootstrap(array:np.ndarray, num_iters:int = 100, func = np.std, alpha:float = 0.95, size_scaler:float = 1, size:int = None) -> dict:
    r"""
    Entry function for obtaining metrics of an array with the alpha confidence interval, by resampling bootstrap method.
    array: array on which to apply the method
    num_iters = 100: number of iterations of the application of the resampling and function to the array
    func = np.std: the function to use to compute the metric (np.mean, np.median, np.std,... etc)
    alpha = 0.95: the alpha value in [0,1] to compute to, eg 0.95 gives the central 95% (2.5% on each tail)
    size_scaler = 1: sometime it's useful to apply the resampling over fewer or more of the array elements than are available (with replacement),
    a value of one means the number of elements sampled from the array in each iteration will be equal to the size of the array, 0.5 will randomly select half of the elements (with replacement) etc. 
    size = None: as above, the number of elements from the array to sample (with replacement) on each iteration, default is determined from the size_scaler but this overried any size_scaler input.
    """
    if alpha > 1 or alpha <= 0:
        raise Exception("Invalid alpha value passed, should be in [0, 1]: {0}".format(alpha))
    if num_iters <= 0:
        raise Exception("Invalid num_iters passed: {0}".format(num_iters))
    if size_scaler is not None and size_scaler <= 0:
        raise Exception("Invalid size_scaler passed: {0}".format(size_scaler))
    if size is not None and size <= 0:
        raise Exception("Invalid size given: {0}".format(size))
    if type(array) != np.ndarray:
        array = np.array(array)
    if size is None:
        size = int(np.prod(array.shape)) * int(size_scaler)
    
    targets = [(array, func, size) for i in range(num_iters)]
    with mp.Pool() as pool:
        result = pool.starmap(_compute_metric, targets)
    
    ave = np.mean(result)
    # get lower%th index and upper%th index as lower and upper bounds
    tail = (1 - alpha)*0.5
    result = np.sort(result)
    lower_idx = int(num_iters * tail)
    upper_idx = int(num_iters * (1 - tail))
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