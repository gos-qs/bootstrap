#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 14:33:47 2020

bootstrap an array with a given metric

@author: George
"""

import numpy as np
import multiprocessing as mp

def _compute_metric(array: np.ndarray or list,
                    func,
                    size: int
                    ) -> float:
    """
    helper function for multiprocessing.

    Parameters
    ----------
    array : np.ndarray or list
        array from which to sample elements.
    func : callable
        function to apply to sample.
    size : int
        number of elements to sample from array.

    Returns
    -------
    float
    """
    return func(np.random.choice(array, size=size, replace=True))

def bootstrap(array: np.ndarray,
              num_iters: int = 100,
              func = np.std,
              alpha: float = 0.95,
              size_scaler: float = 1,
              size: int = None,
              use_mp: bool = True
              ) -> dict:
    """
    function for obtaining metrics of an array with the alpha confidence 
    interval, by a bootstrap resampling method.
    
    >>> array = np.power(np.random.random(size = (200,)) * 100, 0.25) - 1
    >>> res = bootstrap(
    >>>     array,
    >>>     num_iters = 100,
    >>>     func = np.mean,
    >>>     size_scaler = 0.1,
    >>>     use_mp = False
    >>>     )
    
    Parameters
    ----------
    array : np.ndarray
        array on which to apply the method. contains all the elements from 
        which to take random samples.
    num_iters : int, optional
        number of iterations of resampling a dataset from the array and 
        applying the function to the dataset.
        The default is 100.
    func : callable, optional
        function to apply to the dataset.
        The default is np.std.
    alpha : float, optional
        the alpha value in [0,1] to compute to, eg 0.95 gives the central 95% 
        (2.5% on each tail).
        The default is 0.95.
    size_scaler : float, optional
        sometime it's useful to apply the resampling over fewer or more of the
        array elements than are available (with replacement). so pass a decimal
        to compute the size as size_scaler * len(array).
        a value of 1 means the number of elements sampled from the array in each
        iteration will be equal to the size of the array. 0.5 will randomly 
        select half of the elements (with replacement) etc.
        The default is 1.
    size : int, optional
        as above, the number of elements from the array to sample, with 
        replacement, on each iteration. default is determined from the 
        size_scaler but this overried any size_scaler input.
        The default is None.
    use_mp : bool, optional
        if True, uses multiprocessing pool to speed up the function time.
        The default is True.

    Raises
    ------
    Exception
        on invalid parameters.
    
    Returns
    -------
    dict
        {
        "average_metric": mean average of all resamples for the metric given,
        "lower_confidence_bound": lower confidence interval bound value,
        "upper_confidence_bound": upper confidence interval bound value,
        "array": result
        }
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
        size = int(np.prod(array.shape) * size_scaler)
    
    if use_mp:
        targets = [(array, func, size) for i in range(num_iters)]
        with mp.Pool() as pool:
            result = pool.starmap(_compute_metric, targets)
    else:
        result = [func(np.random.choice(
            array,
            size = size,
            replace = True)
            ) for i in range(num_iters)]
    
    ave = np.mean(result)
    # get lower%th index and upper%th index as lower and upper bounds
    tail = (1 - alpha)*0.5
    result = np.sort(result)
    lower_idx = int(num_iters * tail)
    upper_idx = int(num_iters * (1 - tail))
    lower_val = result[lower_idx]
    upper_val = result[upper_idx]
    return {
        "average_metric": ave,
        "lower_confidence_bound": lower_val,
        "upper_confidence_bound": upper_val,
        "array": result
        }


if __name__ == '__main__':
    array = np.power(np.random.random(size = (200,)) * 100, 0.25) - 1
    res = bootstrap(
        array,
        num_iters = 100,
        func = np.mean,
        size_scaler = 0.1,
        use_mp = False
        )
    print(
        'Average:', res['average_metric'],
        'LowerBound', res['lower_confidence_bound'],
        'UpperBound', res['upper_confidence_bound']
        )
    import matplotlib.pyplot as plt
    
    height, _ = np.histogram(array, bins = 11, density = True)
    height = np.max(height)
    plt.hist(array, bins = 11, density = True, color = 'blue')
    plt.plot(
        [res['average_metric'], res['average_metric']],
        [0, height],
        color = 'black',
        label = 'Mean:{0}'.format(np.round(res['average_metric'], 2))
        )
    plt.plot(
        [res['lower_confidence_bound'], res['lower_confidence_bound']],
        [0, height],
        color = 'red',
        label = 'LowerBound:{0}'.format(np.round(res['lower_confidence_bound'], 2))
        )
    plt.plot(
        [res['upper_confidence_bound'], res['upper_confidence_bound']],
        [0, height],
        color = 'red',
        label = 'UpperBound:{0}'.format(np.round(res['upper_confidence_bound'], 2))
        )
    plt.legend()
    plt.show()
