import math

import cython
import numpy as np


@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.cdivision(True)     # Enable integer division (can improve performance)

def float_period(func, args, indices):
    """
    Generalized floating point period function that performs weighted moving averages
    or other operations over floating point periods and returns the result.
    
    Parameters:
    func: The function to call with adjusted arguments (either floor, ceil, or original values).
    args: The tuple of input arguments.
    indices: The tuple of indices that will be adjusted (floored and ceiled).
    
    Returns:
    The result of calling the function with the adjusted arguments, with combined results
    if necessary.
    """
    cdef int idx, floor_arg, ceil_arg
    cdef double floor_ratio, ceil_ratio
    cdef list calls = [[] for _ in range(len(args))]  # Holds adjusted values
    cdef list ratios = []  # Holds the ratios for weighting
    cdef int use_both = False
    cdef int n, m, i, j
    cdef double val

    # Validate input types
    for idx in indices:
        if not isinstance(args[idx], (int, float)):
            raise IndexError(f"Invalid float period index, should be float or int, got {type(args[idx])}")

    # Prepare the calls and compute the required values (floor, ceil, ratio)
    for idx, arg in enumerate(args):
        if idx in indices:
            floor_arg = math.floor(arg)
            ceil_arg = floor_arg + 1
            floor_ratio = ceil_arg - arg
            calls[idx].append(floor_arg)
            calls[idx].append(ceil_arg)
            ratios.append(floor_ratio)  # Store the ratio for later weighting
            use_both = True
        else:
            calls[idx].append(arg)  # Keep original value for non-adjusted indices

    try:
        # Prepare to call the function with the appropriate number of arguments
        results = []
        for i in range(len(calls[0])):  # Iterate over the number of calls (floor/ceil)
            call_args = [calls[j][i] for j in range(len(calls))]  # Gather arguments for this call
            result = func(*call_args)
            results.append(result)
    except:
        if "call_args" in locals():
            print(call_args)
        raise

    # Ensure we have the correct number of results and perform the weighted combination
    if len(results) == 1:
        return results[0]  # If only one result, return it directly
    else:
        # Perform element-wise weighted combination into a numpy array
        n = len(results[0])
        m = len(results)
        combined = np.empty(n, dtype=np.float64)
        for i in range(n):
            val = 0.0
            for j in range(m):
                val += results[j][i] * (ratios[j] if j < len(ratios) else 1)
            combined[i] = val
        return combined

# def float_period(func, tuple args, tuple indices):
#     """
#     Generalized floating point period function that performs weighted moving averages
#     or other operations over floating point periods and returns the result.
    
#     Parameters:
#     func: The function to call with adjusted arguments (either floor, ceil, or original values).
#     args: The tuple of input arguments.
#     indices: The tuple of indices that will be adjusted (floored and ceiled).
    
#     Returns:
#     The result of calling the function with the adjusted arguments, with combined results
#     if necessary.
#     """
#     cdef int idx, floor_arg, ceil_arg
#     cdef double floor_ratio, ceil_ratio
#     cdef list calls = [[], [], []]  # Holds floor, ceil, and ratio values
#     cdef int use_both = False
#     cdef list combined

#     for idx in indices:
#         if not isinstance(args[idx], (int, float)):
#             raise IndexError(f"Invalid float period index, should be float or int, got {type(args[idx])}")

#     # Prepare the calls and compute the required values (floor, ceil, ratio)
#     for idx, arg in enumerate(args):
#         if idx in indices:# and arg != math.floor(arg):
#             floor_arg = math.floor(arg)
#             ceil_arg = floor_arg + 1
#             floor_ratio = ceil_arg - arg
#             calls[0].append(floor_arg)
#             calls[1].append(ceil_arg)
#             calls[2].append(floor_ratio)  # Store the ratio for later weighting
#             use_both = True
#         else:
#             calls[0].append(arg)
#             calls[1].append(arg)
#             # calls[2].append(1.0)  # Full weight for the original period if not in indices

#     cdef list results = []

#     # Call the function with the appropriate number of arguments (1 or 2 sets)
#     for call in calls[:2 if use_both else 1]:
#         result = func(*call)
#         if len(indices) == 1:
#             results.append((np.array(result),))            
#         else:
#             results.append(result)

#     # Ensure we have the correct number of results and perform the weighted combination
#     if len(results) == 1:
#         return results[0][0] if len(results[0]) == 1 else results[0]  # If only one result, return it directly
#     else:
#         # Perform element-wise weighted combination (floor * floor_ratio + ceil * ceil_ratio)
#         combined = []
#         # print(len(calls[2]), len(results[0]))
#         for i, j, ratio in zip(results[0], results[1], calls[2]):
#             minlen = min(i.shape[-1], j.shape[-1])
#             combined.append(i[..., :minlen]*ratio + j[..., :minlen]*(1-ratio))
#         return combined[0] if len(combined) == 1 else combined


def derivative(ma_array, int period=1):
    cdef int n = ma_array.shape[0]
    # cdef np.ndarray[np.float64_t] derivative_result
    
    # Handle period == 1 using np.diff (optimized by Cython)
    if period == 1:
        return np.diff(ma_array)
    
    # Handle period > 1 using manual loop
    derivative_result = np.empty(n - period, dtype=np.float64)
    
    cdef int i
    for i in range(n - period):
        derivative_result[i] = ma_array[i + period] - ma_array[i]
    
    return derivative_result


def lag(array, amount):
    if not amount:
        return array
    return array[:-amount]
