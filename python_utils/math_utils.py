
"""
Created on Sun Apr  1 16:29:48 2018

@author: timcrose
"""

import math, random
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

def round_nearest_multiple(number, a, direction='standard'):
    '''
    Rounds number to nearest multiple of a. The returned number will have the
     same precision as a.
    '''
    if direction == 'down':
        return round(math.floor(number / a) * a, -int(math.floor(math.log10(a))))
    elif direction == 'up':
        return round(math.ceil(number / a) * a, -int(math.floor(math.log10(a))))
    elif direction == 'standard':
        return round(round(number / a) * a, -int(math.floor(math.log10(a))))

def mean(lst):
    return sum(lst) / float(len(lst))

def round(number, num_decimal_places, leave_int=False):
    '''
    number: number
        number to round
    num_decimal_places: int
        number of decimal places to round number to
    leave_int: bool
        True: If int(number) == number, then do not modify number
        False: Convert any ints to floats
    
    return: number (float)
        rounded number
    Purpose: round a number to the specified number of decimal places. Existing 
        round functions may not round correctly so that's why I built my own.
    '''
    if leave_int and int(number) == number:
        return number
    decimal_str = '1.'
    for decimal_place in range(num_decimal_places):
        decimal_str += '1'
    return float(Decimal(str(number)).quantize(Decimal(decimal_str), rounding=ROUND_HALF_UP))

def randrange_float(start, stop, step, num_decimal_places=4):
    return round(random.randint(0, int((stop - start) / step)) * step + start, num_decimal_places)

def round_matrix(matrix, num_decimal_places, leave_int=False):
    '''
    matrix: list of lists of numbers or 2D array of numbers
        matrix to round its elements
    num_decimal_places: int
        number of decimal places to round number to
    leave_int: bool
        True: If int(number) == number, then do not modify number
        False: Convert any ints to floats
    
    return: same type as input
        rounded matrix
    Purpose: round each number in a matrix to the specified number of decimal places. Existing 
        round functions may not round correctly so that's why I built my own.
    '''
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = round(matrix[i][j], num_decimal_places, leave_int=leave_int)
    return matrix

def round_lst(lst, num_decimal_places, leave_int=False):
    '''
    lst: list of numbers or 1D array of numbers
        list to round its elements
    num_decimal_places: int
        number of decimal places to round number to
    leave_int: bool
        True: If int(number) == number, then do not modify number
        False: Convert any ints to floats
    
    return: same type as input
        rounded list
    Purpose: round each number in a list to the specified number of decimal places. Existing 
        round functions may not round correctly so that's why I built my own.
    '''
    rounded_lst = [round(val, num_decimal_places, leave_int=leave_int) for val in lst]
    if isinstance(lst, list):
        return rounded_lst
    if isinstance(lst, np.ndarray):
        return np.array(rounded_lst)
    else:
        raise ValueError('So far only list and 1d numpy arrays are compatible.')


def r_sqrd(x, y):
    '''
    x: np.array, shape (,n)
        x coordinates of (x,y) points

    y: np.array, shape (,n)
        y coordinates of (x,y) points

    Return: float
        R^2 of best fit simple least squares regression line.

    Purpose: Calculate R^2 of best fit simple least squares regression line.
    '''
    x_dot_x = x.dot(x)
    x_dot_y = x.dot(y)
    x_mean = x.mean()
    x_sum = x.sum()
    y_mean = y.mean()
    denominator = x_dot_x - x_mean * x_sum
    m = (x_dot_y - y_mean * x_sum) / denominator
    b = (y_mean * x_dot_x - x_mean * x_dot_y) / denominator
    y_pred = m * x + b
    residuals = y - y_pred
    tot = y - y_mean
    return 1.0 - residuals.dot(residuals) / tot.dot(tot)
    