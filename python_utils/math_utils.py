
"""
Created on Sun Apr  1 16:29:48 2018

@author: timcrose
"""

import math, random
import numpy as np
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
        return round(number, -int(math.floor(math.log10(a))))


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

    
def poly1d_fit(x, y, degree, get_r_sqrd=True, get_local_maximums=False, 
        get_local_mininums=False, get_global_maximum=False, 
        get_global_minimum=False, global_extrema_method='data_or_curve', 
        tol_num_digits=8, x_tol_for_duplicates=None, 
        y_tol_factor_for_extraneous_extrema=None):
    '''
    x: 1D list or np.array
        List of x axis (independent variable) values which correspond in order 
        to the y values.

    y: 1D list or np.array
        List of y axis (dependent variable) values which correspond in order to
        the x values.

    degree: int
        Degree (order) (Highest power) of the polynomial with which to fit the 
        x,y data.

    get_r_sqrd: bool
        True: Add an r_sqrd key to the returned dict. The value is the R^2 
        Pearson correlation of the fitted polynomial to the data.
        
        False: Do not add an r_sqrd key.

    get_local_maximums: bool
        True: Add x_max and y_max keys to the returned dict. See below for a 
            description of the values. Local maxima are the maxima as found by 
            taking the 1st and 2nd order derivatives of the fitted polynomial.

        False: Do  not add an x_max or y_max key. However, these keys will be 
            added if get_global_maximum is True. (But the value will not 
            include local maxima as found by taking the 1st and 2nd order 
            derivatives of the fitted polynomial).

    get_local_mininums: bool
        See get_local_maximums.

    get_global_maximum: bool
        True: Include the global maximum value in the value of the x_max and 
            y_max keys in the returned dict. If more than 1 x value has y 
            value of the global maximum, they are all added. For each value 
            added, an x value is added to x_max and a y value is added to 
            y_max. This includes the endpoints. The "global" max is only taken 
            over the range of x values provided in the x input parameter. See 
            global_extrema_method for more info.

        False: Do not add the global maximum to the returned dict. If 
            get_local_maximums is False too, then x_max and y_max will not be 
            added as keys to the returned dict.

    get_global_minimum: bool
        See get_global_maximum.

    global_extrema_method: str
        This argument allows the user to specify the types of global extrema to
        include. Allowed values are "data", "data_or_curve", or "curve". See 
        try_to_avoid_duplicates for exceptions to the descriptions below.
        
        "data": Use the max value in the provided y values as the global max. 
            (Similar for min).
            
        "curve": Use the max value in the fitted polynomial within the range 
            of x values provided as the global max. (Similar for min).
            
        "data_or_curve": Include the values gotten in "data" and then, if the 
            values gotten in "curve" are higher, include them too.

    tol_num_digits: int
        Total number of decimal digits to be included in the smallest x 
        increment (difference in consecutive x values in the input parameter x.
        This parameter is used for rounding purposes. Thus, usually a large 
        number of digits like 8 is good.

    x_tol_for_duplicates: float, str, or None
        When trying to determine whether certain extrema are duplicates (in x 
        value), the x values do not have to match but have to be 
        within x_tol_for_duplicates (if specified) or within the lowest 
        difference in consecutive x values in the input parameter x 
        (if x_tol_for_duplicates is 'default') to be considered a duplicate. If
        x_tol_for_duplicates is None, then add any extrema gotten by any of the
        methods in this function according to the input parameters regardless 
        if another method has already added this extremum. If it is not None 
        then if there is an extremum that might be added to the returned dict, 
        check to see if it is very close in x value to an extrema that was 
        already added to the returned dict. If it is, then do not add it. For 
        example, let's say you add local maxima and then you want to add the 
        global maximum. If the global maximum is one of the local maxima 
        already-added then try_to_avoid_duplicates=True would decline to add 
        the global maximum because it was already added, but 
        try_to_avoid_duplicates=False would add it anyway.

    y_tol_factor_for_extraneous_extrema: float or None
        Sometimes the max of the fitted polynomial is far above the max of the 
        y data points. This could be undesirable to include in the y_max list. 
        So, if a maximum of the fitted polynomial is greater than 
        y_tol_for_extraneous_extrema + max(y) then do not include it. (similar 
        for min). If None, do not check for this.
        y_tol_factor_for_extraneous_extrema is assumed to be positive.
        Note that
        y_tol_for_extraneous_extrema = (max([abs(max(y)), abs(min(y))]) 
                    * y_tol_factor_for_extraneous_extrema)

    Returns: fit
        fit: dict
            The keys vary depending on what the user asks for, but all the 
            available keys are described here. If a key is only included upon 
            asking for it, this is specified below as well. 
            
            "degree": int
                Degree (order) (Highest power) of the polynomial with which to 
                fit the x,y data.

            "coeffs": np.ndarray, shape: (deg + 1,)
                Polynomial coefficients, highest power first.

            "p": np.poly1d
                Polynomial object. Use it like a function. e.g. p(x) where x 
                is a list returns a list of output values.

            "r_sqrd": float
                Pearson correlation of the fitted polynomial to the data.

            "x_max": list of float
                List of x values that correspond to y values that are a maximum
                either locally or globally. This is only included in the fit 
                dict if get_local_maximums or get_global_maximum is True.

            "y_max": list of float
                See x_max
            
            "x_min": list of float
                See x_max

            "y_min": list of float
                See x_max
    
    Purpose:
        Fit a 1D polynomial to lists of x and y data. Get the coefficients of 
        the polynomial. Get the np.poly1d object for easy evaluation of new 
        data points with the polynomial. Get the R^2 Pearson correlation, if 
        desired. Include the local minima as gotten by taking 1st and 2nd order
        derivatives, if desired. Include the global extrema within the x-range 
        of the provided x list, if desired. These global extrema can be from 
        the data points or from the fitted curve, or both. "duplicate" extrema 
        can be excluded, if desired. Extrema too far above max(y) or too far
        below min(y) can be excluded, if desired.
        
    Other variables in the function:
    
    yhat: float
        The predicted y values based on the fit at the x values contained in
        the input x.
        
    ybar: float
        The average of the input y values.
        
    ssreg: float
        Sum of Squares Regression. This is the sum of the squared differences
        between the predicted y values and the average y value.
        
    sstot: float
        Sum of Squares Total. This is the sum of the squared differences
        between the empirical y values and the average y value.
        
    lower_bound_x: float
        minimum of input x list
        
    upper_bound_x: float
        maximum of input x list
        
    higher_endpoint: float
        higher_endpoint is set to lower_bound_x if the predicted y value at 
        x = lower_bound_x is greater than the predicted y value at 
        x = upper_bound_x. Otherwise, higher_endpoint set to upper_bound_x.
        
    lower_endpoint: float
        lower_endpoint is set to lower_bound_x if the predicted y value at 
        x = lower_bound_x is less than the predicted y value at 
        x = upper_bound_x. Otherwise, lower_endpoint set to upper_bound_x.
        
    global_maximum: float
        maximum of input y list
        
    global_minimum: float
        minimum of input y list
        
    arg_maximums: list of int
        List of indices in the input y list where the value is equal to the
        global_maximum.
        
    arg_minimums: list of int
        List of indices in the input y list where the value is equal to the
        global_minimum.
    '''
    from python_utils import list_utils
    fit = {'degree': degree}
    fit['coeffs'] = np.polyfit(x, y, degree)
    fit['p'] = np.poly1d(fit['coeffs'])
    if get_r_sqrd:
        # Caclulate the Pearson correlation R^2 value between the fitted y
        # values and the empirical y values.
        yhat = fit['p'](x)
        ybar = np.sum(y) / float(len(y))
        ssreg = np.sum((yhat - ybar)**2)
        sstot = np.sum((y - ybar)**2)
        fit['r_squared'] = ssreg / sstot

    if get_local_maximums or get_local_mininums:
        if y_tol_factor_for_extraneous_extrema is not None:
            y_tol_for_extraneous_extrema = (max([abs(max(y)), abs(min(y))]) 
                    * y_tol_factor_for_extraneous_extrema)
        
        x_arr = np.array(x)
        # Take the first derivative, set equal to 0 to find critical values.
        crit = fit['p'].deriv().r
        r_crit = crit[crit.imag == 0].real
        # Take second derivative to tell whether we have a max or min at the
        # critical values.
        test = fit['p'].deriv(2)(r_crit)
        if get_local_maximums:
            if y_tol_factor_for_extraneous_extrema is None:
                fit['x_max'] = r_crit[test < 0]
                fit['y_max'] = fit['p'](fit['x_max'])
            else:
                x_maxes = r_crit[test < 0]
                y_maxes = fit['p'](x_maxes)
                # Gather indices of critical points to delete. Ones that will
                # be deleted are deemed extraneous (y values are outside the 
                # expected or reasonable range).
                i_to_del = []
                for i in range(len(x_maxes)):
                    if y_maxes[i] > y_tol_for_extraneous_extrema + max(y):
                        i_to_del.append(i)
                fit['x_max'] = list_utils.multi_delete(x_maxes, i_to_del)
                fit['y_max'] = list_utils.multi_delete(y_maxes, i_to_del)
        if get_local_mininums:
            if y_tol_factor_for_extraneous_extrema is None:
                fit['x_min'] = r_crit[test > 0]
                fit['y_min'] = fit['p'](fit['x_min'])
            else:
                x_mins = r_crit[test > 0]
                y_mins = fit['p'](x_mins)
                # Gather indices of critical points to delete. Ones that will
                # be deleted are deemed extraneous (y values are outside the 
                # expected or reasonable range).
                i_to_del = []
                for i in range(len(x_mins)):
                    if y_mins[i] < min(y) - y_tol_for_extraneous_extrema:
                        i_to_del.append(i)
                fit['x_min'] = list_utils.multi_delete(x_mins, i_to_del)
                fit['y_min'] = list_utils.multi_delete(y_mins, i_to_del)
    if get_global_maximum or get_global_minimum:
        lower_bound_x = min(x)
        upper_bound_x = max(x)
        if x_tol_for_duplicates == 'default':
            x_tol_for_duplicates = round(np.min(np.abs(x_arr[:-1] - x_arr[1:])),
                    tol_num_digits)
            
        if get_global_maximum:
            global_maximum = max(y)
            if 'x_max' not in fit:
                fit['x_max'] = []
                fit['y_max'] = []
            if (global_extrema_method == 'data_or_curve' or 
                    global_extrema_method == 'data'):
                        
                arg_maximums = [i for i,y_val in enumerate(y) if 
                        y_val == global_maximum]
                
                for i in arg_maximums:
                    # Only keep it if the x value is sufficiently far away to 
                    # avoid risk of duplicate
                    if (len(fit['x_max']) == 0 or x_tol_for_duplicates is None 
                            or np.min(np.abs(x[i] - np.array(fit['x_max']))) 
                            > x_tol_for_duplicates):
                                
                        fit['x_max'] = fit['x_max'] + [x[i]]
                        fit['y_max'] = fit['y_max'] + [y[i]]

            if (global_extrema_method == 'data_or_curve' or 
                    global_extrema_method == 'curve'):
                
                if fit['p'](lower_bound_x) > fit['p'](upper_bound_x):
                    higher_endpoint = lower_bound_x
                else:
                    higher_endpoint = upper_bound_x
                # We are considering whether to add the higher endpoint as a
                # global maximum. We can start to think about adding the 
                # endpoint if either it's higher than any current y value in
                # y_max (because it's supposed to be the global maximum) or if
                # there are no values in y_max (no contenders).
                if (len(fit['y_max']) == 0 or 
                        max(fit['y_max']) < fit['p'](higher_endpoint)):
                    
                    # So we know it's greater than the current highest value
                    # in y_max, but maybe it's within x_tol_for_duplicates in
                    # the x direction of the x coordinate of the highest known
                    # y value (except the endpoint). Thus, adding the endpoint
                    # is kind of like adding the same max again and perhaps the
                    # user wants to avoid such duplicates. But if not (to 
                    # either of those conditions), then continue checking
                    # whether the endpoint should be added.
                    if (x_tol_for_duplicates is None or 
                            len(fit['x_max']) == 0 or
                            np.min(np.abs(
                            higher_endpoint - np.array(fit['x_max']))) 
                            > x_tol_for_duplicates):
                        
                        # So we now know it's a global maximum and is not a 
                        # duplicate (if we cared about duplicates). The final
                        # check is make sure it's not too much higher than the
                        # global_maximum.
                        if (y_tol_for_extraneous_extrema is None or 
                                global_maximum + y_tol_for_extraneous_extrema 
                                > fit['p'](higher_endpoint)):
                            
                            fit['x_max'] = fit['x_max'] + [higher_endpoint]
                            
                            fit['y_max'] = (fit['y_max'] 
                                    + [fit['p'](higher_endpoint)])
                
        if get_global_minimum:
            global_minimum = min(y)
            if 'x_min' not in fit:
                fit['x_min'] = []
                fit['y_min'] = []
            if (global_extrema_method == 'data_or_curve' or 
                    global_extrema_method == 'data'):
                
                arg_minimums = [i for i,y_val in enumerate(y) if 
                        y_val == global_minimum]
                
                for i in arg_minimums:
                    # Only keep it if the x value is sufficiently far away to 
                    # avoid risk of duplicate
                    if (len(fit['x_min']) == 0 or x_tol_for_duplicates is None 
                            or np.min(np.abs(
                            x[i] - np.array(fit['x_min']))) 
                            > x_tol_for_duplicates):
                        
                        fit['x_min'] = fit['x_min'] + [x[i]]
                        fit['y_min'] = fit['y_min'] + [y[i]]

            if (global_extrema_method == 'data_or_curve' 
                    or global_extrema_method == 'curve'):
                
                if fit['p'](lower_bound_x) < fit['p'](upper_bound_x):
                    lower_endpoint = lower_bound_x
                else:
                    lower_endpoint = upper_bound_x
                # We are considering whether to add the lower endpoint as a
                # global minimum. We can start to think about adding the 
                # endpoint if either it's lower than any current y value in
                # y_min (because it's supposed to be the global minimum) or if
                # there are no values in y_min (no contenders).                    
                if (len(fit['y_min']) == 0 or min(fit['y_min']) 
                        > fit['p'](lower_endpoint)):
                    
                    # So we know it's less than the current lowest value
                    # in y_min, but maybe it's within x_tol_for_duplicates in
                    # the x direction of the x coordinate of the lowest known
                    # y value (except the endpoint). Thus, adding the endpoint
                    # is kind of like adding the same min again and perhaps the
                    # user wants to avoid such duplicates. But if not (to 
                    # either of those conditions), then continue checking
                    # whether the endpoint should be added.
                    if (x_tol_for_duplicates is None or np.min(np.abs(
                            lower_endpoint - np.array(fit['x_min']))) 
                            > x_tol_for_duplicates):
                            
                        # So we now know it's a global minimum and is not a 
                        # duplicate (if we cared about duplicates). The final
                        # check is make sure it's not too much lower than the
                        # global_minimum.
                        if (y_tol_factor_for_extraneous_extrema is None or
                                global_minimum - y_tol_for_extraneous_extrema 
                                > fit['p'](higher_endpoint)):
                            
                            fit['x_min'] = fit['x_min'] + [lower_endpoint]
                            fit['y_min'] = (fit['y_min'] 
                                    + [fit['p'](lower_endpoint)])
         
    return fit