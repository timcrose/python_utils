
"""
Created on Sun Apr  1 16:29:48 2018

@author: timcrose
"""

import math, random
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from decimal import Decimal, ROUND_HALF_UP


def prepare_means_stds_weights(
    means, 
    stds,
    weights, 
    already_normalized_weights=True
):
    """
    Ensure means, stds, and weights are properly shaped and that the weights
    are normalized. This is usually done in preparation for mixing Gaussian
    distributions to get the mixed PDF value at a given location, x, or, to get
    the mixed CDF value at the upper integration limit, t.

    Parameters
    ----------
    means: Vector
        These are the means (μᵢ) of the Gaussian distributions you want to mix.
        
    stds: Vector
        These are the standard deviations (σᵢ) of the Gaussian distributions 
        you want to mix.
        
    weights: Vector
        These are the weights (wᵢ) of the Gaussian distributions you want to mix.
        If they don't already sum to 1, set already_normalized_weights to False
        which will make this function automatically normalize them to sum to 1.
        
    already_normalized_weights: bool
        True: Assert weights.sum() == 1
        False: Normalize weights to sum to 1.

    Returns
    -------
    means, stds, weights
    """
    n = len(means)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if already_normalized_weights:
        if not np.allclose(weights.sum(), 1, atol=1e-7):
            raise ValueError('Weights sum should be normalized but arent. weights.sum():', weights.sum())
    else:
        weights = weights / weights.sum()
    if n == 0 or not (means.shape == stds.shape == weights.shape):
        raise ValueError('Invalid shape found. means.shape', means.shape, 'stds.shape', stds.shape, 'weights.shape', weights.shape)
    return means[:, None], stds[:, None], weights


def get_mixed_gaussian_pdf_values(
    x,
    means, 
    stds,
    weights, 
    already_normalized_weights=True
):
    """
    Evalute the probability-density function of a Gaussian mixture at value(s) of x.

    Parameters
    ----------
    x: Union[Scalar, Vector]
        Where to evaluate the mixed PDF at. If x is a Scalar, then the PDF value
        will be returned as a scalar. If instead, x is a Vector, then an 
        ndarray will be returned with each element being the PDF value gotten
        when inputting the corresponding element in x.

    means: Vector
        These are the means (μᵢ) of the Gaussian distributions you want to mix.
        
    stds: Vector
        These are the standard deviations (σᵢ) of the Gaussian distributions 
        you want to mix.
        
    weights: Vector
        These are the weights (wᵢ) of the Gaussian distributions you want to mix.
        If they don't already sum to 1, set already_normalized_weights to False
        which will make this function automatically normalize them to sum to 1.
        
    already_normalized_weights: bool
        True: Assert weights.sum() == 1
        False: Normalize weights to sum to 1.
        
    Returns
    -------
    float or ndarray
        ``f_mix(x) = Σ wᵢ · 𝒩(x | μᵢ, σᵢ²)`` evaluated element-wise.

    Examples
    --------
    >>> means  = [0.0,  2.0,  4.0]
    >>> stds = [1.0,  0.5,  2.0]
    >>> weights  = [0.3,  0.4,  0.3]
    >>> get_mixed_gaussian_pdf_values(0.0, means, stds, weights)
    0.12788839
    >>> xs = np.linspace(-2, 2, 5)
    >>> get_mixed_gaussian_pdf_values(xs, means, stds, weights)
    array([0.01686207 0.07522047 0.12788839 0.13521163 0.37164672])
    """
    means, stds, weights = prepare_means_stds_weights(means, stds, weights, 
already_normalized_weights=already_normalized_weights)
    
    if hasattr(x, '__iter__'):
        x_arr = np.asarray(x, dtype=float)
        return np.dot(weights, norm.pdf(x_arr, means, stds))
    else:
        return np.dot(weights, norm.pdf(x, means, stds))[0]


def get_mixed_gaussian_cdf_values(
    t,
    means, 
    stds,
    weights, 
    already_normalized_weights=True
):
    """
    Cumulative-distribution function value at t of the weighted mixture of Gaussians 
    entailed by the given means and stds (or for each value of t if t is a
    Vector).

    Computes the analytic integral  
    ``F_mix(t) = ∫_{-∞}^{t} f_mix(x) dx = Σ wᵢ · Φ((t-μᵢ)/σᵢ)``,
    where *Φ* is the standard normal CDF. Note that linearity of the integral
    function over addition means that the CDF of the mixture is just the
    sum of the CDFs.

    Parameters
    ----------
    t: Union[Scalar, Vector]
        Upper integration limit(s). If t is a Scalar, then the CDF value
        will be returned as a scalar. If instead, t is a Vector, then an 
        ndarray will be returned with each element being the CDF value gotten
        when inputting the corresponding element in t.

    means: Vector
        These are the means (μᵢ) of the Gaussian distributions you want to mix.
        
    stds: Vector
        These are the standard deviations (σᵢ) of the Gaussian distributions 
        you want to mix.
        
    weights: Vector
        These are the weights (wᵢ) of the Gaussian distributions you want to mix.
        If they don't already sum to 1, set already_normalized_weights to False
        which will make this function automatically normalize them to sum to 1.
        
    already_normalized_weights: bool
        True: Assert weights.sum() == 1
        False: Normalize weights to sum to 1.

    Returns
    -------
    float or ndarray
        Mixture CDF evaluated element-wise at *t*.

    Examples
    --------
    >>> mmeans  = [0.0, 2.0, 4.0]
    >>> stds = [1.0, 0.5, 2.0]
    >>> weights  = [0.3, 0.4, 0.3]
    >>> get_mixed_gaussian_cdf_values(1.0, means, stds, weights)
    0.28154654
    >>> get_mixed_gaussian_cdf_values([-1, 0, 1], means, stds, weights)
    array([0.04945948 0.15683771 0.28154564])
    """
    means, stds, weights = prepare_means_stds_weights(means, stds, weights, 
already_normalized_weights=already_normalized_weights)
    
    if hasattr(t, '__iter__'):
        t_arr = np.asarray(t, dtype=float)
        return np.dot(weights, norm.cdf(t_arr, means, stds))
    else:
        return np.dot(weights, norm.cdf(t, means, stds))[0]
    
    
def get_mixed_gaussian_ppf_values(
    prob,
    means,
    stds,
    weights,
    already_normalized_weights=True,
    tol=1e-8,
):
    """
    Percent point function (inverse CDF) for a Gaussian mixture.

    This function finds the threshold(s) t such that the cumulative distribution 
    of the Gaussian mixture equals the specified probability, i.e.,
    
        get_mixed_gaussian_cdf_values(t, means, stds, weights) = prob.

    The input probability can be a scalar or an array-like object. If an array-like 
    of probabilities is provided, the function returns an ndarray of threshold values 
    with one inverse CDF value per probability. A scalar input probability will 
    cause this function to output a scalar float.

    Parameters
    ----------
    prob : Union[Scalar, Vector]
        Desired cumulative probability (or probabilities). Each value must lie 
        in the interval (0, 1) exclusive. A scalar yields a float output, while an
        array-like input returns an ndarray of the same shape.
        
    means: Vector
        These are the means (μᵢ) of the Gaussian distributions you want to mix.
        
    stds: Vector
        These are the standard deviations (σᵢ) of the Gaussian distributions 
        you want to mix.
        
    weights: Vector
        These are the weights (wᵢ) of the Gaussian distributions you want to mix.
        If they don't already sum to 1, set already_normalized_weights to False
        which will make this function automatically normalize them to sum to 1.
        
    already_normalized_weights: bool
        True: Assert weights.sum() == 1
        False: Normalize weights to sum to 1.
        
    tol : float
        Tolerance for the Brent's method root-finding algorithm (default is 1e-8).

    Returns
    -------
    float or ndarray
         The threshold value(s) t such that the mixture CDF at t equals the specified 
         probability value(s). Scalar input returns a float, and array-like input returns 
         an ndarray.

    Raises
    ------
    ValueError
         If any probability is not in (0, 1) or if means, stds, and weights do not have 
         identical shapes.

    Notes
    -----
    The Gaussian mixture CDF is given by:
    
        F(t) = Σ wᵢ · norm.cdf(t, μᵢ, σᵢ)
    
    and this function numerically inverts that relationship for a given cumulative area.

    Examples
    --------
    >>> means = [0.0, 2.0, 4.0]
    >>> stds = [1.0, 0.5, 2.0]
    >>> weights = [0.3, 0.4, 0.3]
    >>> get_mixed_gaussian_ppf_values(0.5, means, stds, weights)
    1.889796824693636  # a float corresponding to 50th percentile
    >>> get_mixed_gaussian_ppf_values([0.25, 0.5, 0.75], means, stds, weights)
    array([t1, t2, t3])  # an array of threshold values for the 25th, 50th, and 75th percentiles
    array([0.74900243 1.88979682 2.7361815 ])
    """
    means, stds, weights = prepare_means_stds_weights(means, stds, weights, 
already_normalized_weights=already_normalized_weights)

    def _mixture_ppf_scalar(prob_scalar: float) -> float:
        if not (0 < prob_scalar < 1):
            raise ValueError("Each probability must be between 0 and 1 (exclusive).")
        
        def _mix_cdf(t: float) -> float:
            # Compute the mixture CDF at t as a weighted sum of component CDFs.
            return np.dot(weights, norm.cdf(t, means, stds)) - prob_scalar

        # Define a heuristic search interval that should capture nearly all the probability mass.
        lower_bound = np.min(means) - 1e5 * np.max(stds)
        upper_bound = np.max(means) + 1e5 * np.max(stds)

        # Find the root of _mix_cdf(t) = 0 using Brent's method.
        t_val = brentq(_mix_cdf, lower_bound, upper_bound, xtol=tol)
        return t_val

    # Convert prob to a NumPy array to determine if the input is scalar or array-like.
    p_arr = np.asarray(prob, dtype=float)
    if p_arr.ndim == 0:
        return _mixture_ppf_scalar(float(p_arr))
    else:
        # Vectorize the scalar inverse CDF for array-like inputs.
        vec_mixture_ppf = np.vectorize(_mixture_ppf_scalar)
        return vec_mixture_ppf(p_arr)


def remove_outliers(arr, num_std_devs=2):
    if len(arr) < 2:
        return arr
    mean = np.mean(arr)
    std_dev = np.std(arr)
    return arr[np.abs(arr - mean) <= num_std_devs * std_dev]


def arccos2(vector, value):
    if vector[0] == 0:
        if vector[1] >= 0:
            return 0
        else:
            return np.pi
    return -np.sign(vector[0]) * np.arccos(value)


def arctan_0(vector):
    x = float(vector[0])
    y = float(vector[1])
    if x == 0:
        return 0
    if x >= 0:
        return np.arctan(y / x) - (np.pi / 2.0)
    else:
        return np.arctan(y / x) + (np.pi / 2.0)


def rotate_about_z_axis(vector, theta):
    return np.dot(np.array([
[np.cos(theta), -np.sin(theta), 0],
[np.sin(theta), np.cos(theta), 0],
[0, 0, 1]
]), vector)


def rotate_about_x_axis(vector, theta):
    return np.dot(np.array([
[1, 0, 0],
[0, np.cos(theta), -np.sin(theta)],
[0, np.sin(theta), np.cos(theta)]        
]), vector)


def rotate_point_about_axis_by_angle(point, axis, theta):
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    uxx = ux**2
    uyy = uy**2
    uzz = uz**2
    uxy = ux * uy
    uxz = ux * uz
    uyz = uy * uz
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1.0 - c
    rotation_matrix = np.array([
[t * uxx + c, t * uxy - s * uz, t * uxz + s * uy],
[t * uxy + s * uz, t * uyy + c, t * uyz - s * ux],
[t * uxz - s * uy, t * uyz + s * ux, t * uzz + c]
])
    return np.dot(rotation_matrix, point)


def rotate_about_z_then_x(vector, gamma, phi):
    gamma_rotated_vector = rotate_about_z_axis(vector, gamma)
    rotated_x_axis = np.array([np.cos(gamma), np.sin(gamma), 0.0])
    return rotate_point_about_axis_by_angle(gamma_rotated_vector, rotated_x_axis, phi)


def calculate_continuous_differences_2d(angle_arr):
    '''
    Parameters
    ----------
    angle_arr: 2D np.array shape (n,2)
        each column is a list of angles. Each row is a single observation

    Returns
    -------
    dangle_arr: 2D np.array shape (n,2)
        differences from one angle to the next and handles edge cases where
        angles go over the 2 * pi boundary and appear like a large difference
        but actually is a small difference.
    '''
    if len(angle_arr) < 2:
        return np.array([0,0])
    # Initialize an array to hold the differences
    dangle_arr = np.zeros_like(angle_arr)

    # Calculate the differences for each angle
    for i in range(1, angle_arr.shape[0]):
        for j in range(angle_arr.shape[1]):
            # Calculate the raw difference
            diff = angle_arr[i, j] - angle_arr[i - 1, j]

            # Adjust the difference to be within -pi to pi radians
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi

            # Store the adjusted difference
            dangle_arr[i, j] = diff

    return dangle_arr[1:]


def find_intersection(plane_normal, plane_point, ray_direction, ray_point):
    '''
    This function finds the intersection of a line and a plane.
    
    Args:
    plane_normal: A normal vector of the plane.
    plane_point: A point on the plane.
    ray_direction: A direction vector of the line.
    ray_point: A point on the line.
    
    Returns:
    Psi: The intersection point of the line and the plane, or None.
    '''
    epsilon=1e-6
    ndotu = np.dot(plane_normal, ray_direction)
    if abs(ndotu) < epsilon:
        # no intersection or line is within plane
        return None
    w = ray_point - plane_point
    si = -np.dot(plane_normal, w) / ndotu
    Psi = w + si * ray_direction + plane_point #intersections
    return Psi


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

    
def poly1d_fit(x, y, degree, 
        get_r_sqrd=True, get_local_maximums=False, 
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

            "r_squared": float
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


def update_mean(prev_mean, prev_n, new_data):
    '''
    Update the mean without having the previous data, only the previous
    mean and number of data points used to calculate the previous mean.

    Parameters
    ----------
    prev_mean: float
        The mean of data points before new_data is accounted for.
        
    prev_n: int
        The number of data points used to calculate the prev_mean.
        
    new_data: scalar or 1D iterable
        New data point(s) to factor into the mean.

    Returns
    -------
    updated_mean: float
        Mean now that new_data have been included.
        
    Method
    ------
    The mean is the sum of the data points divided by the number of data points.
    This means that to update a mean given new data, one can isolate the previous
    sum by multiplying by the previous number of data points. Then, add the new
    data points to that sum, and divide that by len(new_data) + prev_n
    '''
    if not hasattr(new_data, '__iter__'):
        return (prev_mean * prev_n + new_data) / (prev_n + 1)
    return (prev_mean * prev_n + sum(new_data)) / (prev_n + len(new_data))


def update_mean_std_n(prev_mean, prev_std, prev_n, 
new_data, ddof=0, num_decimal_places=7):
    '''
    Update the standard deviation without having the previous data, only the previous
    standard deviation, mean, and number of data points used to calculate the 
    previous mean and standard deviation.

    Parameters
    ----------
    prev_mean: float
        The mean of data points before new_data is accounted for.
    
    prev_std: float
        The standard deviation of data points before new_data is accounted for.
        
    prev_n: int
        The number of data points used to calculate the prev_mean.
        
    new_data: scalar or 1D iterable
        New data point(s) to factor into the mean.
        
    ddof: int
        Means Delta Degrees of Freedom. The divisor used in calculations is 
        (N - ddof), where N represents the number of elements. When ddof is 0,
        you are using the population standard deviation. When ddof is 1, you 
        are using the sample standard deviation.
        
    num_decimal_places: int
        Number of decimal places to use for atol when determining whether the radical is 0.

    Returns
    -------
    updated_mean, updated_std, updated_n: float
        The standard deviation now that new_data have been included.
        
    Method
    ------
    The sample standard deviation is sqrt((1/(n-1)) * (-n * mu**2 + Sum_i[xi**2])
    So, you can square prev_std, multiply by (n-1), and add (n*mu**2) to get 
    the previous sum of the squared data points (where mu is prev_mean and n is prev_n).
    Sum the squares of each new data point in new_data, and add that to this previous
    sum. Then, update the mean and update n. Then, calculate the updated standard
    deviation by plugging the updated mean, n, and sum of squared data points into
    the formula.
    
    If the number of total data points is 0 or 1, just return 0.
    
    Notes
    -----
    The Cauchy-Schwarz inequality states sum(ai * bi) <= sum(ai**2)sum(bi**2).
    When ai=1 and bi=xi we get sum(xi)**2 <= n * sum(xi**2) which means that the
    radical for the standard deviation equation is always positive because,
    looking at the numerator since updated_n - ddof is >= 1 otherwise we return 0 we have
    (sum(xi**2) - n * mean**2) >= 0
    (sum(xi**2) - (1/n) * sum(xi)**2) >= 0
    (n * sum(xi**2) >= sum(xi)**2)
    '''
    if ddof > 1:
        raise ValueError('ddof must be 0 or 1')
    if np.any(np.isnan([prev_n, prev_mean, prev_std, new_data, ddof, num_decimal_places])):
        print('prev_n', prev_n, 'prev_mean', prev_mean, 'prev_std', prev_std)
        print('new_data', new_data, 'ddof', ddof, 'num_decimal_places', num_decimal_places)
        raise ValueError('Nan in input arguments.')
    if not hasattr(new_data, '__iter__'):
        new_data = [new_data]
    if prev_n == 0:
        if len(new_data) == 0:
            return 0, 0, 0
    elif len(new_data) == 0:
        return prev_mean, prev_std, prev_n
    updated_mean = update_mean(prev_mean, prev_n, new_data)
    updated_n = prev_n + len(new_data)
    if updated_n < 2:
        return updated_mean, 0.0, updated_n
    prev_sum_of_squares = (prev_std**2) * (prev_n - ddof) + prev_n * (prev_mean**2)
    updated_sum_of_squares = prev_sum_of_squares + sum([x**2 for x in new_data])
    radical = (updated_sum_of_squares - updated_n * (updated_mean**2)) / (updated_n - ddof)
    if np.isclose(0, radical, atol=10**(-num_decimal_places)):
        radical = 0.0
    if radical < 0 or np.any(np.isnan([prev_sum_of_squares, updated_sum_of_squares, updated_n, updated_mean])):
        print('prev_sum_of_squares', prev_sum_of_squares,'updated_sum_of_squares', updated_sum_of_squares)
        print('updated_n', updated_n, 'updated_mean', updated_mean)
        print('new_data', new_data)
        print('Radical:', radical)
        raise ValueError('radical is < 0 or nan found')
    updated_std = np.sqrt(radical)
    return updated_mean, updated_std, updated_n
    

def update_mean_with_stats(prev_mean, prev_n, new_mean, new_n):
    '''
    Update the mean without having the previous data, only the previous
    mean and number of data points used to calculate the previous mean.

    Parameters
    ----------
    prev_mean: float
        The mean of data points before new_data is accounted for.
        
    prev_n: int
        The number of data points used to calculate the prev_mean.
        
    new_mean: float
        New mean to factor into the mean.
        
    new_n: int
        The number of data points used to calculate the new_mean.

    Returns
    -------
    updated_mean: float
        Mean now that new_mean and new_n have been included.
        
    Method
    ------
    The mean is the sum of the data points divided by the number of data points.
    This means that to update a mean given new data, one can isolate the previous
    sum by multiplying by the previous number of data points. Then, add the new
    data points to that sum, and divide that by len(new_data) + prev_n
    '''
    return (prev_mean * prev_n + new_mean * new_n) / (prev_n + new_n)


def update_mean_std_n_with_stats(prev_mean, prev_std, prev_n,
new_mean, new_std, new_n, ddof=0, num_decimal_places=7):
    '''
    Update the standard deviation without having the previous data, only the previous
    standard deviation, mean, and number of data points used to calculate the 
    previous mean and standard deviation.

    Parameters
    ----------
    prev_mean: float
        The mean of data points before new stats are accounted for.
    
    prev_std: float
        The standard deviation of data points before new stats are accounted for.
        
    prev_n: int
        The number of data points used to calculate the prev_mean.
        
    new_mean: float
        New mean to factor into the mean.
        
    new_std: float
        The standard deviation of the new data set to include.
        
    new_n: int
        The number of data points used to calculate the new_mean.
        
    ddof: int
        Means Delta Degrees of Freedom. The divisor used in calculations is 
        (N - ddof), where N represents the number of elements. When ddof is 0,
        you are using the population standard deviation. When ddof is 1, you 
        are using the sample standard deviation.
        
    num_decimal_places: int
        Number of decimal places to use for atol when determining whether the radical is 0.

    Returns
    -------
    updated_mean, updated_std, updated_n: float
        The standard deviation now that new_data have been included.
        
    Method
    ------
    The sample standard deviation is sqrt((1/(n-1)) * (-n * mu**2 + Sum_i[xi**2])
    So, you can square prev_std, multiply by (n-1), and add (n*mu**2) to get 
    the previous sum of the squared data points (where mu is prev_mean and n is prev_n).
    Sum the squares of each new data point in new_data, and add that to this previous
    sum. Then, update the mean and update n. Then, calculate the updated standard
    deviation by plugging the updated mean, n, and sum of squared data points into
    the formula.
    
    If the number of total data points is 0 or 1, just return 0.
    '''
    if ddof > 1:
        raise ValueError('ddof must be 0 or 1')
    if np.any(np.isnan([prev_mean, prev_std, prev_n, new_mean, new_std, new_n, ddof, num_decimal_places])):
        print('prev_n', prev_n, 'prev_mean', prev_mean, 'prev_std', prev_std)
        print('new_n', new_n, 'new_mean', new_mean, 'new_std', new_std)
        raise ValueError('Nan in input arguments.')
    if prev_n == 0:
        if new_n == 0:
            return 0, 0, 0
        return new_mean, new_std, new_n
    elif new_n == 0:
        return prev_mean, prev_std, prev_n
    updated_mean = update_mean_with_stats(prev_mean, prev_n, new_mean, new_n)
    updated_n = prev_n + new_n
    if updated_n < 2:
        return updated_mean, 0.0, updated_n
    prev_sum_of_squares = (prev_std**2) * (prev_n - ddof) + prev_n * (prev_mean**2)
    new_sum_of_squares = (new_std**2) * (new_n - ddof) + new_n * (new_mean**2)
    updated_sum_of_squares = prev_sum_of_squares + new_sum_of_squares
    radical = (updated_sum_of_squares - updated_n * (updated_mean**2)) / (updated_n - ddof)
    if np.isclose(0, radical, atol=10**(-num_decimal_places)):
        radical = 0.0
    if radical < 0 or np.any(np.isnan([prev_sum_of_squares,new_sum_of_squares,updated_sum_of_squares,updated_n,updated_mean,ddof])):
        print('prev_n', prev_n, 'prev_mean', prev_mean, 'prev_std', prev_std, 'prev_sum_of_squares', prev_sum_of_squares)
        print('new_n', new_n, 'new_mean', new_mean, 'new_std', new_std, 'new_sum_of_squares', new_sum_of_squares)
        print('updated_sum_of_squares', updated_sum_of_squares, 'updated_n', updated_n, 'updated_mean', updated_mean, 'ddof', ddof)
        print('Radical:', radical)
        raise ValueError('radical is < 0 or nan found')
    updated_std = np.sqrt(radical)
    return updated_mean, updated_std, updated_n


if __name__ == '__main__':
    print(get_mixed_gaussian_ppf_values([0.25, 0.5, 0.75], means=[0, 2, 4], stds=[1, 0.5, 2], weights=[0.3, 0.4, 0.3], already_normalized_weights=True))