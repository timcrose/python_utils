import os
import numpy as np
from numbers import Number
from inspect import getframeinfo, stack
from python_utils import math_utils

def print_iterable(iterable, idx, arg_name):
    if isinstance(idx, slice) or isinstance(idx, int):
        print(arg_name + '[' + str(idx) + ']', iterable[idx])
    elif check_type_nonstr_iterable(idx):
        results = [iterable[i] for i in idx]
        print(arg_name + ' at indices', idx, 'is', results)


def print_np_stat(arr, arg_name, stat):
    print(stat.__name__ + '(' + arg_name + ')', stat(arr))
    for i in range(len(arr.shape)):
        print(stat.__name__ + '(' + arg_name + ', axis=' + str(i), \
                stat(arr, axis=i))
    print(stat.__name__ + '(' + arg_name + ')', stat(arr))
    for i in range(len(arr.shape)):
        print(stat.__name__ + '(' + arg_name + ', axis=' + str(i), \
                stat(arr, axis=i))

def print_arr_stats(arr, arg_name):
    for stat in [np.min, np.max, np.mean, np.median, np.std]:
        print_np_stat(stat, arr, arg_name)

def trose_logging_decorator(wrapped_func):
    def wrapper(*args, **kwargs):
        # If 'trose_log_dct' not passed to wrapped_func, then dont do any
        # logging.
        if 'trose_log_dct' not in kwargs:
            # Simply call the wrapped function and return its result.
            result = wrapped_func(*args, **kwargs)
            return result
        # Set dct variables such that they take up less space in the code
        # editor.
        verbosity_level = kwargs['trose_log_dct']['verbosity_level']
        str_idx = kwargs['trose_log_dct']['str_idx']
        list_idx = kwargs['trose_log_dct']['list_idx']
        arr_row_idx = kwargs['trose_log_dct']['arr_row_idx']
        arr_col_idx = kwargs['trose_log_dct']['arr_col_idx']
        if verbosity_level > 0:
            print('log settings:', kwargs['trose_log_dct'])
            caller = getframeinfo(stack()[1][0])
            print(wrapped_func.__name__ + ' on line ' + \
                    str(caller.lineno) + ' in ' + caller.filename)
        # Iterate through the unnamed arguments to the wrapped function and 
        # print various attributes depending on the verbosity level.
        for i,arg in enumerate(args):
            # Variables in args are unnamed but must have a particular order.
            # Therefore, we will give it a name which is indexed.
            arg_name = 'arg' + str(i)
            if verbosity_level > 0:
                print('type(' + arg_name + ')', type(arg))
                if hasattr(arg, '__len__'):
                    print('len(' + arg_name + ')', len(arg))
                if isinstance(arg, str):
                    print_iterable(arg, str_idx, arg_name)
                elif isinstance(arg, list):
                    print_iterable(arg, list_idx, arg_name)
                elif isinstance(arg, np.ndarray):
                    print(arg_name + '.shape', arg.shape)
                    print(arg_name + '[' + str(arr_row_idx) + ', ' + \
                            str(arr_col_idx) + ']', \
                            arg[arr_row_idx, arr_col_idx])

            if verbosity_level > 1:
                if isinstance(arg, np.ndarray):
                    print_arr_stats(arg, arg_name)
            if verbosity_level > 2:
                print('dir(' + arg_name + ')', dir(arg))

        for kwarg in kwargs:

        del kwargs['trose_log_dct']
        try:
            result = wrapped_func(*args, **kwargs)
        except Exception as e:
            raise(e)
        return result
    return wrapper


def handle_error(e=None, err_message='Alert', fail_gracefully=False, 
                verbose=False):

    '''
    e: Exception or a subclass of Exception or None
        When in a try/except block, saying 'except Exception as e' yields the
        e you should pass to handle_error. If None is passed, e becomes
        Exception.

    err_message: any type
        Message that will be fed to a print statement or Exception statement
        depending on fail_gracefully

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages
    '''
    if not isinstance(fail_gracefully, bool) or not isinstance(verbose, bool):
        raise TypeError('Both fail_gracefully and verbose must be of type '+\
                'bool. type(fail_gracefully):', type(fail_gracefully), \
                'type(verbose):', type(verbose), 'fail_gracefully:', \
                fail_gracefully, 'verbose:', verbose)

    if fail_gracefully:
        if verbose:
            print(err_message)
    else:
        if e is None:
            raise Exception(err_message)
        else:
            if not issubclass(e, BaseException):
                raise Exception('e', e, \
                        'is not a valid subclass of BaseException')

            if verbose:
                print(err_message)
            raise e(err_message)


def try_assign(func, *input_params, fail_value=None, \
        err_message='Alert; could call func with given input parameters', \
        fail_gracefully=False, verbose=False):

    '''
    func: callable function
        This function will be called with input_params as its parameters.

    input_params: parameter set
        These are the parameters passed to func. Any number of parameters
        will work if you pass in *input_params.

    fail_value: any type
        Value to return if the call to func fails.

    err_message: any type
        Message that will be fed to a print statement or Exception statement
        depending on fail_gracefully

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return:
    func_value: unknown type
        Return value of func or fail_value if call to func fails.

    Purpose: This function provides a simpler interface to error handing
        capabilities by allowing you to use try and except blocks without
        the clutter and assign a variable(s) to the output
        value(s) of a function, func, only if the call to that function
        does not raise an error. And, if it does raise an error, then
        the error is handled according to fail_gracefully and prints 
        descriptive messages according to err_message and verbose, and
        can return a specified value if the call to the function fails.
    '''
    try:
        return func(*input_params)
    except Exception as e:
        handle_error(e=e, err_message=err_message,\
                    fail_gracefully=fail_gracefully, verbose=verbose)

        return fail_value


def check_type_number(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The variable you desire to have is a numeric type.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_number
        is_number: bool
            True: var is a numeric quantity
            False: var is not a numeric quantity

    Purpose: Check to make sure var is a numeric quantity. Throw an error if
        it is not a number and fail_gracefully is False.
    '''
    # Test for bool type because isinstance(var, Number) returns True for 
    # bool.
    if isinstance(var, Number) and not isinstance(var, bool):
        is_number = True
        return is_number
    else:
        err_message = 'Alert; wanted number but got bool for var =', var
        handle_error(e=TypeError, err_message=err_message, \
                fail_gracefully=fail_gracefully, verbose=verbose)

        is_number = False
        return is_number


def check_type_file(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The path of the file you desire to exist.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_file
        is_file: bool
            True: var is a path to an existent file.
            False: var is not a path to an existent file.

    Purpose: Check to make sure var is an existent file. Throw an error if
        it is not an existent file and fail_gracefully is False. An existent
        directory path does not count. The check will succeed if
        os.path.isfile(var) is True.
    '''
    # We desire var to be a valid and existent file (but not dir) path.
    err_message = 'Alert; wanted var to be existent file but got var =', var
    is_file = try_assign(os.path.isfile, var, fail_value=False,\
            err_message=err_message, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    return is_file


def check_type_dir(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The path of the directory you desire to exist.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_dir
        is_dir: bool
            True: var is a path to an existent directory.
            False: var is not a path to an existent directory.

    Purpose: Check to make sure var is an existent directory. Throw an error if
        it is not an existent directory and fail_gracefully is False. An 
        existent file path does not count. i.e. the check will succeed if
        os.path.isdir(var) is True.
    '''
    # We desire var to be a valid and existent directory (but not file) path.
    err_message = 'Alert; wanted var to be existent directory but got var =',\
            var

    is_dir = try_assign(os.path.isdir, var, fail_value=False,\
            err_message=err_message, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    return is_dir


def check_type_path(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The path you desire to exist.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_dir
        is_dir: bool
            True: var is an existent path.
            False: var is not an existent path.

    Purpose: Check to make sure var is an existent path. Throw an error if
        it is not an existent path and fail_gracefully is False. i.e. the
        check will succeed if os.path.exists(var) is True.
    '''
    # We desire var to be a valid and existent path.
    err_message = 'Alert; wanted var to be existent path but got var =',\
            var

    is_path = try_assign(os.path.exists, var, fail_value=False,\
            err_message=err_message, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    return is_path


def check_type_nonstr_iterable(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The variable you desire to be a non-string iterable. An iterable is
        a variable with an __iter__ attribute. e.g. list, tuple, np.ndarray. A
        str also has an __iter__ attribute, so we need a further check to make
        sure the variable is not of type str.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_nonstr_iterable
        is_nonstr_iterable: bool
            True: var is a non-string iterable.
            False: var is either a string, or is not an iterable.

    Purpose: Check to make sure var is a non-string iterable. An iterable is
        a variable with an __iter__ attribute. e.g. list, tuple, np.ndarray. A
        str also has an __iter__ attribute, so we need a further check to make
        sure the variable is not of type str. Throw an error if it is not a
        non-string iterable and fail_gracefully is False. i.e. the check will
        succeed if hasattr(var, '__iter__') and not isinstance(var, str).
    '''
    if isinstance(var, str) or not hasattr(var, '__iter__'):
        err_message = 'Alert; var was not a non-string iterable but '+\
                'should be. var =', var, 'type(var) =', type(var)

        handle_error(e=TypeError, err_message=err_message, \
                fail_gracefully=fail_gracefully, verbose=verbose)

        is_nonstr_iterable = False
    else:
        is_nonstr_iterable = True
    return is_nonstr_iterable


def check_type_1D_vec(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The variable you desire to have is a 1-dimensional vector. This could
        be a 1D list, row-vector, column-vector, tuple, set, or any other 
        non-string iterable that can be made into a 1D vector by applying
        np.array() to it.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_1D_vec
        is_1D_vec: bool
            True: var is a 1D vector
            False: var is not a 1D vector

    Purpose: Check to make sure var is a 1D vector. This could be a 1D list,
        row-vector, column-vector, tuple, set, or any other non-string
        iterable that can be made into a 1D vector by applying np.array() to
        it. Throw an error if it is not a 1D vector and fail_gracefully is
        False.
    '''
    # We desire var to be a 1-dimensional vector.
    if not check_type_nonstr_iterable(var, fail_gracefully=fail_gracefully,\
            verbose=verbose):

        is_1D_vec = False
    elif not isinstance(var, np.ndarray):
        tmp_var = np.array(var)
        if len(tmp_var) != tmp_var.size:
            is_1D_vec = False
    elif len(var) != var.size:
        is_1D_vec = False
    else:
        is_1D_vec = True
    if not is_1D_vec:
        err_message = 'Alert; var was not a 1D vector but '+\
                'should be. var =', var, 'type(var) =', type(var)

        handle_error(e=TypeError, err_message=err_message, \
                fail_gracefully=fail_gracefully, verbose=verbose)

    return is_1D_vec


def check_type_1D_arr(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The variable you desire to have is a 1-dimensional numpy array. Both
        arrays of shape (n,) and (n,1) are considered 1-dimensional arrays.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_1D_arr
        is_1D_arr: bool
            True: var is a 1D array
            False: var is not a 1D array

    Purpose: Check to make sure var is a 1D array. Both arrays of shape (n,)
        and (n,1) are considered 1-dimensional arrays. Throw an error if it is
        not a 1D array and fail_gracefully is False.
    '''
    # We desire var to be a 1-dimensional numpy array.
    if not isinstance(var, np.ndarray):
        is_1D_arr = False
    elif len(var) != var.size:
        is_1D_arr = False
    else:
        is_1D_arr = True
    if not is_1D_arr:
        err_message = 'Alert; var was not a 1D numpy array but '+\
                'should be. var =', var, 'type(var) =', type(var)

        handle_error(e=TypeError, err_message=err_message, \
                fail_gracefully=fail_gracefully, verbose=verbose)
        
    return is_1D_arr


def check_type_1D_numeric_arr(var, fail_gracefully=False, verbose=False):
    '''
    var: ?
        The variable you desire to have is a 1-dimensional numpy array
        containing only numeric entries of the same dtype.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_1D_numeric_arr
        is_1D_numeric_arr: bool
            True: var is a 1D array that only contains numeric elements
            False: var is not a 1D array with only numeric elements

    Purpose: Check to make sure var is a 1D numpy array containing only
        numeric entries of the same dtype. Throw an error if it is not a 1D
        numpy array containing only numeric entries of the same dtype and
        fail_gracefully is False.
    '''
    # We desire var to be a 1-dimensional numpy array containing only numeric
    # entries of the same dtype.
    if not check_type_1D_arr(var, fail_gracefully=True, verbose=\
        verbose):

        is_1D_numeric_arr = False
    else:
        test_arr = np.array([1], dtype=var.dtype)
        is_1D_numeric_arr = check_type_number(test_arr[0], fail_gracefully=\
                True, verbose=verbose)

    if not is_1D_numeric_arr:
        err_message = 'Alert; var was not a 1D array containing only numeric'+
                'elements, but should be. var =', var, 'type(var) =', type(var)

        handle_error(e=TypeError, err_message=err_message, \
                fail_gracefully=fail_gracefully, verbose=verbose)

    return is_1D_numeric_arr


def check_type_nonempty_1D_numeric_arr(var, fail_gracefully=False, verbose=\
        False):

    '''
    var: ?
        The variable you desire to have is a non-empty 1-dimensional numpy array
        containing only numeric entries of the same dtype.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_nonempty_1D_numeric_arr
        is_nonempty_1D_numeric_arr: bool
            True: var is a non-empty 1D array
            False: var is not a non-emptry 1D array

    Purpose: Check to make sure var is a non-empty 1D numpy array containing
        only numeric entries of the same dtype. Throw an error if it is not a
        non-empty 1D numpy array containing only numeric entries of the same
        dtype and fail_gracefully is False.
    '''
    # We desire var to be a non-empty 1-dimensional numpy array containing
    # only numeric entries of the same dtype.
    if not check_type_1D_numeric_arr(var, fail_gracefully=True, verbose=\
            verbose):

        is_nonempty_1D_numeric_arr = False
    elif len(var) > 0:
        is_nonempty_1D_numeric_arr = True
    else:
        is_nonempty_1D_numeric_arr = False

    if not is_nonempty_1D_numeric_arr:
        err_message = 'Alert; var was not a non-empty 1D array containing ' +\
            'only numeric elements but should be. var =', var, 'type(var) =',\
            type(var)

        handle_error(e=TypeError, err_message=err_message, \
                fail_gracefully=fail_gracefully, verbose=verbose)

    return is_nonempty_1D_numeric_arr


def check_var_type(var, desired_type, fail_gracefully=False, \
        verbose=False):

    '''
    var: any type
        A variable that you want to make sure has a desired type.

    desired_type: any type
        The type that you expect var to have. Allowed values for desired_type
        are all python types (returned by type()), and also 'path', 'file', 
        'dir', 'number'.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return:
    passed_checks: bool
        True: The variable is of the desired type
        False: The variable is not of the desired type

    Purpose: Check that a variable is of the desired type. This can include
        any python type or
        'file': Check for existent file path. Success if os.path.isfile(var)
            returns True.
        'path': Check for existent file path. Success if os.path.exists(var)
            returns True.
        'dir': Check for existent directory path. Success if os.path.isdir(var)
            returns True.
        'number': Check for a number of any type except str. Success if
            isinstance(var, Number) and not isinstance(var, bool) returns True.
        'nonstr_iterable': Check for an iterable variable that is not a string.
            Success if hasattr(var, '__iter__') and not isinstance(var, str).
    '''
    # Check types
    if not isinstance(fail_gracefully, bool) or not isinstance(verbose, bool):
        err_message = 'Either fail_gracefully or verbose were not of type '+\
                'bool. type(fail_gracefully):', type(fail_gracefully), \
                'type(verbose):', type(verbose), 'fail_gracefully:', \
                fail_gracefully, 'verbose:', verbose

        handle_error(e=TypeError, err_message=err_message, fail_gracefully=\
                False, verbose=verbose)

    if hasattr(desired_type, '__iter__') and not isinstance(desired_type, str):
        err_message = 'Error; desired_type must not be iterable if it is not'+\
                ' a string. desired_type:', desired_type, 'type(desired_type'+\
                ')', type(desired_type), 'Use the check_any_acceptable_type '+\
                'function if you want var to be any type from a provided list.'

        handle_error(e=TypeError, err_message=err_message, fail_gracefully=\
                False, verbose=verbose)

    if desired_type == 'path':
        # We desire var to be a valid and existent path.
        return check_type_path(var, fail_gracefully=\
                fail_gracefully, verbose=verbose)

    elif desired_type == 'dir':
        # We desire var to be a valid and existent directory path.
        return check_type_dir(var, fail_gracefully=fail_gracefully,\
                verbose=verbose)

    elif desired_type == 'file':
        # We desire var to be a valid and existent file (but not dir) path.
        return check_type_file(var, fail_gracefully=\
                fail_gracefully, verbose=verbose)

    elif desired_type == 'number':
        # We desire var to be any valid numeric value.
        return check_type_number(var, fail_gracefully=\
                fail_gracefully, verbose=verbose)

    elif desired_type == 'nonstr_iterable':
        # We desire var to be a non-string iterable.
        return check_type_nonstr_iterable(var, fail_gracefully=\
                fail_gracefully, verbose=verbose)

    elif desired_type == '1D_vec':
        # We desire var to be a 1-dimensional vector.
        return check_type_1D_vec(var, fail_gracefully=fail_gracefully,\
            verbose=verbose)
        
    elif desired_type == '1D_arr':
        # We desire var to be a 1-dimensional numpy array.
        return check_type_1D_arr(var, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    elif desired_type == '1D_numeric_arr':
        # We desire var to be a 1-dimensional numpy array that contains only
        # numeric elements.
        return check_type_1D_numeric_arr(var, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    elif desired_type == 'nonempty_1D_numeric_arr':
        # We desire var to be a non-empty 1-dimensional numpy array that
        # contains only numeric elements.
        return check_type_nonempty_1D_numeric_arr(var, fail_gracefully=\
                fail_gracefully, verbose=verbose)

    if isinstance(desired_type, str):
        err_message = 'Error; Unsupported desired_type:', desired_type
        handle_error(e=TypeError, err_message=err_message, fail_gracefully=\
            False, verbose=verbose)

    # We desire var to be of desired_type type.
    if isinstance(var, desired_type):
        return True
    else:
        err_message = 'Alert; wanted var to be of type', desired_type, \
                        'but got', type(var), 'for var =', var

        handle_error(e=TypeError, err_message=err_message, \
                    fail_gracefully=fail_gracefully, verbose=verbose)

        return False


def check_var_type_lst(var_type_lst, fail_gracefully=False,\
        verbose=False):

    '''
    var_type_lst: list of (list of length 2)
        Each element of var_type_lst must be a list of length 2 with format
        [var, type]. e.g. [foo, int]. Allowed Values for type are all python
        types (returned by type()), and also 'path', 'file', 'dir', 'number',
        'nonstr_iterable'.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return:
    passed_checks: bool
        True: all variables are of the desired type
        False: not all variables are of the desired type

    Purpose: Check that a list of variables are of the desired type which can
        include python types or 'path', 'file', 'dir', 'number', 
        'nonstr_iterable'.
    '''
    check_var_type(var_type_lst, list, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    # Call check_var_type for every var desired_type pair.
    checks = [check_var_type(var, desired_type, \
            fail_gracefully=fail_gracefully, verbose=verbose) 
            for var, desired_type in var_type_lst]

    # See if all values in checks are True which means all vars are of the
    # desired type.
    passed_checks = len(checks) == checks.count(True)
    return passed_checks


def check_any_acceptable_type(var, acceptable_type_lst,\
            fail_gracefully=False, verbose=False):
    
    '''
    var: ?
        Variable whose type is desired to match one of the types in
        acceptable_type_lst.
        
    acceptable_type_lst: flattened list
        This is the list of types that var can have to be considered an
        appropriate type for var. An error will be handled if the type of
        var is not in acceptable_type_lst. The kinds of entries
        in acceptable_type_lst must be any python type or 'file', 'path',
        'dir', 'number', 'nonstr_iterable'.
        
    fail_gracefully: bool
        True: Only print an error instead of raising an error if var is not
            an instance of any type in acceptable_type_lst.
        False: Raise an error if var is not an instance of any type in
            acceptable_type_lst.

    verbose: bool
        True: Make sure err_message is printed if var is not an instance of
            any type in acceptable_type_lst.
        False: Do not print err_message if fail_gracefully is True and var is
            not an instance of any type in acceptable_type_lst.
            
    Return: None
    
    Purpose: Check to make sure the variable var has a type of the ones listed
        in acceptable_type_lst. Handle an error otherwise. The kinds of entries
        in acceptable_type_lst must be any python type or 'file', 'path',
        'dir', 'number', 'nonstr_iterable'.
    '''
    # Check type of input variables.
    check_var_type(acceptable_type_lst, 'nonstr_iterable', \
            fail_gracefully=False, verbose=verbose)
    
    for desired_type in acceptable_type_lst:
        if check_var_type(var, desired_type, \
                fail_gracefully=fail_gracefully, verbose=verbose):

            return

    err_message = 'The type of var was not in the list of desired types. var'\
            + ':', var, 'type(var):', type(var), 'desired_types:', \
            desired_types

    handle_error(e=TypeError, err_message=err_message, fail_gracefully=\
        fail_gracefully, verbose=verbose)


def assert_gracefully(condition, err_message='assertion failed',\
            fail_gracefully=False, verbose=False):
    
    '''
    condition: bool
        True: Return. The assertion succeeded.
        False: The assertion failed. Raise an error if fail_gracefully is
            False.
        
    err_message: str or tuple of str
        Message to be printed out when handling the error if condition is
        False.
        
    fail_gracefully: bool
        True: Only print an error instead of raising an error if condition is
            False.
        False: Raise an error if condition is False.

    verbose: bool
        True: Make sure err_message is printed if condition is False.
        False: Do not print err_message if condition is False and
            fail_gracefully is True. 
            
    Return: bool
        True: assertion passed
        False: assertion failed
    
    Purpose: Instead of the built-in assert function, use assert_gracefully to
        allow writing custom and descriptive error messages. This function also
        allows continuing program flow if the assertion fails by setting
        fail_gracefully to True. The err_message is printed if the assertion
        failed regardless if fail_gracefully is True if verbose is True.
    '''
    check_var_type(condition, bool, fail_gracefully=False, verbose=\
            verbose)
            
    # If condition is True, then the assertion passes and nothing left to do.
    if condition:
        return True
    
    # The assertion has failed so we will throw an error.
    handle_error(e=AssertionError, err_message=err_message,\
            fail_gracefully=fail_gracefully, verbose=verbose)
    
    return False


def numeric_var_in_bounds(var, lower_bound, upper_bound, le=True, ge=True,\
        num_decimal_places=7, fail_gracefully=False, verbose=False):

    '''
    var: number
        We are querying if this numeric value is within the bounds.

    lower_bound: number
        Lower bound

    upper_bound: number
        Upper bound

    le: bool
        True: test whether var <= upper_bound
        False: test whether var < upper_bound

    ge: bool
        True: test whether var >= lower_bound
        False: test whether var > lower_bound

    num_decimal_places: int
        Number of decimal places to round number to. This is done to 
        prevent numerical error from interfering with the checks.

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return:
    var_in_bounds: bool
        True: var is within the bounds
        False: var is not within the bounds

    Purpose: Test whether a numeric variable var is within the bounds
        specified.
    '''

    # Check input types
    var_type_lst = [[var, 'number'], [lower_bound, 'number'], [upper_bound, 'number'],
                    [le, bool], [ge, bool], [num_decimal_places, int]]

    if not check_var_type_lst(var_type_lst, fail_gracefully=fail_gracefully, \
        verbose=verbose):

        return False

    # Round input values to prevent numerical error from interfering with the checks.
    var = math_utils.round(var, num_decimal_places, leave_int=True)
    lower_bound = math_utils.round(upper_bound, num_decimal_places, leave_int=True)
    upper_bound = math_utils.round(upper_bound, num_decimal_places, leave_int=True)

    if le:
        if ge:
            return var >= lower_bound and var <= upper_bound
        else:
            return var > lower_bound and var <= upper_bound
    else:
        if ge:
            return var >= lower_bound and var < upper_bound
        else:
            return var > lower_bound and var < upper_bound


def check_suitable_two_vec_dot_product(vec0, vec1, fail_gracefully=False,\
        verbose=False):

    '''
    vec0: non-empty 1D numeric array
        The vector to put as the first argument to np.dot(vec0, vec1)

    vec1: non-empty 1D numeric array
        The vector to put as the second argument to np.dot(vec0, vec1)

    fail_gracefully: bool
        True: only print an error instead of raising an error.
        False: Raise an error if one arises.

    verbose: bool
        True: print log messages
        False: don't print log messages

    Return: is_suitable_two_vec_dot_product
        is_suitable_two_vec_dot_product: bool
            True: np.dot(vec0, vec1) should work
            False: np.dot(vec0, vec1) won't work

    Purpose: Check whether np.dot(vec0, vec1) will work or not.
    '''
    # Check inputs types
    check_var_type(vec0, 'nonstr_iterable',\
            fail_gracefully=False, verbose=verbose)
    
    check_var_type(vec1, 'nonstr_iterable',\
            fail_gracefully=False, verbose=verbose)
    
    # Assert that the length of vec0 is equal to the length of vec1
    err_message = 'Alert; len(vec0) != len(vec1) but they should be '+\
            'equal. len(vec0):', len(vec0), 'len(vec1):', len(vec1)
            
    assert_gracefully(len(vec0) == len(vec1), \
            err_message=err_message, fail_gracefully=fail_gracefully, \
            verbose=verbose)
    
    if not isinstance(vec0, np.ndarray):
        tmp_vec0 = np.array(vec0)
        
    if not isinstance(vec1, np.ndarray):
        tmp_vec1 = np.array(vec1)
    
    vec0_is_nonempty_1D_numeric_arr = check_var_type(tmp_vec0,\
            'nonempty_1D_numeric_arr', fail_gracefully=fail_gracefully,\
            verbose=verbose)
    
    vec1_is_nonempty_1D_numeric_arr = check_var_type(tmp_vec1,\
            'nonempty_1D_numeric_arr', fail_gracefully=fail_gracefully,\
            verbose=verbose)

    if vec0_is_nonempty_1D_numeric_arr and vec1_is_nonempty_1D_numeric_arr:
        if len(tmp_vec0.shape) > 2 or len(tmp_vec1.shape) > 2:
            is_suitable_two_vec_dot_product = False

            err_message = 'vec0 and vec1 are not suitable for np.dot(vec0, vec1'+\
                ') as non-empty 1D numeric arrays because they are tensors'+\
                ' with shapes of more than 2 dimensions. vec0.shape: ' +\
                str(tmp_vec0.shape) + ' vec1.shape: ' + str(tmp_vec1.shape)

            handle_error(e=ValueError, err_message=err_message, \
                    fail_gracefully=fail_gracefully, verbose=verbose)

        elif tmp_vec0.shape[-1] == tmp_vec1.shape[0]:
            is_suitable_two_vec_dot_product = True
    else:
        is_suitable_two_vec_dot_product = False

    return is_suitable_two_vec_dot_product