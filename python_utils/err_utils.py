import os
from numbers import Number
from python_utils import math_utils

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


def check_input_var_type(var, desired_type, fail_gracefully=False, \
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
            isinstance(var, Number) returns True.
        'nonstr_iterable': Check for an iterable variable that is not a string.
            Success if hasattr(var, '__iter__') and not isinstance(var, str).
    '''
    if hasattr(desired_type, '__iter__') and not isinstance(desired_type, str):
        err_message = 'Error; desired_type must not be iterable if it is not'+\
                ' a string. desired_type:', desired_type, 'type(desired_type'+\
                ')', type(desired_type), 'Use the check_any_acceptable_type '+\
                'function if you want var to be any type from a provided list.'

        handle_error(e=TypeError, err_message=err_message, fail_gracefully=\
                False, verbose=verbose)

    if desired_type == 'path':
        # We desire var to be any valid and existent file or directory path.
        err_message = 'Alert; wanted var to be existent path but got var =', var
        return try_assign(os.path.exists, var, fail_value=False,\
                err_message=err_message, fail_gracefully=fail_gracefully,\
                verbose=verbose)

    if desired_type == 'dir':
        # We desire var to be any valid and existent directory path.
        err_message = 'Alert; wanted var to be existent dir but got var =', var
        return try_assign(os.path.isdir, var, fail_value=False,\
                err_message=err_message, fail_gracefully=fail_gracefully,\
                verbose=verbose)

    if desired_type == 'file':
        # We desire var to be any valid and existent file (but not dir) path.
        err_message = 'Alert; wanted var to be existent file but got var =', var
        return try_assign(os.path.isfile, var, fail_value=False,\
                err_message=err_message, fail_gracefully=fail_gracefully,\
                verbose=verbose)

    if desired_type == 'number':
        # We desire var to be any valid numeric value.

        # Test for bool type because isinstance(var, Number) returns True for bool.
        if isinstance(var, bool):
            err_message = 'Alert; wanted number but got bool for var =', var
            handle_error(e=TypeError, err_message=err_message, \
                    fail_gracefully=fail_gracefully, verbose=verbose)

            return False

        # This tests if var is a numeric value (given we know var isn't a 
        # bool at this point).
        elif isinstance(var, Number):
            return True
        else:
            err_message = 'Alert; wanted number but got', type(var), 'for var =', var
            handle_error(e=TypeError, err_message=err_message, \
                    fail_gracefully=fail_gracefully, verbose=verbose)

            return False

    if desired_type == 'nonstr_iterable':
        if isinstance(var, str):
            err_message = 'Alert; acceptable_type_lst was not a non-string '+\
                    'iterable but should be. acceptable_type_lst =', \
                    acceptable_type_lst, 'type(acceptable_type_lst) =', type(\
                    acceptable_type_lst)

            handle_error(e=TypeError, err_message=err_message, \
                    fail_gracefully=fail_gracefully, verbose=verbose)

            return False
        if hasattr(var, '__iter__'):
            return True

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


def check_input_var_type_lst(var_type_lst, fail_gracefully=False,\
        verbose=False):

    '''
    var_type_lst: list of (list of length 2)
        Each element of var_type_lst must be a list of length 2 with format
        [var, type]. e.g. [foo, int]. Allowed Values for type are all python
        types (returned by type()), and also 'path', 'file', 'dir', 'number'.

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
        include python types or 'path', 'file', 'dir', 'number'.
    '''
    check_input_var_type(var_type_lst, list, fail_gracefully=fail_gracefully,\
            verbose=verbose)
    
    check_input_var_type(fail_gracefully, bool, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    check_input_var_type(verbose, bool, fail_gracefully=fail_gracefully,\
            verbose=verbose)

    # Call check_input_var_type for every var desired_type pair.
    checks = [check_input_var_type(var, desired_type, \
            fail_gracefully=fail_gracefully, verbose=verbose) 
            for var, desired_type in var_type_lst]

    # See if all values in checks are True which means all vars are of the
    # desired type.
    passed_checks = len(checks) == checks.count(True)
    return passed_checks


def check_any_acceptable_type(var, acceptable_type_lst,\
            fail_gracefully=False, verbose=False)):
    
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
    check_input_var_type(acceptable_type_lst, 'nonstr_iterable', \
            fail_gracefully=False, verbose=verbose)
    
    for desired_type in acceptable_type_lst:
        if check_input_var_type(var, desired_type, \
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
            
    Return: None
    
    Purpose: Instead of the built-in assert function, use assert_gracefully to
        allow writing custom and descriptive error messages. This function also
        allows continuing program flow if the assertion fails by setting
        fail_gracefully to True. The err_message is printed if the assertion
        failed regardless if fail_gracefully is True if verbose is True.
    '''
    check_input_var_type(condition, bool, fail_gracefully=False, verbose=\
            verbose)
            
    # If condition is True, then the assertion passes and nothing left to do.
    if condition:
        return
    
    # The assertion has failed so we will throw an error.
    handle_error(e=AssertionError, err_message=err_message,\
            fail_gracefully=fail_gracefully, verbose=verbose))


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

    if not check_input_var_type_lst(var_type_lst, fail_gracefully=fail_gracefully, \
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