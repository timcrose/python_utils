import re, random, itertools
from python_utils import math_utils
import numpy as np

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    #Remove the .lower() if you want uppercase letters to come before all lowercase letters
    alphanum_key = lambda key: [convert(c.lower()) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def concat_str_list(str_list, delimiter=' '):
    concat_str = ''
    for elem in str_list:
        concat_str += elem + delimiter
    return concat_str

def indices(lst, val):
    '''
    lst: list
        list to search
    val: *
        value to search the list for

    return: list
        All indices matching val. If there are no indices
        in lst matching val, return empty list
    Purpose: output indices where the val is found in lst. If val is
        not found in lst, output empty list
    '''
    return [i for i, item in enumerate(lst) if item is val or item == val]

def random_index(lst):
    '''
    lst: list
        List
    
    return: int
        random number that could be an index

    Purpose: get a valid random index
    '''
    if len(lst) == 0:
        raise ValueError('lst has len 0.')
    return random.randint(0, len(lst) - 1)

def random_value(lst):
    '''
    lst: list
        List
    
    return: int
        random element of lst

    Purpose: return one of the values in lst at random
    '''
    if len(lst) == 0:
        raise ValueError('lst has len 0.')
    return lst[random_index(lst)]

def flatten_list(lst):
    '''
    lst: list
        List of lists

    return: list
        1D list which is the flattened list
    Purpose: Flatten a list of lists
    '''
    return list(itertools.chain.from_iterable(lst))

def sort_list_by_col(lst, col, reverse=False):
    return sorted(lst,key=lambda l:l[col], reverse=reverse)

def multi_insert(lst, indices_to_insert, data_to_insert=None, direction='left'):
    '''
    lst: list
        parent list which will get items inserted into it by this function
    indices_to_insert: iterable of int
        indices of lst to insert into (to the direction specified)
    data_to_insert: iterable or None
        list of entries to insert corresponding to the order specified in
        indices_to_insert. If None, then use the data at indices_to_insert in lst
        to fill.
    direction: str
        'left': If index to insert is 0, then the new datum will be in position
        0 in the final list.
        'right': If index to insert is 0, then the new datum will be in position
        1 in the final list.
    
    return: list
       list of entries in iterable but now with data_to_insert inserted
    Purpose: insert many indices from an iterable at once to avoid 
        clashes of indices changing as elements are added
    '''
    lst = list(lst)
    if data_to_insert is None:
        data_to_insert = np.array(lst)[indices_to_insert]
    assert(len(data_to_insert) == len(indices_to_insert))
    acc = 0
    if direction == 'left':
        for i in range(len(data_to_insert)):
            lst.insert(indices_to_insert[i] + acc, data_to_insert[i])
            acc += 1
    elif direction == 'right':
        for i in range(len(data_to_insert)):
            lst.insert(indices_to_insert[i] + acc + 1, data_to_insert[i])
            acc += 1
    else:
        raise Exception('only "left" or "right" are acceptable arguments for "direction".')
    return lst


def multi_put(lst, indices_to_put, data_to_insert, append_if_beyond_length=False):
    '''
    lst: list
        parent list which will get items inserted into it by this function
    indices_to_put: iterable of int
        non-negative indices of the returned list that data_to_insert will be located in.
        This is not possible in the cases where an index is beyond the resulting length
        of the returned list. In these cases, see append_if_beyond_length description.
    data_to_insert: iterable
        list of entries to put in lst corresponding to the order specified in
        indices_to_put and according to append_if_beyond_length.
    append_if_beyond_length: bool
        When an index is beyond the resulting length of the returned list, if
        True: append the corresponding value in data_to_insert to the end of the
            resultant list, in increasing order of all values whos indices are
            beyond the resultant length.
            e.g. lst = [0.1, 2.3, 4.5], indices_to_put = [0,15,9], data_to_insert = [1.05, 0.2, 3.4]
            then 15 and 9 are beyond the length of the resultant list which will be 6. So,
            return [1.05, 0.1, 2.3, 4.5, 3.4, 0.2]
        False: throw an IndexError
    
    return: list
       list of entries in iterable but now with data_to_insert put at indicies_to_put in the returned list, or
       according to append_if_beyond_length if some indices are beyond the resulting length of the returned list.

    Purpose: Use this function if you want to insert some values into a list such that their resulting indices in
        that list are the provided indicies_to_put (with the exception of when some indices provided are beyond) 
        the length of the resultant list (len(input lst) + len(data_to_insert)). See description of
        append_if_beyond_length for these cases.
    '''
    lst = list(lst)
    assert(len(data_to_insert) == len(indices_to_put))

    len_of_resultant_list = len(lst) + len(data_to_insert)

    # Begin by sorting indices_to_put and data_to_insert together by indices_to_put
    idx_val_pairs = list(map(list, zip(indices_to_put, data_to_insert)))
    idx_val_pairs = np.array(sort_list_by_col(idx_val_pairs, 0))
    for i,idx_val_pair in enumerate(idx_val_pairs):
        idx, val = idx_val_pair
        if idx >= len_of_resultant_list and not append_if_beyond_length:
            raise IndexError('append_if_beyond_length is False but got an index beyond the length of what will be the final list',
                            idx)

        lst.insert(idx, val)
    return lst


def fill_missing_data_evenly(lst, expected_len, direction='left'):
    '''
    lst: list
        parent list which will get items inserted into it by this function
    expected_len: int
        length you want your list to contain.
    direction: str
        'left': If index to insert is 0, then the new datum will be in position
        0 in the final list.
        'right': If index to insert is 0, then the new datum will be in position
        1 in the final list.
    
    return: list
       list of entries in iterable but now with filled in evenly
    Purpose: An example of when to use this function is when data is supposed to 
        be gathered every second from the internet, but the internet connection is
        spotty and so only 90% of the data was collected. You'd like to fill in the
        missing values to end up with 100% of the data points, but concatenating
        them all to the end doesn't make much sense. Instead, insert data as evenly
        as possible (using the nearest known data value for each inserted point.)
    '''
    raw_len = len(lst)
    additional_num_points_needed = expected_len - raw_len
    if additional_num_points_needed <= 0:
        return lst
    partition_size = float(raw_len) / float(additional_num_points_needed)
    #Divide the list up into additional_num_points_needed partitions and insert
    # a point (equal to its neighbor) in the center of each partition
    partitions = [0] + [(i + 1) * partition_size for i in range(additional_num_points_needed)]
    indices_to_insert = [int(math_utils.round((partitions[i + 1] - partitions[i]) / 2.0) + partitions[i]) for i in range(len(partitions) - 1)]
    filled_list = multi_insert(lst, indices_to_insert, direction=direction)
    return filled_list


def multi_delete(lst, indices_to_delete, axis=0):
    '''
    lst: iterable
        list or iterable
    indices_to_delete: iterable
        indices of lst to delete from lst
    axis: int
        Either 0 or 1 allowed. axis only used when a list of lists or matrix provided as lst.
        0: delete rows
        1: delete columns. May result in throwing an error if a 1D array or list of number is passed (needs
            to be list of lists or matrix)
    
    return: list
       list of entries in iterable but now with indices_to_delete deleted
    Purpose: delete many indices from an iterable at once to avoid 
        clashes of indices changing as elements are deleted
    '''
    indexes = sorted(list(indices_to_delete), reverse=True)
    type_of_lst = type(lst)
    if axis == 1 and isinstance(lst, list):
        lst = np.array(lst)
    if isinstance(lst, np.ndarray):
        for index in indexes:
            lst = np.delete(lst, index, axis=axis)
    else:
        for index in indexes:
            del lst[index]
    if type_of_lst is list and not isinstance(lst, list):
        lst = list(map(list, lst))
    return lst


def randomsubset_not_in_other_set(n_choose, subset_to_avoid, total_set):
    '''
    n_choose: int
        number of indices to choose

    subset_to_avoid: iterable
        set of indices that you don't want to include in your subset

    total_set: iterable or None
        The set of indices to pick from. If None, then

    Return: np array, shape: ,n_choose
        A list of indices to include in the subset.

    Purpose: Determine a random subset of the entire population of indices
        (total_set) to include
       in a subset.
    '''
    if type(subset_to_avoid) is not set:
        subset_to_avoid = set(subset_to_avoid)
    if type(total_set) is not set:
        total_set = set(total_set)

    valid_set_to_choose_from = total_set.difference(subset_to_avoid)
    return np.array(random.sample(valid_set_to_choose_from, n_choose))


def sort_by_col(data, col, reverse=False):
    '''
    data: numpy array, shape (at least one column)
        array to sort

    col: int
        column index to sort

    reverse: bool
        True: sort from largest to smallest
        False: sort from smallest to largest

    return: the data sorted by the specified column

    purpose: Sort data by the specified column
    '''
    if reverse:
        sorted_data = data[np.flip(np.argsort(data[:,col]))]
    else:
        sorted_data = data[np.argsort(data[:,col])]
    return sorted_data

def is_contiguous(indices, mat_shape):
    '''
    indices: np.array, shape (x,2)
        Matrix of x indices where each index is an int i,j pair corresponding to an element index in a matrix
        of shape mat_shape.

    mat_shape: length 2 tuple, list, or 1D array
        Shape of the matrix which indices are indices for.

    Return: bool
        True: This list of indices is in contiguous order
        False: o/w

    Purpose: This function returns a bool for whether the provided indices of a 2D matrix are in contiguous
        order (this would be useful e.g. when writing to a np.memmap where you only get an int offset value).
    '''
    if indices.dtype != np.int64:
        raise Exception('indices must have all integers')
    if indices.shape[1] != 2:
        raise Exception('indices, must have shape (x,2) because it is a list of indices of a 2D matrix.')
    if len(indices) < 2:
        return True
    n, m = mat_shape
    prev_i, prev_j = indices[0][0], indices[0][1]
    for i,j in indices[1:]:
        if i >= n or j >= m:
            raise IndexError('Index given in indices,', i,j, ', is out of bounds for matrix of shape', mat_shape)
        if i != prev_i:
            if i < prev_i or i != prev_i + 1:
                return False
            if j != 0 or prev_j != m - 1:
                return False
        elif j != prev_j + 1:
            return False
        prev_i, prev_j = i, j
    return True


def range_float_incr(lower_bound, upper_bound, increment_value, include_lower_bound=True, include_upper_bound=False):
    '''
    lower_bound: float
        Minimum and first value in returned list if include_lower_bound=True, else lower_bound + increment_value
        is the minimum and first value.

    upper_bound: float
        Maximum and last value in returned list if include_upper_bound=True, else upper_bound - increment_value
        is the maximum and last value.

    increment_value: float
        Value by which each successive value in the returned list should differ.

    include_lower_bound: bool
        See lower_bound

    include_upper_bound: bool
        See upper_bound

    Return: list
        Ascending ordered list which begins according to lower_bound documentation and ends according to
        upper_bound documentation with each successive value incremented by increment_value.

    Purpose: Create a list which starts from one number and includes all numbers up until (and possibly including)
        another, higher number at intervals of increment_value. Similar to range but can use float increments.
    '''

    if include_lower_bound:
        val = lower_bound
    else:
        val = lower_bound + increment_value
    if include_upper_bound:
        max_val = upper_bound
    else:
        max_val = upper_bound - increment_value
    lst = []
    i = 0
    while val < max_val:
        val = lower_bound + i * increment_value
        lst.append(val)
        i += 1
    return lst


def split_up_list_evenly(lst, num_partitions):
    '''
    lst: list or np.array
        Overall list of elements to split up amongst partitions

    num_partitions: int
        Number of partitions to split up the list into.

    Return:
    split_lst: list or np.array
        List of elements that each partition should get.

    Purpose: This function divides up the given list or array as evenly as possible.
        e.g. if num_partitions = 3, and lst = [0,1,2,3,4], then partition 0 gets [0,1],
        partition 1 gets [2,3], and partition 2 gets [4].
    '''
    err_utils.check_input_var_type(num_partitions, int)
    if num_partitions < 1:
        raise ValueError('num_partitions must be >= 1 but got num_partitions =',num_partitions)
    num_tasks = len(lst)
    tasks_per_partition = int(num_tasks / num_partitions)
    num_remainder_tasks = num_tasks - tasks_per_partition * num_partitions
    split_lst = [lst[i * (tasks_per_partition + 1) : (i + 1) * (tasks_per_partition + 1)] \
                for i in range(num_partitions - 1)
                ] \
                + lst[num_remainder_tasks + (num_partitions - 1) * tasks_per_partition : num_remainder_tasks + num_partitions * tasks_per_partition]
    if isinstance(lst, list):
        return split_lst
    else:
        return np.array(split_lst)
