import re, random, functools, operator
from python_utils import math_utils, err_utils
import numpy as np
from copy import deepcopy
import itertools, collections
from scipy.spatial import distance


def build_grid(grid_info, include_endpoint=False, 
allow_to_spill_over=False, return_np=False, 
dtype=int):
    '''
    grid_info: np.array shape (n_dim, 3)
        n_dims is the number of dimensions you want grid points for
        Each row is [lower_bound, upper_bound, step]
        
    include_endpoint: bool
        True: Include upper_bound if it is a multiple of step
        False: Do not include anything >= upper_bound
        
    allow_to_spill_over: bool
        True: If include_endpoint, include the first value >= upper_bound using step
        False: Do not include anything > upper_bound
        
    return_np: bool
        True: return np.array with dtype dtype
        False: return collections.deque
        
    dtype: type
        if return_np, this is the dtype of the np array
        
    Returns
    -------
    grid: collections.deque or np.array shape (permutations, n_dims)
        Grid points equally spaced in each dimension by each dimension's step
    '''
    ranges = [np.arange(grid_info[i,0], 
grid_info[i,1] + include_endpoint * (grid_info[i,2] * ((grid_info[i,1] % grid_info[i,2] == 0) or allow_to_spill_over)), 
grid_info[i,2]).tolist() for i in range(grid_info.shape[0])]
    
    grid = collections.deque(itertools.product(*ranges))
    if return_np:
        return np.array(grid, dtype=dtype)
    return grid


def build_grid_traversal(grid_info, include_endpoint=False, 
allow_to_spill_over=False, dtype=int):
    '''
    grid_info: np.array shape (n_dim, 3)
        n_dims is the number of dimensions you want grid points for
        Each row is [lower_bound, upper_bound, step]
        
    include_endpoint: bool
        True: Include upper_bound if it is a multiple of step
        False: Do not include anything >= upper_bound
        
    allow_to_spill_over: bool
        True: If include_endpoint, include the first value >= upper_bound using step
        False: Do not include anything > upper_bound
        
    dtype: type
        this is the dtype of the returned np array
        
    Returns
    -------
    grid: np.array shape (permutations, n_dims)
        Grid points equally spaced in each dimension by each dimension's step in
        the order that will be minimal total distance traveled.
    '''
    grid = build_grid(grid_info, include_endpoint=include_endpoint, allow_to_spill_over=allow_to_spill_over, return_np=True, dtype=dtype)
    # First, collapse grid such that each dimension's step is 1
    collapsed_grid_info = np.zeros(grid_info.shape)
    for i in range(grid.shape[1]):
        collapsed_grid_info[i, 2] = i + 1
        collapsed_grid_info[i, 1] = len(set(grid[:,i])) * collapsed_grid_info[i, 2]
    collapsed_grid = build_grid(collapsed_grid_info, include_endpoint=False, allow_to_spill_over=False, return_np=True, dtype=int)
    if len(grid) != len(collapsed_grid):
        raise Exception('len(grid) != len(collapsed_grid)', 'len(grid)', len(grid), 'len(collapsed_grid)', len(collapsed_grid))
    # The kernel is a pairwise distance matrix
    kernel = distance.squareform(distance.pdist(collapsed_grid))
    del collapsed_grid
    # Fill the diagonal so that you will not choose yourself as the node with the minimum distance.
    kernel_max = kernel.max()
    np.fill_diagonal(kernel, kernel_max + 1)
    visited_idxs = [0]
    while len(visited_idxs) < len(grid):
        # make the col of the already visited node maxed out so it wont be picked again
        kernel[:,visited_idxs[-1]] = kernel_max + 1
        # Use the row of the current node to see its distance to all other nodes and pick the minimal one
        min_dist_idx = np.argmin(kernel[visited_idxs[-1]])
        visited_idxs.append(min_dist_idx)
    return grid[visited_idxs]
    

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    #Remove the .lower() if you want uppercase letters to come before all lowercase letters
    alphanum_key = lambda key: [convert(c.lower()) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def indices(seq, val):
    '''
    seq: Sequence
        list to search
    val: Any
        value to search the sequence for

    return: list of int
        All indices matching val. If there are no indices
        in seq matching val, return empty list
    Purpose: output indices where the val is found in seq. If val is
        not found in seq, output empty list
    '''
    return [i for i, item in enumerate(seq) if item == val]

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
    #return list(itertools.chain.from_iterable(lst))
    return functools.reduce(operator.iadd, lst, [])

def sort_list_by_col(lst, col, reverse=False):
    return sorted(lst, key=lambda l:l[col], reverse=reverse)

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


def get_onehot_contiguous_idxs(onehot_arr, contiguous_len_range):
    '''
    Parameters
    ----------
    onehot_arr: 1D iterable of int
        1D array or list of only 1s and 0s.
        
    contiguous_len_range: iterable of int of length 2
        Only keep start,end ranges that are at least contiguous_len_range[0]
        long and are at most contiguous_len_range[1] long. (Inclusive)

    Returns
    -------
    contiguous_idxs: np.array shape (*,2)
        a 2D np array where each row is [start index, end index] of a contiguous streak
        in the onehot array where end - start is in the contiguous_len_range.

    Purpose
    -------
    Given a onehot 1D list or array, (only comprising 0s and 1s), return a 2D
    np array where each row is [start index, end index] of a contiguous streak
    in the onehot array. Only return rows where the end index - start index is
    at least contiguous_len_range[0] and at most contiguous_len_range[1].
    
    Notes
    -----
    1. end indices returned are actually 1 + the last index a 1 was found (because
        they are ranges)
    
    Examples
    --------
    onehot_arr = [1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1]
    contiguous_len_range = [2,10]
    return 
    [[ 4  7]
     [11 15]
     [20 23]]
    '''
    onehot_arr = np.append(onehot_arr, [0])
    shifted_arr = np.append([0], onehot_arr[:-1])
    diff_arr = onehot_arr - shifted_arr
    # 1s mark where a contiguous section starts and -1 marks one position after the end of a contiguous section
    ones = np.where(diff_arr == 1)[0]
    minus_ones = np.where(diff_arr == -1)[0]
    ranges = np.array(tuple(zip(ones, minus_ones)))
    if len(ranges) == 0:
        return []
    range_diffs = ranges[:,1] - ranges[:,0]
    return ranges[(range_diffs >= contiguous_len_range[0]) & (range_diffs <= contiguous_len_range[1])]
    

def get_contiguous_values(arr, contiguous_len=5):
    '''
    Parameters
    ----------
    arr: iterable, 1D
        1D iterable which has numbers that might be contiguous.
                                                              
    contiguous_len: int
        Each contiguous section is only included in the returned contiguous_values
        if the 'end' - 'start' value is >= contiguous_len

    Returns
    -------
    contiguous_values: list of dict
        Records the start value and end value of each contiguous section that is
        >= contiguous_len in length.
        
    Examples
    --------
    arr = [1,4,5,6,9,10,10,11,23,24,25] has contiguous sequences 4,5,6 and 23,24,25, 
    if contiguous_len is 3, so then we want to return 
    contiguous_values = [{'start': 4, 'end':6}, {'start': 23, 'end': 25}]

    Purpose
    -------
    Find a list of start,end pairs for contiguous sections >= contiguous_len in 1D array.
    '''
    contiguous_values = []
    # Find contiguous sections greater than contiguous_len in a row
    prev_elem = arr[0]
    current_streak = {'start': arr[0],'end': arr[0] + 1}
    for elem in arr:
        if elem == prev_elem + 1:
            current_streak['end'] = elem
        else:
            if current_streak['end'] - current_streak['start'] >= contiguous_len:
                contiguous_values.append(current_streak)
            current_streak = {'start': elem, 'end': elem + 1}
        prev_elem = deepcopy(elem)
    if current_streak['end'] - current_streak['start'] >= contiguous_len - 1:
        contiguous_values.append(current_streak)
    return contiguous_values
        

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
        i = 0
    else:
        i = 1
    if include_upper_bound:
        max_val = upper_bound
    else:
        max_val = upper_bound - increment_value
    lst = []
    val = lower_bound + i * increment_value
    while val <= max_val:
        lst.append(math_utils.round_nearest_multiple(val, increment_value))
        i += 1
        val = lower_bound + i * increment_value
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
    if not isinstance(lst, list):
        lst = list(lst)
    num_tasks = len(lst)
    tasks_per_partition = int(num_tasks / num_partitions)
    num_remainder_tasks = num_tasks - tasks_per_partition * num_partitions
    split_lst = [lst[\
                     i * tasks_per_partition + min(i, num_remainder_tasks):\
                     (i + 1) * tasks_per_partition + min(i, num_remainder_tasks) + int(num_remainder_tasks > i)] \
                 for i in range(num_partitions)
                ]
    return split_lst


def moving_average(arr, period, dtype=np.float64):
    '''
    arr: 1D np.array
        array of numbers over which to calculate a moving average

    period: int
        period must be >= 1 and is the number of elements at a time to take
        the average of.

    dtype: numpy type function
        Type of resulting numbers in the returned array. Can be,
        np.float32, np.float64, np.int32, etc

    Return: ma
        ma: np.array shape (len(arr) - period + 1,)
            Moving average of values in arr.

    Purpose: Calculate the simple arithmetic moving average of an array of numbers.
    '''
    ret = np.cumsum(arr)
    ret[period:] = ret[period:] - ret[:-period]
    ma = ret[period - 1:] / period
    return dtype(ma)


def equal_frequency_bins(arr, nbin):
    '''
    arr: sortable 1D iterable
        Array of numbers to perform frequency binning on.
    
    nbin: int
        The number of partitions (bins) to create.
    
    Returns
    -------
    equal_frequency_bin_edges: np.array shape (nbin + 1, )
        Values that are the edges for all bins such that all bins have approximately equal number of
        observations (data points) in them.
        e.g.
        arr = [1,2,3,4,5]
        nbin = 3
        equal_frequency_bin_edges = [1.0, 2.66666667, 4.33333333, 5.0] because then bin 0 contains 2 points: 1,2;
        bin 1 contains 2 points: 3,4; bin 2 contains 1 point: 5, and so the bins are approximately equally populated.

    Purpose
    -------
    This function creates bins for a group of observations such that each bin contains approximately the same number
    of observations (data points). This is useful for when, you need to focus your efforts on areas of higher density.
    For example, in Q-learning, you need to create a Q-table that partitions the state space. However, some regions
    of a particular feature could be highly populated while others are sparse. To simply split up the space for this
    feature into equidistant bins would mean a lot of unnecessary states and thus a lot of useless computation.
    '''
    nlen = len(arr)
    equal_frequency_bin_edges = np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(arr))
    return equal_frequency_bin_edges


def arrs_to_matrix(*arrs):
    '''
    Parameters
    ----------
    *arrs: iterables
        Any number of iterables of the same shape (usually)

    Returns
    -------
    matrix: np.array, shape: see zip documentation and below examples
    
    e.g.
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    c = np.array([[7,8],[9,10],[11,12]])
    d = np.array([[13,14],[15,16],[17,18]])
    
    arrs_to_matrix(*[a,b]) -> array(
        [[1, 4],
        [2, 5],
        [3, 6]])
    
    arrs_to_matrix([a,b]) -> array(
        [[[1, 2, 3]],
         
         [[4, 5, 6]]])
    
    arrs_to_matrix(*c,a) -> array(
        [[ 7,  9, 11,  1],
         [ 8, 10, 12,  2]])
    
    arrs_to_matrix(a,*c) -> array(
        [[ 1,  7,  9, 11],
         [ 2,  8, 10, 12]]
        
    arrs_to_matrix(c,d) -> array(
        [[[ 7,  8],
         [13, 14]],

        [[ 9, 10],
         [15, 16]],

        [[11, 12],
         [17, 18]]])
    
    arrs_to_matrix([c,d]) -> array(
        [[[[ 7,  8],
          [ 9, 10],
          [11, 12]]],
    
    
        [[[13, 14],
          [15, 16],
          [17, 18]]]])
    
    arrs_to_matrix(*c,*d) -> array(
        [[ 7,  9, 11, 13, 15, 17],
         [ 8, 10, 12, 14, 16, 18]])
    
    Purpose
    -------
    Pass in a list of 1D arrays (with * in front) and get out a matrix where 
    the first column is the first arr, the second column is the second arr, etc.
    You can also pass in a variety of different things.
    '''
    return np.array(tuple(zip(*arrs)))


def where_mat(condition):
    '''
    Parameters
    ----------
    condition: bool
        expression that is passed into np.where(condition)

    Returns
    -------
    Rectangular matrix output format of the results of np.where
    
    Purpose
    -------
    Often when you run np.where(), you're hoping for a matrix where the first
    column is the found row indices and the second column is the corresponding
    found col indices. This allows iterating over the [row, col] pairs instead
    of just the index.
    '''
    return arrs_to_matrix(*np.where(condition))


def enum(*sequential, **named):
    """
    Purpose
    -------
    Handy way to fake an enumerated type in Python. Useful, for example, for 
    organizing MPI tags.
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    
    Example
    -------
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    comm.send(data, dest=1, tag=tags.START)
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


