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

def multi_insert(list_, indices_to_insert, data_to_insert=None, direction='left'):
    '''
    list_: list
        parent list which will get items inserted into it by this function
    indices_to_insert: iterable
        indices of list_ to insert into (to the direction specified)
    data_to_insert: iterable or None
        list of entries to insert corresponding to the order specified in
        indices_to_insert. If None, then use the data at indices_to_insert in list_
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
    list_ = list(list_)
    if data_to_insert is None:
        data_to_insert = np.array(list_)[indices_to_insert]
    assert(len(data_to_insert) == len(indices_to_insert))
    acc = 0
    if direction == 'left':
        for i in range(len(data_to_insert)):
            list_.insert(indices_to_insert[i] + acc, data_to_insert[i])
            acc += 1
    elif direction == 'right':
        for i in range(len(data_to_insert)):
            list_.insert(indices_to_insert[i] + acc + 1, data_to_insert[i])
            acc += 1
    else:
        raise Exception('only "left" or "right" are acceptable arguments for "direction".')
    return list_


def fill_missing_data_evenly(list_, expected_len, direction='left'):
    '''
    list_: list
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
    raw_len = len(list_)
    additional_num_points_needed = expected_len - raw_len
    if additional_num_points_needed <= 0:
        return list_
    partition_size = float(raw_len) / float(additional_num_points_needed)
    #Divide the list up into additional_num_points_needed partitions and insert
    # a point (equal to its neighbor) in the center of each partition
    partitions = [0] + [(i + 1) * partition_size for i in range(additional_num_points_needed)]
    indices_to_insert = [int(math_utils.round((partitions[i + 1] - partitions[i]) / 2.0) + partitions[i]) for i in range(len(partitions) - 1)]
    filled_list = multi_insert(list_, indices_to_insert, direction=direction)
    return filled_list


def multi_delete(list_, indices_to_delete):
    '''
    list_: iterable
        list or iterable
    indices_to_delete: iterable
        indices of list_ to delete from list_
    
    return: list
       list of entries in iterable but now with indices_to_delete deleted
    Purpose: delete many indices from an iterable at once to avoid 
        clashes of indices changing as elements are deleted
    '''
    indexes = sorted(list(indices_to_delete), reverse=True)
    for index in indexes:
        del list_[index]
    return list_


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


def sort_by_col(data, col):
    '''
    data: numpy array, shape (at least one column)
        array to sort

    col: int
        column index to sort

    return: the data sorted by the specified column

    purpose: Sort data by the specified column
    '''
    #try:
    #    sorted_data = data[np.argsort(data[:,col])]
    #except:
    #    return None
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