import re, random

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
