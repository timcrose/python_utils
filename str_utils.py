import list_utils

def find_nth_occurrence_in_str(input_str, search_str, n, reverse=False, overlapping=0):
    if reverse:
        match = input_str.rfind(search_str)
        while match >= 0 and n > 0:
            match = input_str.rfind(search_str, 0, match + (len(search_str) - 1) * overlapping)
            n -= 1
        return match
    else:
        match = input_str.find(search_str)
        while match >= 0 and n > 0:
            match = input_str.find(search_str, match + len(search_str) ** overlapping)
            n -= 1
        return match

def str_item_insert(string, item, i, after=True):
    '''
    string: str
        string to operate on
    item: str
        The string that will be inserted into string
    i: int
        the elem index string to do the inserting
    after: bool
        True: insert after i
        False: insert before i
        Note that if i >= len(string) then T/F causes the
        same behavior (after)

    return: str
        the string now with item inserted before/after position i
    Purpose: Insert item into string before/after position i
    '''
    if len(string) == 0:
        return item
    if i < 0:
        i = len(string) + i
    if after:
        return string[:i + 1] + item + string[i + 1:]
    else:
        return string[:i] + item + string[i:]


def str_item_assignment(string, item, i):
    '''
    string: str
        string to operate on
    item: str
        string that will replace the ith elem of string
    i: int
        the elem index of string to replace with item

    return: str
        the string now with the ith element replaced by item
    Purpose: replace the ith character of string with item
    '''
    return string[:i] + item + string[i + 1:] 

def multiple_str_item_assignment(string, item_list, i_list):
    '''
    string: str
        string to operate on
    item_list: list of str
        strings that will replace the ith elem of string
    i_list: list of int
        the elem index of string to replace with item

    return: str
        the string now with the ith element replaced by item
        for each i
    Purpose: replace characters of string with items
    '''
    assert(len(item_list) == len(i_list))
    item_data = [[item_list[j], i_list[j]] for j in range(len(item_list))]
    item_data = list_utils.sort_list_by_col(item_data, 1)
    offset = 0
    for item, i in item_data:
        if len(item) >= 1:
            offset += len(item) - 1
        string = str_item_assignment(string, item, i + offset)
        if len(item) < 1:
            offset -= 1
    return string


def delete_items_from_str_by_idx(string, i_list):
    '''
    string: str
        string to operate on
    i_list: list of int
        the elem index of string to delete

    return: str
        the string now with the ith element deleted
    Purpose: Deleted characters of string at positions in given i_list
    '''
    i_list = list(i_list)
    for i in i_list:
        assert(type(i) is int)
    item_list = ['' for i in i_list]
    modified_str = multiple_str_item_assignment(string, item_list, i_list)   
    return modified_str


def split_str_with_many_delimiters(string, delimiters=[' ', '(', ')', '.', ',', '[', ']', '/', '+', '-', '*', '%', '#', ':'], return_delim_indices=False):
    if return_delim_indices:
        delim_indicies = []
        for s, char in enumerate(string):
            if char in delimiters:
                string = str_item_assignment(string, ' ', s)
                delim_indicies.append([s, delimiters[delimiters.index(char)]])
        return string.split(), delim_indicies
    else:
        for delim in delimiters:
             string = string.replace(delim, ' ')
        return string.split()
