from python_utils import list_utils
from python_utils.type_utils import Optional, Tuple, Str_List, List, Int_List
# Edit this list of characters as desired.
BASE_ALPH = tuple("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`~!@#$%^&*()-=[]\;',./_+{}|:<>?")
BASE_DICT = dict((c, v) for v, c in enumerate(BASE_ALPH))
BASE_LEN = len(BASE_ALPH)

def base_encode(string: str, base_dct: Optional[dict]=None) -> int:
    '''
    string: str
        String that you want to encode into a number.

    base_dct: dict
        Keys are the characters in the alphabet (list of characters included in the encoding mapping).
        Values are the index of the character in the list.
        e.g.
        if BASE_ALPHA = tuple("abcd") then base_dct should be
        {'a':0,'b':1,'c':2,'d':3}

    Return
    ------
    num: int
        Number that is the encoded version of the string.

    Purpose
    -------
    Convert string into an integer using a particular alphabet (list of characters) such that a unique
    integer exists for every string made up of characters from the "alphabet".
    '''
    num = 0
    if base_dct is not None:
        base_len = len(base_dct)
        for char in string:
            num = num * base_len + base_dct[char]
        return num
    else:
        for char in string:
            num = num * BASE_LEN + BASE_DICT[char]
        return num


def base_decode(num: int, base_alphabet: Optional[Tuple]=None) -> str:
    '''
    num: int
        Number that is the encoded version of the string that you want to decode (turn into the string).

    base_alphabet: tuple
        Tuple of characters that was used in the base_dct when the string was encoded into num.
        e.g.
        tuple("abcd2") would include all strings made up of characters a, b, c, d, and 2.

    Return
    ------
    decoded_str: str
        String that was encoded into num using base_alphabet.

    Purpose
    -------
    Once a string has been encoded into an integer (using a particular base_alphabet), use this function to 
    get back the original string (using the same base_alphabet).
    '''
    decoded_str = ""
    if base_alphabet is not None:
        base_len = len(base_alphabet)
        while num:
            num, rem = divmod(num, base_len)
            decoded_str = base_alphabet[rem] + decoded_str
        return decoded_str
    else:
        while num:
            num, rem = divmod(num, BASE_LEN)
            decoded_str = BASE_ALPH[rem] + decoded_str
        return decoded_str


def find_nth_occurrence_in_str(input_str: str, search_str: str, n: int, reverse: bool=False, overlapping: int=0) -> int:
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


def encode_lst_to_ascii(lst_of_str: Str_List) -> List[bytes]:
    '''
    lst_of_str: list of str
        List of python strings that you want to encode into an ascii object

    Return
    ------
    lst_of_ascii_str: list of ascii-encoded strings
        List of asci-encoded strings.

    Purpose
    -------
    Encode each string in lst_of_str into its acii encoding using the built-in encode function.
    This is useful when trying to write a list of strings to the attrs_dct of a .h5 file.

    Notes
    -----
    1. Decode with decode_lst_of_ascii
    '''
    lst_of_ascii_str = [strng.encode('ascii', 'ignore') for strng in lst_of_str]
    return lst_of_ascii_str


def decode_lst_of_ascii(lst_of_ascii_str: List[bytes]) -> Str_List:
    '''
    lst_of_ascii_str: list of ascii-encoded strings
        List of ascii-encoded strings.

    Return
    ------
    lst_of_str: list of str
        List of python strings that you had previously encoded into ascii strings

    Purpose
    -------
    Use this function on a list of strings that have been encoded with the function encode_lst_to_ascii
    in order to get what the original strings were before the encoding.
    '''
    lst_of_str = [ascii_str.decode() for ascii_str in lst_of_ascii_str]
    return lst_of_str


def str_item_insert(string: str, item: str, i: int, after: bool=True) -> str:
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


def str_item_assignment(string: str, item: str, i: int) -> str:
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

def multiple_str_item_assignment(string: str, item_list: Str_List, i_list: Int_List) -> str:
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


def delete_items_from_str_by_idx(string: str, i_list: Int_List) -> str:
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


def split_str_with_many_delimiters(string: str, 
delimiters: Str_List=[' ', '(', ')', '.', ',', '[', ']', '/', '+', '-', '*', '%', '#', ':'], 
return_delim_indices: bool=False) -> Str_List:
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
