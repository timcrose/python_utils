import re
from glob import glob

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    #Remove the .lower() if you want uppercase letters to come before all lowercase letters
    alphanum_key = lambda key: [convert(c.lower()) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)
