
"""
Created on Sun Apr  1 16:29:48 2018

@author: timcrose
"""

import math

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
        return round(round(number / a) * a, -int(math.floor(math.log10(a))))

def mean(lst):
    return sum(lst) / float(len(lst))
