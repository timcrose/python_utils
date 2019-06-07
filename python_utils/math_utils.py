
"""
Created on Sun Apr  1 16:29:48 2018

@author: timcrose
"""

import math, random
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN

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

def round(number, num_decimal_places):
    '''
    number: number
        number to round
    num_decimal_places: number of decimal places to round number to
    
    return: number (float)
        rounded number
    Purpose: round a number to the specified number of decimal places. Existing 
        round functions may not round correctly.
    '''
    decimal_str = '1.'
    for decimal_place in range(num_decimal_places):
        decimal_str += '1'
    return float(Decimal(str(number)).quantize(Decimal(decimal_str), rounding=ROUND_HALF_UP))

def randrange_float(start, stop, step, num_decimal_places=4):
    return round(random.randint(0, int((stop - start) / step)) * step + start, num_decimal_places)
