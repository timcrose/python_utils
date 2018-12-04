

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

def split_str_with_many_delimiters(string, delimiters=[' ', '(', ')', '.', ',', '[', ']', '/', '+', '-', '*', '%', '#', ':'], return_delim_indices=False):
    if return_delim_indices:
        delim_indicies = []
        for s, char in enumerate(string):
            if char in delimiters:
                string[s] = ' '
                delim_indicies.append([s, delimiters[delimiters.index(char)])
        return string.split(), delim_indicies
    else:
        for delim in delimiters:
             string = string.replace(delim, ' ')
        return string.split()
