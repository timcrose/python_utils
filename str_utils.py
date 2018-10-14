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
