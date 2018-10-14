
def find_nth_occurrence_in_str(input_str, search_str, n):
    parts= input_str.split(search_str, n+1)
    if len(parts)<=n+1:
        return -1
    return len(input_str)-len(parts[-1])-len(search_str)
