from python_utils import file_utils
import os
import pandas as pd
import numpy as np
from copy import deepcopy


def insert_df_into_df(inserted_df, df):
    '''
    Insert the rows of one DataFrame into another and sort the result by index.

    Parameters:
        inserted_df (pd.DataFrame): The DataFrame to be inserted. Must have the same columns
            and compatible index types as `df`. For example:
            `inserted_df = pd.DataFrame({"onset": 30.0, "length": 1.3}, index=[2.5])`
        df (pd.DataFrame): The host DataFrame into which rows will be inserted.

    Returns:
        pd.DataFrame: A new DataFrame with rows from `inserted_df` incorporated into `df`,
        sorted by index.

    Notes:
        If there are duplicate index values, the inserted rows will appear after the original
        ones when sorted. For example:

        >>> df = pd.DataFrame({"onset": [30.0, 20.0], "length": [1.3, 1.2]}, index=range(2))
        >>> inserted_df = pd.DataFrame({"onset": [40.0, 50.0], "length": [1.4, 1.5]}, index=[0, 1])
        >>> insert_df_into_df(inserted_df, df)
             onset  length
        0   30.0     1.3
        1   40.0     1.4
        2   20.0     1.2
        3   50.0     1.5
    '''
    df = df.append(inserted_df, ignore_index=False)
    df = df.sort_index().reset_index(drop=True)
    return df


def get_value_from_row(df, col_name, row_value, value_col_name):
    '''
    Retrieve a value from a specific row in a DataFrame where a given column matches a target value.

    Parameters:
        df (pd.DataFrame): The DataFrame to search.
        col_name (str): The name of the column used to identify the target row.
        row_value (Any): The value to match in the specified column.
        value_col_name (Union[str, int]): The column name or index from which to extract the value.

    Returns:
        Any: The value from the specified column in the first row where `df[col_name] == row_value`.

    Raises:
        IndexError: If no matching row is found.
        KeyError: If `col_name` or `value_col_name` does not exist in the DataFrame.

    Example:
        >>> get_value_from_row(df, 'name', 'Alice', 'score')
        85
    '''
    return df[df[col_name] == row_value][value_col_name].values[0]


def write_to_excel_carefully(df, fpath, timeout=1, total_timeout=10):
    """
    Attempts to write a DataFrame to an Excel file with retry logic in case of an `OSError`.

    This function retries writing the DataFrame to the specified Excel file path up to
    `num_write_attempts` times, waiting `iteration_delay` seconds between each attempt.
    If all attempts fail, it prints a custom error message and re-raises the last `OSError`.

    Parameters:
        df (pd.DataFrame): The DataFrame to write to Excel.
        fpath (str): The file path where the Excel file should be saved.
        timeout (float): Time in seconds between each attempt
        total_timeout (float): Time in seconds before raising an error.
    Raises:
        OSError: Re-raises the last encountered `OSError` if writing fails after all attempts.

    Example:
        >>> write_to_excel_carefully(df, 'output.xlsx')
    """
    fname = file_utils.fname_from_fpath(fpath)
    dirname = os.path.dirname(fpath)
    lock_file_fpath = file_utils.write_intention_lock_file(dirname, fname + '.lock', wait_to_write_file=True, timeout=timeout, total_timeout=total_timeout)
    if not lock_file_fpath:
        raise OSError(f'Could not write lock file in {total_timeout} seconds so could not write to fpath {fpath}')
    df.to_excel(fpath, index=False)
    file_utils.rm(lock_file_fpath)
        
        
def dict_matches_row_index(d, df, return_all=True):
    """
    Search for rows in a DataFrame that match all key-value pairs in a given dictionary.

    Parameters
    ----------
    d : dict
        Dictionary of column-value pairs to match against the DataFrame.
    df : pandas.DataFrame
        The DataFrame to search.
    return_all : bool, optional (default=True)
        If True, returns all matching row indices as a pandas Index.
        If False, returns the index of the first matching row.

    Returns
    -------
    int, pandas.Index, or None
        - If return_all is False: returns the index of the first matching row.
        - If return_all is True: returns a pandas Index of all matching rows.
        - If no match is found: returns None.

    Notes
    -----
    - Matching is exact and case-sensitive.
    - Only the keys in `d` are used for comparison.
    
    Example Usage
    -------------
    import pandas as pd

    df = pd.DataFrame({
        'type': ['apple', 'banana', 'apple', 'orange'],
        'color': ['red', 'yellow', 'green', 'orange'],
        'price': [1.2, 0.5, 1.0, 0.8]
    })
    
    query = {'type': 'apple'}
    
    # Return first match
    first_match = dict_matches_row_index(query, df)
    print(first_match)  # Output: 0
    
    # Return all matches
    all_matches = dict_matches_row_index(query, df, return_all=True)
    print(all_matches)  # Output: Int64Index([0, 2], dtype='int64')
    
    # No match
    no_match = dict_matches_row_index({'type': 'kiwi'}, df)
    print(no_match)  # Output: None
    """
    mask = df[list(d.keys())].apply(lambda row: all(row[k] == v for k, v in d.items()), axis=1)
    matches = df[mask]
    if matches.empty:
        return None
    return matches.index if return_all else matches.index[0]