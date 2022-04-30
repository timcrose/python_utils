
"""
Created on Tue Feb  6 19:16:48 2018

@author: timcrose
"""

import csv, json, os, sys, pickle
from glob import glob
import shutil
from python_utils import time_utils

def output_from_rank(message_args, rank, mode='a', output_fpath_prefix='output_from_world_rank_'):
    output_fpath = output_fpath_prefix + str(rank)
    with open(output_fpath, mode=mode) as f:
          print(message_args, file=f)


def find(start_paths, name_search_str=None, recursive=True, \
        absolute_paths=False, clean_format=True, find_files=True, \
        find_dirs=True):
    '''
    start_paths: str or list of str
        Path of a directory from which search for files begins. If start_path
        contains a '*', then the returned list will include paths found
        starting from any directory matching the pattern. The start_path
        may be a relative path such as 'dir' and 'dir/dir'. See Params
        description below for detailed return values based on start_path and 
        name_search_str and recursive combinations.

        If start_path is a list of paths to directories, then files matching
        name_search_str will be searched for starting from all directories in
        this list.
        

    name_search_str: str or None
        Path name search pattern which may include '*' and may be relative. So,
        'file' and 'dir/file' work. '' typically yields directories and None
        typically yields files and directories. See Params description
        below for detailed return values based on start_path and 
        name_search_str and recursive combinations.

    recursive: bool
        True: Search all subdirectories of path(s) matching the start_path for
            matches of name_search_str and include them all in the returned
            list.
        False: Search only the immediate directories that match the pattern 
            given in start_path for matches of name_search_str and include them 
            in the returned list.

    absolute_paths: bool
        True: paths in the returned list will be absolute paths (as returned
        by os.path.abspath).
        False: paths in the returned list will be relative paths.

    clean_format: bool
        True: Returned paths will not have './' prepended or '/.' or '/'
            appended. Note that './' and './.' will be returned as '.'
        False: Returned paths may or may not have './' prepended or '/.'
            appended. Only set clean_format to False if you know that having
            such strings in your path will not affect anything. In this case,
            not converting to clean format will save a (usually tiny) bit of
            time.

    find_files: bool
        True: Include found files in the returned list of paths.
        False: Exclude found files in the returned list of paths.

    find_dirs: bool
        True: Include found directories in the returned list of paths.
        False: Exclude found directories in the returned list of paths.

    Params refers to start_path, name_search_str, recursive
    Behavior refers to if the returned list contains:
    'files in .', 'dirs in .', 'files in dirs of .',
    'dirs in dirs of .', 'files in dirs of dirs of .',
    'dirs in dirs of dirs of .', '.'

    Notes on format of Params:
    (1) 'dir_*' refers to a pattern for start_path.

    (2) 'f*' refers to a pattern for name_search_str.

    Notes on format of Behavior:
    (1) file refers to a file in the '.' (current) directory.

    (2) dir refers to a directory in the '.' (current) directory.

    (3) dir/dir/.../file refers to files in any number of sub-directories
        of the first dir. e.g. If 'dir_0' is in '.', 'f0.txt' and 'dir_1' are
        in ./dir_0, and 'f1.txt' is in ./dir_0/dir_1, then f0.txt and f1.txt
        will be found.

    (4) dir/dir/.../dir refers to directories in any number of sub-directories
        of the first dir. e.g. If 'dir_0' is in '.', 'dir_1' is in ./dir_0, and
        'dir_2' is in ./dir_0/dir_1, then dir_1 and dir_2 will be found.

    (5) dir/file refers to a file in the ./dir directory

    (6) dir/file refers to a directory in the ./dir directory
        
    (7) If start_path is 'dir_*', then the first dir in each of the format
        strings must match the pattern. e.g. If 'dir', 'dir1', 'dir_2', 'dir_3'
        are in ./, then dir/dir/.../dir is dir_*/dir/.../dir and only 
        dir_2 and dir_3 match.

    (8) If name_search_str is 'f*', then file in each of the format
        strings must match the pattern. e.g. If 'f1.txt', 'f2.txt', 'log.txt'
        are in ./dir/dir, then dir/dir/.../dir is dir/dir/f* and only 
        f1.txt and f2.txt match.

    Params:
    ''     , ''  , True  | '*'    , ''  , True  | '*'    , '.' , True
    'dir_*', ''  , True  | 'dir_*', '.' , True  | 'dir_*', '.' , True
    Behavior:
    no , yes, no , yes, no , yes, no
    dir, dir/dir/.../dir

    Params:
    '.'    , ''  , True  | ''     , '.' , True  | '.'    , '.' , True
    Behavior:
    no , yes, no , yes, no , yes, yes
    dir, dir/dir/.../dir

    Params:
    ''     , '*' , True  | '.'    , '*' , True  | ''     , None, True
    Behavior:
    yes, yes, yes, yes, yes, yes, no
    file, dir, dir/dir/.../file, dir/dir/.../dir

    Params:
    '.'    , None, True
    Behavior:
    yes, yes, yes, yes, yes, yes, yes
    file, dir, dir/dir/.../file, dir/dir/.../dir

    Params:
    '*'    , '*' , True  | 'dir_*', '*' , True  | 'dir_*', 'f*', True
    Behavior:
    no , no , yes, yes, yes, yes, no
    dir/dir/.../file, dir/dir/.../dir

    Params:
    '*'    , None, True
    Behavior:
    no , yes, yes, yes, yes, yes, no
    dir, dir/dir/.../file, dir/dir/.../dir

    Params:
    ''     , ''  , False
    Behavior:
    no , no , no , no , no , no , no
    []

    Params:
    '.'    , ''  , False | ''     , '.' , False | '.'    , '.' , False
    Behavior:
    no , no , no , no , no , no , yes
    ['.']

    Params:
    '*'    , ''  , False | '*'    , '.' , False | 'dir_*', ''  , False
    'dir_*', '.' , False
    Behavior:
    no , yes , no , no , no , no , no
    dir

    Params:
    ''     , '*' , False | '.'    , '*' , False | ''     , None, False
    '.'    , None, False
    Behavior:
    yes, yes, no , no , no , no , no
    file, dir

    Params:
    '*'    , '*' , False | '*'    , None, False | 'dir_*', '*' , False
    'dir_*', None, False | 'dir_*', 'f*', False
    Behavior:
    no , no , yes, yes, no , no , no
    dir/file, dir/dir

    Return:
    path_lst: list of str or []
        List of paths that originate from directories matching the start_path
        pattern and match the name_search_str pattern and recursively according
        to recursive and format according to clean_format and absolute_paths.
        Includes files if find_files and directories if find_dirs.

    Purpose: Similar to the bash find function. Find a list of paths that
            originate from directories matching the pattern(s) in start_paths 
            and match the name_search_str pattern and recursively according to
            recursive. Clean up path names or provide absolute paths if
            desired.

    Methodology: Use the glob.glob function.
    '''
    if type(start_paths) is str:
        start_paths = [start_paths]
    path_lsts = []
    for start_path in start_paths:
        if name_search_str is None:
            # When name_search_str is not None, we want to find all files and dirs
            # in dirs matching start_path pattern either only in those matching
            # dirs if recursive is False or in all sub-directories of those 
            # matching too if recursive is True.
            if recursive:
                # The ** notation combined with recursive=True is glob's fancy
                # notation for searching all directories matching start_path
                # recursively.
                path_lst = glob(os.path.join(start_path, '**'), recursive=True)
            else:
                # recursive is False so just get all files and dirs in the first
                # level of dirs matching the start_path pattern.
                path_lst = glob(os.path.join(start_path, '*'))
        else:
            # When name_search_str is not None, we want to find all files and dirs
            # matching name_search_str pattern in dirs matching start_path pattern
            # either only in those matching dirs if recursive is False or in all
            # sub-directories of those matching too if recursive is True.
            if recursive:
                # The ** notation combined with recursive=True is glob's fancy
                # notation for searching all directories matching start_path
                # recursively.
                path_lst = glob(os.path.join(start_path, '**', name_search_str), \
                                recursive=True)
            else:
                # recursive is False so just get all files and dirs matching
                # the name_search_str pattern in the first level of dirs matching
                # the start_path pattern.
                path_lst = glob(os.path.join(start_path, name_search_str))
        if not find_files:
            # If we don't want to return files, remove any file paths found.
            path_lst = [path for path in path_lst if not os.path.isfile(path)]
        if not find_dirs:
            # If we don't want to return dirs, remove any dir paths found.
            path_lst = [path for path in path_lst if not os.path.isdir(path)]
        if absolute_paths:
            # Convert all paths to absolute paths.
            path_lst = [os.path.abspath(path) for path in path_lst]
        elif clean_format:
            # If absolute_paths, no need to clean b/c they will already be clean.
            # Clean paths so they don't have './' prepended or '/.' or '/' appended
            from python_utils.file_utils import format_all_paths_cleanly
            path_lst = format_all_paths_cleanly(path_lst)
        path_lsts += path_lst
    return path_lsts


def grep_dir_recursively(search_str, dir_path, read_mode, case_sensitive):
    from python_utils.file_utils import grep_single_file
    found_lines = []
    found_line_nums = []
    found_fpaths = []
    for sub_path in find(dir_path, recursive=True):
        if not os.path.isdir(sub_path):
            found_result, found_result_line_nums, found_result_fpaths = grep_single_file(search_str, sub_path, read_mode, case_sensitive)
            found_lines += found_result
            found_line_nums += found_result_line_nums
            found_fpaths += found_result_fpaths
    return found_lines, found_line_nums, found_fpaths
    
def write_row_to_csv(path, one_dimensional_list, mode='a', delimiter=','):
    if type(one_dimensional_list) != list:
        raise TypeError('row is not type list, cannot write to csv. type(one_dimensional_list):', \
              type(one_dimensional_list), 'one_dimensional_list:', one_dimensional_list)
    if 'b' == mode[-1]:
        mode = mode[:-1]
    with open(path, mode, newline='') as f:
        csvWriter = csv.writer(f, delimiter=delimiter)
        csvWriter.writerow(one_dimensional_list)
        
def write_rows_to_csv(path, two_Dimensional_list, mode='w', delimiter=','):

    if 'b' == mode[-1]:
        mode = mode[:-1]
    f = open(path, mode, newline='')

    csvWriter = csv.writer(f, delimiter=delimiter)
    for row in two_Dimensional_list:
        if type(row) is not list:
            raise TypeError('row is not type list, cannot write to csv. The type of row is ' + str(type(row)), 'row', row)
        csvWriter.writerow(row)
    f.close()
