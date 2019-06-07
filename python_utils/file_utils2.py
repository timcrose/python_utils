
"""
Created on Tue Feb  6 19:16:48 2018

@author: timcrose
"""

import csv, json, os, sys, pickle
from glob import glob
import shutil, fnmatch
from python_utils import time_utils

def glob2(start_path, pattern, recursive=True):
    '''
    Purpose: recursively get files like recursive glob but for python2
    '''
    if not recursive:
        return glob(os.path.join(start_path, pattern))
    matches = []
    for root, dirnames, filenames in os.walk(start_path):
        for filename in fnmatch.filter(filenames, pattern):
           matches.append(os.path.join(root, filename))
    return matches

def output_from_rank(message_args, rank, mode='ab', output_fpath_prefix='output_from_world_rank_'):
    output_fpath = output_fpath_prefix + str(rank)
    with open(output_fpath, mode=mode) as f:
          print >> f, message_args

def grep_dir_recursively(search_str, dir_path, read_mode):
    from file_utils import grep_single_file
    found_lines = []
    found_line_nums = []
    found_fpaths = []
    for sub_path in glob2(dir_path, '*'):
        if not os.path.isdir(sub_path):
            found_result, found_result_line_nums, found_result_fpaths = grep_single_file(search_str, sub_path, read_mode)
            found_lines += found_result
            found_line_nums += found_result_line_nums
            found_fpaths += found_result_fpaths
    return found_lines, found_line_nums, found_fpaths

def write_row_to_csv(path, one_dimensional_list, mode='ab', delimiter=','):
    if path[-4:] != '.csv':
        raise Exception('path must have .csv extension. path:', path)

    if type(one_dimensional_list) != list:
        raise TypeError('row is not type list, cannot write to csv. type(one_dimensional_list):', \
              one_dimensional_list, 'one_dimensional_list:', one_dimensional_list)

    with open(path, mode) as f:
        csvWriter = csv.writer(f, delimiter=delimiter)
        csvWriter.writerow(one_dimensional_list)
        
def write_rows_to_csv(path, two_Dimensional_list, mode='wb', delimiter=','):

    f = open(path, mode)
    csvWriter = csv.writer(f, delimiter=delimiter)
    for row in two_Dimensional_list:
        if type(row) != list:
            raise TypeError('row is not type list, cannot write to csv. The type of row is ' + str(type(row)), 'row:', row)
        csvWriter.writerow(row)
    f.close()