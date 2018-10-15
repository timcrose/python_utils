
"""
Created on Tue Feb  6 19:16:48 2018

@author: timcrose
"""

import csv, json, os, sys, pickle
from glob import glob
import shutil, fnmatch
import time_utils

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

def read_pickle(fname):
    '''
    fname: str
        filename of pickle file (must include .pickle ext)

    Return: unpickled object

    Purpose: unpickle a python object that was stored in
         a .pickle file.
    '''

    if not fname.endswith('.pickle'):
        print('filename must end with .pickle. Filename gotten: ' + fname)
        print('Returning None')
        return None

    with open(fname, 'rb') as f:
        python_obj = pickle.load(f)
    return python_obj

def grep_single_file(search_str, fpath, read_mode, found_lines):
    with open(fpath, read_mode) as f:
        lines = f.readlines()
    for line in lines:
         if search_str in line:
             found_lines.append(line)
    return found_lines

def grep_dir_recursively(search_str, dir_path, read_mode, found_lines):
    for sub_path in glob2(dir_path, '*'):
        if not os.path.isdir(sub_path):
            found_lines = grep_single_file(search_str, sub_path, read_mode, found_lines)
    return found_lines

def grep_str(search_str, path, read_mode, found_lines):
    if type(path) is str:
        if os.path.isdir(path):
            found_lines = grep_dir_recursively(search_str, path, read_mode, found_lines)
        elif os.path.isfile(path):
            found_lines = grep_single_file(search_str, path, read_mode, found_lines)
        else:
            print('path DNE: ' + path)
    else:
        print('cannot handle non-string path: ' + path)
    return found_lines

def grep(search_str, paths, read_mode='r'):
    found_lines = []
    if type(paths) is str:
        path = paths
        found_lines = grep_str(search_str, path, read_mode, found_lines)
    elif hasattr(paths, '__iter__'):
        for path in paths:
            found_lines = grep_str(search_str, path, read_mode, found_lines)
    else:
        print('could not interpret path as str or iterable: ' + paths)
    return found_lines

def get_lines_of_file(fname, mode='r'):
    with open(fname, mode) as f:
        lines = f.readlines()
    return lines

def mkdir_if_DNE(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def cp_str_src(src_path, dest_dir, dest_fname):
    if type(src_path) is str:
        if os.path.isdir(src_path):
            rm(dest_dir)
            shutil.copytree(src_path, dest_dir)
        elif os.path.isfile(src_path):
            mkdir_if_DNE(dest_dir)
            shutil.copy(src_path, os.path.join(dest_dir, dest_fname))
        else:
            print('cannot handle src input. src_path is ', src_path)
            if not os.path.exists(src_path):
                print('src path: ' + src_path + ' DNE')
    else:
        print('needed str input. src_path: ', src_path)

def cp(src_paths_list, dest_dir, dest_fname=''):
    if type(dest_dir) is not str:
        raise ValueError('destination path must be a str. dest_dir: ', dest_dir)
    if type(src_paths_list) is str:
        cp_str_src(src_paths_list, dest_dir, dest_fname)
        return
    if not hasattr(src_paths_list, '__iter__'):
        raise TypeError('src must be of type str or iterable. src_paths_list: ', src_paths_list)
    for src_path in src_paths_list:
        cp_str_src(src_path, dest_dir, dest_fname)

def rm(paths):
    if type(paths) is str:
        if os.path.isdir(paths):
            shutil.rmtree(paths)
        elif os.path.exists(paths):
                os.remove(paths)
        return

    if not hasattr(paths, '__iter__'):
        raise ValueError('paths must be a string of one path or an iterable of paths which are strings. paths:', paths)
    for path in paths:
        if type(path) is not str:
            raise ValueError('path must be a string. path:', path)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        else:
            print('path ' + path + ' DNE. Skipping.')

def mv(src_fpath, dest_fpath, dest_fname=''):
    #copy then delete
    if type(dest_fpath) is not str:
        raise IOError('Cannot move, destination path needs to be of type str. dest_fpath:', dest_fpath)
    cp(src_fpath, dest_fpath, dest_fname)
    rm(src_fpath)

def read_csv(path,mode='r'):
    if path[-4:] != '.csv':
        raise Exception('fname must have .csv extension')

    red_csv = []

    if not os.path.exists(path):
        return red_csv

    with open(path, mode) as f:
        csv_reader = csv.reader(f)
        red_csv.extend(csv_reader)

    return red_csv
    
def write_row_to_csv(path, one_dimensional_list, mode='ab', delimiter=','):
    if path[-4:] != '.csv':
        raise Exception('path must have .csv extension')

    if type(one_dimensional_list) != list:
        raise TypeError('row is not type list, cannot write to csv')

    with open(path, mode) as f:
        csvWriter = csv.writer(f, delimiter=delimiter)
        csvWriter.writerow(one_dimensional_list)
        
def write_rows_to_csv(path, two_Dimensional_list, mode='wb', delimiter=','):
    if path[-4:] != '.csv':
        raise TypeError('path must have .csv extension. The path you gave was '  + path)

    f = open(path, mode)
    csvWriter = csv.writer(f, delimiter=delimiter)
    for row in two_Dimensional_list:
        if type(row) != list:
            raise TypeError('row is not type list, cannot write to csv. The type of row is ' + str(type(row)))
        csvWriter.writerow(row)
    f.close()

def write_dct_to_json(path, dct, indent=4, dump_type='dump'):
    if path[-5:] != '.json':
        raise Exception('path must have .json extension')

    if type(dct) != dict:
        raise TypeError('dct is not type dict, cannot write to json')

    with open(path, 'w') as f:
        if dump_type == 'dump':
            json.dump(dct, f, indent=4)
        elif dump_type == 'dumps':
            json.dumps(dct, f, indent=4)

def get_dct_from_json(path, load_type='load'):
    if path[-5:] != '.json':
        raise Exception('path must have .json extension')

    with open(path, 'r') as f:
        if load_type == 'load':
            dct = json.load(f)
        elif load_type == 'loads':
            dct = json.loads(dct, f)

    return dct

def lock_file(fname):
    with open(fname, 'w') as f:
        f.write('locked')

def wait_for_file_to_vanish(fname):
    #wait until a file is removed by some other process
    while os.path.exists(fname):
        #sleep a random amount of time to help prevent clashing (if multiple ranks)
        time_utils.sleep(random.uniform(0.2, 1.2))

def wait_for_file_to_exist_and_written_to(fpath, total_timeout=100000, time_frame=0.05):
    '''
    fpath: str
        path to file to check
    total_timeout: number
        total number of seconds before aborting the wait command
    time_frame: number
        number of seconds to wait between each check of file size.
    Purpose: Wait until file exists and the filesize remains constant in
        a given time frame.
    '''
    start_time = time_utils.gtime()
    while not os.path.exists(fpath):
        if time_utils.gtime() - start_time > total_timeout:
            raise Exception('file ' + fpath + ' still DNE after a total of ' + str(total_timeout) + ' seconds')
    fsize = os.path.getsize(fpath)
    time_utils.sleep(time_frame)
    while fsize != os.path.getsize(fpath):
        time_utils.sleep(time_frame)
        if time_utils.gtime() - start_time > total_timeout:
            raise Exception('file ' + fpath + ' still not done being written to after a total of ' + str(total_timeout) + ' seconds')
