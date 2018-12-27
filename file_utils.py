
"""
Created on Tue Feb  6 19:16:48 2018

@author: timcrose
"""

import csv, json, os, sys, pickle
from glob import glob
import shutil, fnmatch, random
import time_utils
import platform
import numpy as np

python_version = float(platform.python_version()[:3])
if python_version >= 3.0:
    from file_utils3 import *
elif python_version >= 2.0:
    from file_utils2 import *
else:
    print('python version below 2.0, potential for some unsupported functions')

def write_pickle(fpath, data):
    '''
    fpath: str
        file path of pickle file (must include .pickle ext)
    data: python obj
        object to be pickled

    Purpose: pickle a python object and store in
         a .pickle file.
    '''
    with open(fpath, "wb") as pickle_out:
        pickle.dump(data, pickle_out)

def read_pickle(fname):
    '''
    fname: str
        filename of pickle file (must include .pickle ext)

    Return: unpickled object

    Purpose: unpickle a python object that was stored in
         a .pickle file.
    '''

    if not fname.endswith('.pickle'):
        print('filename must end with .pickle. fname: ', fname)
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

def grep_str(search_str, path, read_mode, found_lines):
    if type(path) is str:
        if os.path.isdir(path):
            found_lines = grep_dir_recursively(search_str, path, read_mode, found_lines)
        elif os.path.isfile(path):
            found_lines = grep_single_file(search_str, path, read_mode, found_lines)
        else:
            print('path DNE: ', path)
    else:
        print('cannot handle non-string path: ', path)
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
        print('could not interpret path as str or iterable. paths: ', paths)
    return found_lines

def get_lines_of_file(fname, mode='r'):
    with open(fname, mode) as f:
        lines = f.readlines()
    return lines

def write_lines_to_file(fpath, lines, mode='w'):
    with open(fpath, mode) as f:
        f.writelines(lines)

def mkdir_if_DNE(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def cp_str_src(src_path, dest_dir, dest_fname):
    if type(src_path) is str:
        src_match_paths = glob(src_path)
        for src_match_path in src_match_paths:
            if os.path.isdir(src_match_path):
                rm(dest_dir)
                shutil.copytree(src_match_path, dest_dir)
            elif os.path.isfile(src_match_path):
                mkdir_if_DNE(dest_dir)
                shutil.copy(src_match_path, os.path.join(dest_dir, dest_fname))
            else:
                print('cannot handle src input. src_path is ', src_path, 'src_match_path:', src_match_path)
                if not os.path.exists(src_match_path):
                    print('src_match_path: ' + src_match_path + ' DNE')
    else:
        print('needed str input. src_path: ', src_path)

def cp(src_paths_list, dest_dir, dest_fname=''):
    if type(dest_dir) is not str:
        raise ValueError('destination path must be a str. dest_dir: ', dest_dir)
    if type(src_paths_list) is str:
        if '*' in src_paths_list:
            src_paths_list = glob(src_paths_list)
        else:
            cp_str_src(src_paths_list, dest_dir, dest_fname)
            return
    if not hasattr(src_paths_list, '__iter__'):
        raise TypeError('src must be of type str or iterable. src_paths_list: ', src_paths_list)
    for src_path in src_paths_list:
        cp_str_src(src_path, dest_dir, dest_fname)

def rm_str(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    else:
        print('path ' + path + ' DNE. Skipping.')

def rm(paths):
    if type(paths) is str:
        if '*' in paths:
            paths = glob(paths)
        else:
            rm_str(paths)
            return

    if not hasattr(paths, '__iter__'):
        raise ValueError('paths must be a string of one path or an iterable of paths which are strings. paths:', paths)
    for path in paths:
        if type(path) is not str:
            raise ValueError('path must be a string. path:', path)
        rm_str(path)

def mv(src_fpath, dest_fpath, dest_fname=''):
    #copy then delete
    if type(dest_fpath) is not str:
        raise IOError('Cannot move, destination path needs to be of type str. dest_fpath:', dest_fpath)
    cp(src_fpath, dest_fpath, dest_fname)
    rm(src_fpath)

def rms(paths):
    '''
    safe rm
    '''
    trash_path = os.path.join(os.environ('HOME'), 'trash')
    mkdir_if_DNE(trash_path)
    if type(paths) is str:
        mv(paths, trash_path)
    elif hasattr(paths, '__iter__'):
        for path in paths:
            mv(path, trash_path)
    else:
        raise ValueError('paths must be a string of one path or an iterable of paths which are strings. paths:', paths)

def read_csv(path,mode='r', map_type=None, dtype=None):
    if path[-4:] != '.csv':
        raise Exception('fname must have .csv extension. path:', path)

    red_csv = []

    if not os.path.exists(path):
        return red_csv

    with open(path, mode) as f:
        csv_reader = csv.reader(f)
        red_csv.extend(csv_reader)

    if map_type == 'float':
        red_csv = [list(map(float, row)) for row in red_csv]
    elif map_type == 'int':
        red_csv = [list(map(int, row)) for row in red_csv]

    if dtype is not None:
        red_csv = np.array(red_csv, dtype=dtype)

    return red_csv
    
def write_dct_to_json(path, dct, indent=4, dump_type='dump'):
    if path[-5:] != '.json':
        raise Exception('path must have .json extension. path:', path)

    if type(dct) != dict:
        raise TypeError('dct is not type dict, cannot write to json. type(dct):', type(dct))

    with open(path, 'w') as f:
        if dump_type == 'dump':
            json.dump(dct, f, indent=indent)
        elif dump_type == 'dumps':
            json.dumps(dct, f, indent=indent)

def get_dct_from_json(path, load_type='load'):
    if path[-5:] != '.json':
        raise Exception('path must have .json extension. path:', path)

    with open(path, 'r') as f:
        if load_type == 'load':
            dct = json.load(f)
        elif load_type == 'loads':
            dct = json.loads(dct, f)

    return dct

def write_to_file(fname, str_to_write, mode='w'):
    '''
    fname: str
        path to file including file name
    str_to_write: str
        str to write to the file
    mode: str
        valid modes include w for overwrite and a for append.
    
    Purpose: write a string to a file.
    '''
    with open(fname, mode=mode) as f:
        f.write(str_to_write)

def lock_file(fname):
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('locked')

def wait_for_file_to_vanish(fname, total_timeout=100000, time_frame=0.05):
    start_time = time_utils.gtime()
    #wait until a file is removed by some other process
    while os.path.exists(fname):
        #sleep a random amount of time to help prevent clashing (if multiple ranks)
        time_utils.sleep(random.uniform(time_frame, 24.0 * time_frame))
        if time_utils.gtime() - start_time > total_timeout:
            raise Exception('file ' + fname + ' still locked after a total of ' + str(total_timeout) + ' seconds')

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
        fsize = os.path.getsize(fpath)
        time_utils.sleep(time_frame)
        if time_utils.gtime() - start_time > total_timeout:
            raise Exception('file ' + fpath + ' still not done being written to after a total of ' + str(total_timeout) + ' seconds')

def fname_from_fpath(fpath, include_ext=False):
    '''
    fpath: str
        path to file
    include_ext: bool
        True: return basename
        False: return basename without extension

    return: str
        filename with or without extension
    Purpose: get the file name from the file path
    '''
    basename = os.path.basename(fpath)
    if include_ext:
        return basename
    return os.path.splitext(basename)[0]
