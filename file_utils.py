
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

def get_lines_of_file(fname, mode='r'):
    with open(fname, mode) as f:
        lines = f.readlines()
    return lines

def grep_single_file(search_str, fpath, read_mode, found_lines, return_list=True, search_from_top_to_bottom=True):
    lines = get_lines_of_file(fpath, mode=read_mode)
    if return_list:
        return [line for line in lines if search_str in line]
    else:
        if search_from_top_to_bottom:
            i = 0
            while i < len(lines):
                if search_str in lines[i]:
                    return True
                i += 1
        else:
            i = len(lines) - 1
            while i > -1:
                if search_str in lines[i]:
                    return True
                i -= 1
    return False
            

def grep_str(search_str, path, read_mode, found_lines, return_list=True, search_from_top_to_bottom=True, fail_if_DNE=False, verbose=False):
    if type(path) is str:
        if os.path.isdir(path):
            found_result = grep_dir_recursively(search_str, path, read_mode, found_lines, return_list=return_list, search_from_top_to_bottom=search_from_top_to_bottom)
        elif os.path.isfile(path):
            found_result = grep_single_file(search_str, path, read_mode, found_lines, return_list=return_list, search_from_top_to_bottom=search_from_top_to_bottom)
        else:
            if not fail_if_DNE:
                if verbose:
                    print('path DNE: ', path)
                if return_list:
                    return []
                else:
                    return False
            else:
                raise FileNotFoundError('path: ' + path + ' DNE')
    elif verbose:
        print('cannot handle non-string path: ', path)
    if return_list:
        return found_lines + found_result
    else:
        return found_result

def grep(search_str, paths, read_mode='r', return_list=True, search_from_top_to_bottom=True, fail_if_DNE=False, verbose=False):
    found_lines = []
    if type(paths) is str:
        path = paths
        found_result = grep_str(search_str, path, read_mode, found_lines, return_list=return_list, search_from_top_to_bottom=search_from_top_to_bottom, fail_if_DNE=fail_if_DNE, verbose=verbose)
        if return_list:
            found_lines += found_result
        else:
            found_lines = found_result
    elif hasattr(paths, '__iter__'):
        for path in paths:
            found_result = grep_str(search_str, path, read_mode, found_lines, return_list=return_list, search_from_top_to_bottom=search_from_top_to_bottom, fail_if_DNE=fail_if_DNE, verbose=verbose)
            if return_list:
                found_lines += found_result
            elif found_result:
                return True
    elif verbose:
        print('could not interpret path as str or iterable. paths: ', paths)
    return found_lines

def read_file(fpath, mode='r'):
    with open(fpath, mode) as f:
        file_contents = f.read()
    return file_contents

def write_lines_to_file(fpath, lines, mode='w'):
    with open(fpath, mode) as f:
        f.writelines(lines)

def mkdir_if_DNE(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def cp_str_src(src_path, dest_dir, dest_fname, fail_if_cant_rm=False, verbose=True, replace_dest_dir=True):
    if type(src_path) is str:
        src_match_paths = glob(src_path)
        for src_match_path in src_match_paths:
            if os.path.isdir(src_match_path):
                if replace_dest_dir:
                    rm(dest_dir, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)
                else:
                    dest_dir = os.path.join(dest_dir, os.path.basename(src_match_path))
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

def cp(src_paths_list, dest_dir, dest_fname='', fail_if_cant_rm=False, verbose=True, replace_dest_dir=True):
    if type(dest_dir) is not str:
        raise ValueError('destination path must be a str. dest_dir: ', dest_dir)
    if type(src_paths_list) is str:
        if '*' in src_paths_list:
            src_paths_list = glob(src_paths_list)
        else:
            cp_str_src(src_paths_list, dest_dir, dest_fname, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose, replace_dest_dir=replace_dest_dir)
            return
    if not hasattr(src_paths_list, '__iter__'):
        raise TypeError('src must be of type str or iterable. src_paths_list: ', src_paths_list)
    for src_path in src_paths_list:
        cp_str_src(src_path, dest_dir, dest_fname, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose, replace_dest_dir=replace_dest_dir)

def rm_str(path, fail_if_cant_rm=False, verbose=True):
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            if not fail_if_cant_rm:
                print('path ' + path + ' DNE. Skipping.')
            else:
                raise OSError('path ' + path + ' DNE unexpectedly')
    elif os.path.exists(path):
        try:
            os.remove(path)
        except:
            if not fail_if_cant_rm:
                print('path ' + path + ' DNE. Skipping.')
            else:
                raise OSError('path ' + path + ' DNE unexpectedly')
    elif fail_if_cant_rm:
        raise OSError('cannot rm because path DNE :' + path)
    elif verbose:
        print('path ' + path + ' DNE. Skipping.')

def rm(paths, fail_if_cant_rm=False, verbose=True):
    if type(paths) is str:
        if '*' in paths:
            paths = glob(paths)
        else:
            rm_str(paths, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)
            return

    if not hasattr(paths, '__iter__'):
        raise ValueError('paths must be a string of one path or an iterable of paths which are strings. paths:', paths)
    for path in paths:
        if type(path) is not str:
            raise ValueError('path must be a string. path:', path)
        rm_str(path, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)

def mv(src_fpath, dest_fpath, dest_fname='', fail_if_cant_rm=False, verbose=True, replace_dest_dir=True):
    #copy then delete
    if type(dest_fpath) is not str:
        raise IOError('Cannot move, destination path needs to be of type str. dest_fpath:', dest_fpath)
    cp(src_fpath, dest_fpath, dest_fname, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose, replace_dest_dir=replace_dest_dir)
    rm(src_fpath, fail_if_cant_rm=False, verbose=True)

def rms(paths, fail_if_cant_rm=False, verbose=True):
    '''
    safe rm
    '''
    trash_path = os.path.join(os.environ['HOME'], 'trash')
    mkdir_if_DNE(trash_path)
    if type(paths) is str:
        mv(paths, trash_path, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)
    elif hasattr(paths, '__iter__'):
        for path in paths:
            mv(path, trash_path, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)
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

def lock_file(fname, total_timeout=100000, time_frame=0.05, go_ahead_if_out_of_time=False):
    wait_for_file_to_vanish(fname, total_timeout=total_timeout, time_frame=time_frame,  go_ahead_if_out_of_time=go_ahead_if_out_of_time)
    with open(fname, 'w') as f:
        f.write('locked')

def wait_for_file_to_vanish(fname, total_timeout=100000, time_frame=0.05, go_ahead_if_out_of_time=False):
    start_time = time_utils.gtime()
    #wait until a file is removed by some other process
    while os.path.exists(fname):
        #sleep a random amount of time to help prevent clashing (if multiple ranks)
        time_utils.sleep(random.uniform(time_frame, 24.0 * time_frame))
        if time_utils.gtime() - start_time > total_timeout:
            if go_ahead_if_out_of_time:
                return
            else:
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
