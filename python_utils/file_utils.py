
"""
Created on Tue Feb  6 19:16:48 2018

@author: timcrose
"""

import csv, json, os, sys, time
import dill
try:
    import h5py
except:
    pass
from glob import glob
import shutil, fnmatch, random
from python_utils import time_utils, err_utils, list_utils
import platform
import numpy as np
import pandas as pd
from copy import deepcopy
import socket
python_version = float(platform.python_version()[:3])
if python_version >= 3.0:
    from python_utils.file_utils3 import *
elif python_version >= 2.0:
    from python_utils.file_utils2 import *
else:
    print('python version below 2.0, potential for some unsupported ' +
        'functions')


def write_pickle(fpath, data, err_message='Failed to write pickle', fail_gracefully=False, verbose=False):
    '''
    pickle a python object and store in a .pickle file with path fpath using 
    pickle.dump().
    
    Parameters
    ----------
    fpath: str
        file path of pickle file (must include .pickle ext).
        
    data: python obj
        object to be pickled

    fail_gracefully: bool
        True: Return None but do not raise an error to abort if something 
            goes wrong in this function. You should only set this to True
            if you know that failing won't cause problems later on in 
            your program - i.e. you can fully recover.
        False: Raise an error to abort the program if something goes
            wrong in this function.

    verbose: bool
        True: Print informative statements, if any.
        False: Do not print any non-crucial statements

    Returns
    -------
    None

    Notes
    -----
    1. The .pickle file will be loadable by pickle.load()
    '''
    # Write the python object data to the file with path fpath
    try:
        with open(fpath, "wb") as pickle_out:
            dill.dump(data, pickle_out)
    except Exception as e:
        err_utils.handle_error(e=e, err_message=err_message, 
fail_gracefully=fail_gracefully, verbose=verbose)
        
        return None


def read_pickle(fpath, fail_gracefully=True, verbose=False):
    '''
    Unpickle a python object that was stored in a .pickle file.
    
    Parameters
    ----------
    fpath: str
        file path of pickle file (must include .pickle extension)

    fail_gracefully: bool
        True: Return None but do not raise an error to abort if something 
            goes wrong in this function. You should only set this to True
            if you know that failing won't cause problems later on in 
            your program - i.e. you can fully recover.
        False: Raise an error to abort the program if something goes
            wrong in this function.

    verbose: bool
        True: Print informative statements, if any.
        False: Do not print any non-crucial statements

    Returns
    -------
    pickle_contents: Any
        The unpickled object. None if could not unpickle it.

    Notes
    -----
    1. The .pickle file must be loadable by pickle.load()
    '''

    # Make sure fpath has a .pickle extension
    if not fpath.endswith('.pickle'):
        err_utils.handle_error(
err_message='file path must end with .pickle. fpath: ' + fpath, 
fail_gracefully=fail_gracefully, verbose=verbose)

        return None

    # Load the pickled python object into python_obj
    try:
        with open(fpath, 'rb') as f:
            python_obj = dill.load(f)
        return python_obj
    except Exception as e:
        err_utils.handle_error(e=e, 
err_message='Could not load the file at fpath =  ' + fpath, 
fail_gracefully=fail_gracefully, verbose=verbose)

        return None


def get_lines_of_file(fpath, mode='r', timeout=1.5, interval_delay=0.5):
    '''
    Read a file and return its contents in the form of a list of strings where
    each string is a line in the file.

    Parameters
    ----------
    fpath: str
        File path to read.
    
    mode: str
        File open mode. 'r' for read and 'rb' for read bytes.
        
    timeout: Scalar
        How long in seconds to wait for the file open command to work before 
        giving up and throwing an error.
        
    interval_delay: Scalar
        How long in seconds to wait in between attempts to open the file.

    Returns
    -------
    lines: Str_List
        The file contents in the form of a list of strings where each string is
        a line in the file.
    '''
    f = open_file(fpath, mode=mode, timeout=timeout, interval_delay=interval_delay)
    lines = f.readlines()
    f.close()
    return lines


def grep_single_file(search_str, fpath, read_mode, verbose=False, case_sensitive=True):
    '''
    if case_sensitive:
        search_str = search_str.lower()
        file_str = read_file(fpath, mode=read_mode).lower()
        lines = file_str.split('\n')
        lst = [[i,line + '\n'] for i,line in enumerate(lines) if search_str in line]
    else:
        try:
            lines = get_lines_of_file(fpath, mode=read_mode)
        except UnicodeDecodeError:
            if verbose:
                print('Warning, UnicodeDecodeError encountered by grep_single_file for fpath', fpath)
            lines = []
        lst = [[i,line] for i,line in enumerate(lines) if search_str in line]
    '''
    file_str = read_file(fpath, mode=read_mode)
    lines = file_str.split('\n')
    if case_sensitive:
        lst = [[i,line] for i,line in enumerate(lines) if search_str in line]
    else:
        file_str_lower = file_str.lower()
        search_str_lower = search_str.lower()
        lines_lower = file_str_lower.split('\n')
        lst = [[i,lines[i] + '\n'] for i,line_lower in enumerate(lines_lower) if search_str_lower in line_lower]
    if len(lst) > 0:
        found_result_line_nums, found_result = list(zip(*lst))
    else:
        found_result_line_nums, found_result = [], []
    found_result_fpaths = [fpath] * len(found_result)
    return found_result, found_result_line_nums, found_result_fpaths
            

def grep_str(search_str, path, read_mode, fail_if_DNE=False, verbose=False, case_sensitive=True):
    if type(path) is str:
        if os.path.isdir(path):
            return grep_dir_recursively(search_str, path, read_mode, case_sensitive)
        elif os.path.isfile(path):
            return grep_single_file(search_str, path, read_mode, verbose, case_sensitive=case_sensitive)
        
    if not fail_if_DNE:
        if verbose:
            print('path DNE: ', path)
        return [], [], []
    else:
        raise FileNotFoundError('path DNE: ', path)


def grep(search_str, paths, read_mode='r', fail_if_DNE=False, verbose=False, return_line_nums=False, return_fpaths=False, case_sensitive=True):
    found_lines = []
    found_line_nums = []
    found_fpaths = []
    if type(paths) is str:
        paths = [paths]         

    if hasattr(paths, '__iter__'):
        for path in paths:
            found_result, found_result_line_nums, found_result_fpaths = grep_str(search_str, path, read_mode, fail_if_DNE=fail_if_DNE, verbose=verbose, case_sensitive=case_sensitive)
            found_lines += found_result
            found_line_nums += found_result_line_nums
            found_fpaths += found_result_fpaths

    elif not fail_if_DNE:
        if verbose:
            print('could not interpret path as str or iterable. paths: ', paths)
    else:
        raise FileNotFoundError('could not interpret path as str or iterable. paths: ', paths)

    if return_line_nums and return_fpaths:
        return found_lines, found_line_nums, found_fpaths
    elif return_line_nums and not return_fpaths:
        return found_lines, found_line_nums
    elif not return_line_nums and return_fpaths:
        return found_lines, found_fpaths
    else:
        return found_lines


def open_file(fpath, mode='r', timeout=1.5, interval_delay=0.5):
    '''
    The first attempt to open a file might be thwarted by some other process
    currently using the file. So, try multiple times.

    Parameters
    ----------
    fpath: str
        File path to read.
    
    mode: str
        File open mode. 'r' for read and 'rb' for read bytes.
        
    timeout: Scalar
        How long in seconds to wait for the file open command to work before 
        giving up and throwing an error.
        
    interval_delay: Scalar
        How long in seconds to wait in between attempts to open the file.

    Returns
    -------
    f: IO
        File handle of the opened file.
    '''
    start_time = time.perf_counter()
    exception = None
    while time.perf_counter() - start_time < timeout:
        try:
            f = open(fpath, mode)
            break
        except Exception as e:
            exception = e
            time.sleep(interval_delay)
    else:
        if exception is None:
            raise TimeoutError('Could not open the file within timeout=', timeout, 'seconds')
        raise exception
    return f


def read_file(fpath, mode='r', timeout=10, interval_delay=1):
    f = open_file(fpath, mode=mode, timeout=timeout, interval_delay=interval_delay)
    file_contents = f.read()
    f.close()
    return file_contents


def write_lines_to_file(fpath, lines, mode='w', timeout=1.5, interval_delay=0.5):
    f = open_file(fpath, mode=mode, timeout=timeout, interval_delay=interval_delay)
    f.writelines(lines)
    f.close()


def mkdir_if_DNE(path, fail_gracefully=True):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except Exception as e:
            if fail_gracefully:
                return
            raise e


def cp_str_src(src_path, dest_dir, dest_fname, fail_if_cant_rm=False, verbose=True, overwrite=True):
    if type(src_path) is str:
        src_match_paths = glob(src_path)
        for src_match_path in src_match_paths:
            if os.path.isdir(src_match_path):
                dest_path = os.path.join(dest_dir, os.path.basename(src_match_path))
                if os.path.isdir(dest_path):
                    if overwrite:
                        # shutil.copytree throws an error rather than overwriting by default, thus we're removing here
                        rm(dest_path, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)
                    else:
                        raise Exception('dest_dir', dest_dir, 'exists and overwrite == False so cannot copy', src_path)
                # shutil.copytree expects the final path be the second argument
                shutil.copytree(src_match_path, dest_path)
            elif os.path.isfile(src_match_path):
                if dest_fname == '':
                    dest_fname = os.path.basename(src_match_path)
                dest_fpath = os.path.join(dest_dir, dest_fname)
                if os.path.isfile(dest_fpath) and not overwrite:
                    raise Exception('dest_fpath', dest_fpath, 'exists and overwrite == False so cannot copy', src_path)
                else:
                    # shutil.copy overwrites by default
                    shutil.copy(src_match_path, dest_fpath)
            elif not os.path.exists(src_match_path):
                raise Exception('src_match_path: ' + src_match_path + ' DNE')
    else:
        raise Exception('needed str input. src_path: ', src_path)


def cp(src_paths_list, dest_dir, dest_fname='', fail_if_cant_rm=False, verbose=True, overwrite=True):
    if type(dest_dir) is not str:
        raise ValueError('destination path must be a str. dest_dir: ', dest_dir)
    mkdir_if_DNE(dest_dir)
    if type(src_paths_list) is str:
        if '*' in src_paths_list:
            src_paths_list = glob(src_paths_list)
        else:
            cp_str_src(src_paths_list, dest_dir, dest_fname, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose, overwrite=overwrite)
            return
    if not hasattr(src_paths_list, '__iter__'):
        raise TypeError('src must be of type str or iterable. src_paths_list: ', src_paths_list)
    for src_path in src_paths_list:
        cp_str_src(src_path, dest_dir, dest_fname, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose, overwrite=overwrite)


def rm_str(path, fail_if_cant_rm=False, verbose=True):
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            if (not fail_if_cant_rm) and verbose:
                print('path ' + path + ' DNE. Skipping.')
            else:
                raise OSError('path ' + path + ' DNE unexpectedly')
    elif os.path.exists(path):
        try:
            os.remove(path)
        except:
            if (not fail_if_cant_rm) and verbose:
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
        if '*' in path:
            sub_paths = glob(path)
            for sub_path in sub_paths:
                rm_str(sub_path, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)
        else:
            rm_str(path, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose)
            

def mv(src_paths_list, dest_dir, dest_fname='', fail_if_cant_rm=False, verbose=True, overwrite=True):
    if type(dest_dir) is not str:
        raise IOError('Cannot move, destination path needs to be of type str. dest_dir:', dest_dir)
    #copy then delete. This is the most robust because other methods aren't faster if src and dest are on a different disk. If
    # they are on the same disk, then os.rename is faster, but only works if src and dest are files (not directories).
    cp(src_paths_list, dest_dir, dest_fname=dest_fname, fail_if_cant_rm=fail_if_cant_rm, verbose=verbose, overwrite=overwrite)
    rm(src_paths_list, fail_if_cant_rm=False, verbose=True)


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


def read_csv(path, mode='r', map_type=None, 
dtype=None, timeout=1.5, interval_delay=0.5):
    red_csv = []

    if not os.path.exists(path):
        return red_csv
    f = open_file(path, mode=mode, timeout=timeout, interval_delay=interval_delay)
    csv_reader = csv.reader(f)
    red_csv.extend(csv_reader)
    f.close()
    if map_type == 'float':
        red_csv = [list(map(float, row)) for row in red_csv]
    elif map_type == 'int':
        red_csv = [list(map(int, row)) for row in red_csv]

    if dtype is not None:
        red_csv = np.array(red_csv, dtype=dtype)

    return red_csv

    
def write_dct_to_json(path, dct, indent=4, dump_type='dump', timeout=1.5, interval_delay=0.5):
    if path[-5:] != '.json':
        raise Exception('path must have .json extension. path:', path)

    if type(dct) != dict:
        raise TypeError('dct is not type dict, cannot write to json. type(dct):', type(dct))

    f = open_file(path, mode='w', timeout=timeout, interval_delay=interval_delay)
    if dump_type == 'dump':
        json.dump(dct, f, indent=indent)
    elif dump_type == 'dumps':
        json.dumps(dct, f, indent=indent)
    f.close()
    

def get_dct_from_json(path, load_type='load', timeout=1.5, interval_delay=0.5):
    if path[-5:] != '.json':
        raise Exception('path must have .json extension. path:', path)
    f = open_file(path, mode='r', timeout=timeout, interval_delay=interval_delay)
    if load_type == 'load':
        dct = json.load(f)
    elif load_type == 'loads':
        dct = json.loads(dct, f)
    f.close()
    return dct


def write_to_file(fpath, str_to_write, mode='w', timeout=1.5, interval_delay=0.5):
    '''
    fpath: str
        full path to file (relative or absolute)
    str_to_write: str
        str to write to the file
    mode: str
        valid modes include w for overwrite and a for append.
    
    Purpose: write a string to a file.
    '''
    f = open_file(fpath, mode=mode, timeout=timeout, interval_delay=interval_delay)
    f.write(str_to_write)
    f.close()
    
    
def write_intention_lock_file(dirname, name_search_str, hostname=socket.gethostname(), wait_to_write_file=True, timeout=100, total_timeout=1000, fail_gracefully=True):
    if not name_search_str.endswith('.lock'):
        raise Exception('Lock file path search string must end with .lock')
    start_time = time.perf_counter()
    lock_file_fpath = os.path.join(dirname, hostname + '_locked_files_matching_' + name_search_str)
    intention_fpaths = find(dirname, name_search_str, find_dirs=False)
    while len([intention_fpath for intention_fpath in intention_fpaths if not intention_fpath.startswith(hostname)]) > 0 and time.perf_counter() - start_time < total_timeout:
        loop_start_time = time.perf_counter()
        intention_fpaths = find(dirname, name_search_str, find_dirs=False)
        while len([intention_fpath for intention_fpath in intention_fpaths if not intention_fpath.startswith(hostname)]) > 0 and time.perf_counter() - start_time < total_timeout and time.perf_counter() - loop_start_time < timeout:
            if not wait_to_write_file:
                if fail_gracefully:
                    return False
                else:
                    raise Exception('The lock file already existed and so cannot continue.')
            time.sleep(0.05)
            intention_fpaths = find(dirname, name_search_str, find_dirs=False)
        if time.perf_counter() - loop_start_time >= timeout:
            if fail_gracefully:
                return False
            else:
                raise Exception('Ran out of time to wait for the lock file to vanish.')
        write_to_file(lock_file_fpath, 'locked at' + str(time.perf_counter()))
        time.sleep(0.1)
        intention_fpaths = find(dirname, name_search_str, find_dirs=False)
    if time.perf_counter() - start_time < total_timeout:
        return lock_file_fpath
    elif fail_gracefully:
        return False
    else:
        raise Exception(f'After {total_timeout} seconds, the lock file was still there.')


def lock_file(fpath, lockfile_message='locked', total_timeout=100000, time_frame=0.05, go_ahead_if_out_of_time=False, timeout=1.5, interval_delay=0.5):
    start_time = time_utils.gtime()
    wait_for_file_to_vanish(fpath, total_timeout=total_timeout, time_frame=time_frame,  go_ahead_if_out_of_time=go_ahead_if_out_of_time)
    read_lockfile_message = 'Nonelkjlkj'
    while read_lockfile_message != lockfile_message:
        f = open_file(fpath, mode='w', timeout=timeout, interval_delay=interval_delay)
        f.write(lockfile_message)
        f.close()
        time_utils.sleep(0.05)
        try:
            with open(fpath) as f:
                read_lockfile_message = f.read()
        except:
            pass
        if time_utils.gtime() - start_time > total_timeout and not go_ahead_if_out_of_time:
            raise Exception('Took longer than total_timeout =', total_timeout, 'seconds to acquire lock file.')
    

def wait_for_file_to_vanish(fpath, total_timeout=100000, time_frame=0.05, go_ahead_if_out_of_time=False):
    start_time = time_utils.gtime()
    if time_frame == 0:
        while os.path.exists(fpath):
            if time_utils.gtime() - start_time > total_timeout and not go_ahead_if_out_of_time:
                raise Exception('file ' + fpath + ' still exists after a total of ' + str(total_timeout) + ' seconds') 
        return   
    #wait until a file is removed by some other process
    while os.path.exists(fpath):
        #sleep a random amount of time to help prevent clashing (if multiple ranks)
        time_utils.sleep(random.uniform(time_frame, 1.1 * time_frame))
        if time_utils.gtime() - start_time > total_timeout and not go_ahead_if_out_of_time:
            raise Exception('file ' + fpath + ' still exists after a total of ' + str(total_timeout) + ' seconds')
        

def wait_for_file_to_exist(fpath, total_timeout=100000, time_frame=0.05):
    '''
    fpath: str
        path to file to check
    total_timeout: number
        total number of seconds before aborting the wait command
    time_frame: number
        number of seconds to wait between each check of file size.
    Purpose: Wait until file exists for up to total_timeout seconds.
    '''
    start_time = time_utils.gtime()
    while not os.path.exists(fpath):
        if time_utils.gtime() - start_time > total_timeout:
            raise Exception('file ' + fpath + ' still DNE after a total of ' + str(total_timeout) + ' seconds')
        time_utils.sleep(time_frame)


def wait_for_file_to_be_written_to(fpath, total_timeout=100000, time_frame=0.05):
    '''
    fpath: str
        path to file to check
    total_timeout: number
        total number of seconds before aborting the wait command
    time_frame: number
        number of seconds to wait between each check of file size.
    Purpose: Wait until a file that exists to have its filesize remains constant in
        a given time frame. It will not be constant if it is currently being written to.
    '''
    start_time = time_utils.gtime()
    while True:
        try:
            fsize = os.path.getsize(fpath)
            break
        except FileNotFoundError:
            pass
        if time_utils.gtime() - start_time > total_timeout:
            raise Exception('file ' + fpath + ' still not done being written to after a total of ' + str(total_timeout) + ' seconds')
    time_utils.sleep(time_frame)
    while fsize != os.path.getsize(fpath) and fsize != 0:
        fsize = os.path.getsize(fpath)
        time_utils.sleep(time_frame)
        if time_utils.gtime() - start_time > total_timeout:
            raise Exception('file ' + fpath + ' still not done being written to after a total of ' + str(total_timeout) + ' seconds')


def wait_for_file_to_exist_and_written_to(fpath, total_timeout=100000, time_frame=0.05):
    '''
    fpath: str
        path to file to check
    total_timeout: number
        total number of seconds before aborting the wait command
    time_frame: number
        number of seconds to wait between each check of file size.
    Purpose: Wait until file exists and the filesize remains constant in
        a given time frame. It will not be constant if it is currently being written to.
    '''
    wait_for_file_to_exist(fpath, total_timeout=total_timeout, time_frame=time_frame)
    wait_for_file_to_be_written_to(fpath, total_timeout=total_timeout, time_frame=time_frame)


def rm_file_with_message(fpath, message):
    while os.path.exists(fpath):
        try:
            with open(fpath) as f:
                read_message = f.read()
        except:
            return
        if read_message == message:
            rm(fpath)
        else:
            return
        time_utils.sleep(0.1)


def read_fragile_csv(fpath):
    wait_for_file_to_be_written_to(fpath, total_timeout=1000, time_frame=0.1)
    read_success = False
    start_time = time_utils.gtime()
    while not read_success:
        try:
            df = pd.read_csv(fpath)
            read_success = True
        except:
            time_utils.sleep(0.1)
        if time_utils.gtime() - start_time > 1000:
            raise Exception('Took more than 1000 seconds to try to read', fpath,'\nExpected the file to be existant and non-empty.')
    return df


def get_new_task(lockfile_fpath, incomplete_tasks_fpath):
    lockfile_message = str(int(time_utils.gtime() * 10000))
    lock_file(lockfile_fpath, lockfile_message=lockfile_message, total_timeout=1000, time_frame=0.1, go_ahead_if_out_of_time=False)
    tasks_df = read_fragile_csv(incomplete_tasks_fpath)
    if len(tasks_df.values[len(tasks_df) - 1]) > 0:
        task_id = tasks_df.values[len(tasks_df) - 1][0]
    else:
        rm(lockfile_fpath)
        return None
    tasks_df.drop(index=len(tasks_df) - 1, inplace=True)
    tasks_df.to_csv(incomplete_tasks_fpath, index=False)
    num_incomplete_tasks = len(tasks_df)
    del tasks_df
    rm_file_with_message(lockfile_fpath, lockfile_message)
    return task_id


def add_completed_task(lockfile_fpath, complete_tasks_fpath, task_id, intermediate_func=None, intermediate_args=[]):
    # Use lockfile for complete tasks to let me know this task_id was complete.
    lockfile_message = str(int(time_utils.gtime() * 10000))
    lock_file(lockfile_fpath, lockfile_message=lockfile_message, total_timeout=1000, time_frame=0.1, go_ahead_if_out_of_time=False)
    if os.path.exists(complete_tasks_fpath):
        tasks_df = read_fragile_csv(complete_tasks_fpath)
        tasks_df = tasks_df.append(pd.DataFrame({'task_id':[task_id]}))
    else:
        tasks_df = pd.DataFrame({'task_id':[task_id]})
    if intermediate_func is not None:
        intermediate_func(*intermediate_args)
    # Write to complete_tasks_fpath that this task is complete
    tasks_df.to_csv(complete_tasks_fpath, index=False)
    del tasks_df
    rm_file_with_message(lockfile_fpath, lockfile_message)
    
    
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


def replace_text_in_file(fpath, search_str, 
replacement_str=None, replacement_line=None, 
num_of_occurrences=-1, search_from_top_to_bottom=True, 
percent_range=None, line_range=None, 
overwrite=False):
    '''
    fpath: str
        file path to search and do the replacing on
    search_str: str
        This string is searched for in the file as a substring of a line in that file.
        This line on which this string is found will be replaced by replacement_line if not None, unless search_str is not None, and then search_str will be replaced by replacement_str
    replacement_str: str or None
        if replacement_line is None, use this to replace num_of_occurrences of search_str.
    replacement_line: str or None
        The line to replace the line that search_str was found on. If this is None, use replacement_str to replace num_of_occurrences of search_str.
    num_of_occurrences: int
        The first num_of_occurrences lines containing search_str will be replaced with replacement_line (in
        the direction specified by search_from_top_to_bottom). If num_of_occurrences is None then all lines
        containing search_str will be replaced with replacement_line. (Similar for search_str if it is not None)
    search_from_top_to_bottom: bool
        True: Iterate through the lines sequentially
        False: Iterate through the lines in reverse order
    percent_range: list of Scalar or None
        [lower line percent, upper line percent]
        e.g. if percent_range is [0.3,0.5], then only look at lines that are between
        30% and 50% of the file's lines. So, if the file has 100 lines, then only consider
        lines 30 through 50, inclusive.
    line_range: list of int or None
        [line index lower bound (inclusive), line index upper bound (exclusive)]
        
        If None, then use either all_lines or use line_range if not None.
    line_range: list of int or None
        [lower line number, upper line number]
        Only search between these line numbers in the file. If None, use percent_range if not None, else use all_lines.
        First line is 0.
        
    overwrite: bool
        True: The next len(replacement_str) characters after and including the location of search_str will be overwritten to be replacement_str (unless an end-line is found)
        False: search_str is removed and replacement_str is inserted where search_str was found.
        
    Return: None

    Purpose: Find a line in a specified file that contains a search string, then replace
        that line with replacement_line, or replace instances of search_str with replacement_str.

    Notes
    -----
    1. Put a newline character at the end of your replacement_line string if you wish one to be there.
    2. If both replacement_str and replacement_line are not None, then use replacement_str instead of replacement_line.
    '''
    if not search_from_top_to_bottom:
        reverse_search_str = search_str[::-1]
        reverse_replacement_str = replacement_str[::-1]
    # If the whole file is to be considered and you want to just find & replace a string, then just do the following if block.
    if (percent_range is None or percent_range == 1) and replacement_str is not None and not overwrite:
        with open(fpath, 'r') as file:
            data = file.read()
            if not search_from_top_to_bottom:
                data = data[::-1]
                data = data.replace(reverse_search_str, reverse_replacement_str, num_of_occurrences)
                data = data[::-1]
            else:
                data = data.replace(search_str, replacement_str, num_of_occurrences)
        with open(fpath, 'w') as file:
            file.write(data)
        return
    all_lines = get_lines_of_file(fpath)
    lines = None
    if not search_from_top_to_bottom:
        all_lines.reverse()
    if percent_range is not None:
        lines = all_lines[int(percent_range[0] * len(all_lines)) : int(percent_range[1] * len(all_lines))]
    if line_range is not None:
        lines = all_lines[line_range[0]: line_range[1]]
    if lines is None:
        lines = all_lines
    num_occurrences = 0
    for i,line in enumerate(lines):
        if num_of_occurrences != -1 and num_occurrences >= num_of_occurrences:
            break
        if search_str in line:
            if replacement_str is None:
                lines[i] = replacement_line
            elif overwrite:
                newline_i = lines[i].find('\n')
                if newline_i == -1:
                    last_i = len(lines[i]) - 1
                else:
                    last_i = newline_i - 1
                search_str_i = lines[i].find(search_str)
                num_chars_til_end = last_i - search_str_i + 1
                new_replacement_str = replacement_str[:num_chars_til_end]
                new_replacement_line = lines[i][:search_str_i] + new_replacement_str + lines[i][search_str_i + len(new_replacement_str):]
                lines[i] = deepcopy(new_replacement_line)
            else:
                lines[i] = lines[i].replace(search_str, replacement_str, n_occurrences=(-1 if num_of_occurrences == -1 else max(num_of_occurrences - num_occurrences, 0)))
            num_occurrences += 1
    if not search_from_top_to_bottom:
        lines.reverse()
    if line_range is not None:
        all_lines = np.array(all_lines)
        lines = np.array(lines)
        all_lines[int(percent_range[0] * len(all_lines)) : int(percent_range[1] * len(all_lines))] = lines
        write_lines_to_file(fpath, all_lines, mode='w')
        return
    if percent_range is not None:
        all_lines = np.array(all_lines)
        lines = np.array(lines)
        all_lines[int(percent_range[0] * len(all_lines)) : int(percent_range[1] * len(all_lines))] = lines
        write_lines_to_file(fpath, all_lines, mode='w')
        return
    write_lines_to_file(fpath, lines, mode='w')


def concatenate_files(flist, new_fpath, write_concatenated_file=True, return_lines=False):
    '''
    flist: non-string iterable
        list of files to concatenate

    new_fpath: str
        full or relative path including file name of the concatenated file.

    write_concatenated_file: bool
        Whether to write the concatenated lines to a file

    return_lines: bool
        Whether to return the concatenated lines

    return: None or list
        None if return_lines is False, list of lines of the concatenated files otherwise

    Purpose: Concatenate the contents of files with the first file in the list being at the top of the new file.
        Return the lines as a list if requested.
    '''
    all_lines = [get_lines_of_file(fpath) for fpath in flist]
    all_lines = list_utils.flatten_list(all_lines)
    if write_concatenated_file:
        write_lines_to_file(new_fpath, all_lines)
    if return_lines:
        return all_lines


def safe_np_load(npy_fpath, total_timeout=10000, time_frame=0.05, verbose=False, check_file_done_being_written_to=True):
    '''
    npy_fpath: str
        Path to file that is loadable by np.load()

    total_timeout: number
        total number of seconds before aborting the wait command

    time_frame: number
        number of seconds to wait between each check of file size.

    verbose: bool
        Whether to print some log info

    check_file_done_being_written_to: bool
        Whether to check file size to determine if the file is being written to
        and thus unsafe to load.

    Return: np.array
        The contents of npy_fpath as loaded by np.load()

    Purpose: Check to make sure file exists before loading it. If DNE, wait until
        it does exist or your timeout is reached.
    '''
    start_time = time_utils.gtime()
    if check_file_done_being_written_to:
        wait_for_file_to_exist_and_written_to(npy_fpath, total_timeout=total_timeout, time_frame=time_frame)
    else:
        wait_for_file_to_exist(npy_fpath, total_timeout=total_timeout, time_frame=time_frame)
    if verbose:
        print('took {} seconds to wait for file to exist and written to according to the function wait_for_file_to_exist_and_written_to'.format(time_utils.gtime() - start_time))
        start_time_load = time_utils.gtime()
    while time_utils.gtime() - start_time < total_timeout:
        try:
            npy = np.load(npy_fpath)
            if verbose:
                print('took {} seconds after file {} exists to load it'.format(time_utils.gtime() - start_time_load, npy_fpath))
            return npy
        except ValueError:
            time_utils.sleep(time_frame)
    raise TimeoutError('total_timeout was reached in save_np_load')


def format_path_cleanly(path):
    '''
    path: str
        Path of file or directory.

    Return:
    clean_path: str
        Path of file or directory now without ./ prepended or /. or / appended.
        ./ , . , and ./. are returned as .

    Purpose: Sometimes paths contain ./ prepended or /. or / appended but these
        might not be desirable. This function pretties up the format.
    
    Methodology:
    We will be removing prepended ./ or appended /. or / characters. If any
    one of these characters are removed, repeat the loop seeing if any 
    additional of these characters needs to be removed. This enables
    'dir/./.' to be returned as 'dir'. For each character, if it is not found
    (in an offending place) then increase a couner num_passes by 1. Thus, if an
    iteration of this loop removes none of these characters, then num_passes 
    will be 3 and we are done.
    '''

    # num_passes is the counter for the number of consecutive times we 'pass a 
    # check' which means we did not find a character in an offending place
    # (e.g. /. in the [-2:] position).
    num_passes = 0
    while num_passes != 3:
        # num_passes is reset at each iteration because we want every check to 
        # pass on an iteration of this loop.
        num_passes = 0
        # If len(path) is 1, then it will not need trimming.
        if len(path) > 1:
            if path[-1] == '/':
                # Remove appended /
                path = path[:-1]
            else:
                num_passes += 1
            if path[-2:] == '/.':
                # Remove appended /.
                path = path[:-2]
            else:
                num_passes += 1
            if path[:2] == './' and path != './':
                # Remove prepended ./
                path = path[2:]
            else:
                num_passes += 1
        else:
            # If len(path) is 1, then it will not need trimming.
            break
    return path


def format_all_paths_cleanly(path_lst):
    '''
    path_lst: list of str
        List of paths.

    Return:
    clean_paths: list of str
        List of paths with clean format which means no './' strings prepended
        and no '/' or '/.' strings appended.

    Purpose: Purpose: Sometimes paths contain ./ prepended or /. or / appended but these
        might not be desirable. This function pretties up the format for a list of 
        paths.
    '''
    return [format_path_cleanly(path) for path in path_lst]


def write_h5_file(h5_fpath, data, attrs_dct={}, dset_name=None, overwrite=True, fail_if_already_exists=False, verbose=False, include_write_check=False):
    '''
    h5_fpath: str
        Path to .h5 output file

    data: np.array (any shape)
        array to store in the h5 file

    attrs_dct: dict
        Each key value pair in attrs_dct will be an attribute of the dataset
        stored as dset.attrs[key] = value

    dset_name: str or None
        Name of dataset to reference when reading it later. If None, use the fname of the 
        supplied h5_fpath.

    overwrite: bool
        True: overwrite or create file.
        False: If DNE, create file. If exists, look to fail_if_already_exists

    fail_if_already_exists: bool
        True: If exists, raise error
        False: If exists and verbose, print that it already exists.

    verbose: bool
        True: If exists and fail_if_already_exists and verbose, print that it already exists.
        False: pass
    
    include_write_check: bool
        True: add a 'writing_complete' attribute to the attrs dict last. This enables
            reading this file and ensuring that the writing process was not
            abruptly terminated leaving an incomplete file.
        False: pass

    Return: None

    Purpose: Write an h5 file using h5py to contain a dataset supplied by
        data and attributes supplied by attrs_dct.
    '''
    if os.path.exists(h5_fpath):
        if not overwrite:
            if fail_if_already_exists:
                raise Exception('File', h5_fpath, 'already exists and you did not want to overwrite it.')
            elif verbose:
                print('File', h5_fpath, 'already exists and you did not want to overwrite it...skipping...')
                return
    if dset_name is None:
        dset_name = fname_from_fpath(h5_fpath)
    with h5py.File(h5_fpath, 'w') as hf:
        dset = hf.create_dataset(dset_name, data=data)
        for key in attrs_dct:
            dset.attrs[key] = attrs_dct[key]
        if include_write_check:
            hf.attrs['writing_complete'] = True


def read_h5_file(h5_fpath, row_start_idx=0, 
row_end_idx=None, col_start_idx=0, 
col_end_idx=None, dset_name=None, 
return_data=True, return_attrs=True, query_write_check=False,
fail_gracefully=False, verbose=False):
    '''
    h5_fpath: str
        Path to .h5 file to read

    dset_name: str or None
        Name of dataset to reference when reading it later. If None, use the fname of the 
        supplied h5_fpath.

    return_data: bool
        True: Return data contained in the dataset referred to by dset_name in h5_fpath.
        False: Return None in place of this data.

    return_attrs: bool
        True: Return a dictionary where keys are those in dset.attrs and values are dset.attrs[key]
        False: Return None in place of this dictionary.

    query_write_check: bool
        True: Check to make sure the file exists and was successfully written to via the 'writing_complete' attribute
            of the root level attrs dict. If writing was complete, then return data, dset attrs if requested.
            If writing was not complete, raise an exception if not fail_gracefully. Else, return False, False if 
            either return_data or return_attrs is True, but return False neither were True.
        False: pass
            
    fail_gracefully: bool
        True: If an error occurs, raise an exception.
        False: If an error occurs, return False, False or False if query_write_check is True.
        If query_write_check is False, return False, False.
        
    verbose: bool
        True: print messages
        False: do not print messages

    Return:
        data: np.array or None
            Data contained in the dataset referred to by dset_name in h5_fpath.

        attrs_dct: dict or None
            Keys are those in dset.attrs and values are dset.attrs[key]
            
    Purpose: Read dataset referred to by dset_name in h5_fpath and return the data inside and/or
        the attribute dictionary, if desired.
    '''
    try:    
        if dset_name is None:
            dset_name = fname_from_fpath(h5_fpath)
        with h5py.File(h5_fpath, 'r') as hf:
            if query_write_check:
                if 'writing_complete' in hf.attrs and hf.attrs['writing_complete']:
                    if not return_data and not return_attrs:
                        return True
                else:
                    if return_data or return_attrs:
                        return False, False
                    else:
                        return False
            dset = hf[dset_name]
            if return_data and return_attrs:
                if row_end_idx is None:
                    if col_end_idx is None:
                        return dset[row_start_idx : , col_start_idx : ], {key:dset.attrs[key] for key in dset.attrs}
                    else:
                        return dset[row_start_idx : , col_start_idx : col_end_idx + 1], {key:dset.attrs[key] for key in dset.attrs}
                else:
                    if col_end_idx is None:
                        return dset[row_start_idx : row_end_idx + 1, col_start_idx : ], {key:dset.attrs[key] for key in dset.attrs}
                    else:
                        return dset[row_start_idx : row_end_idx + 1, col_start_idx : col_end_idx + 1], {key:dset.attrs[key] for key in dset.attrs}
            if return_data and not return_attrs:
                if row_end_idx is None:
                    if col_end_idx is None:
                        return dset[row_start_idx :, col_start_idx :], None
                    else:
                        return dset[row_start_idx :, col_start_idx : col_end_idx + 1], None
                else:
                    if col_end_idx is None:
                        return dset[row_start_idx : row_end_idx + 1, col_start_idx :], None
                    else:
                        return dset[row_start_idx : row_end_idx + 1, col_start_idx : col_end_idx + 1], None
            if not return_data and return_attrs:
                return None, {key:dset.attrs[key] for key in dset.attrs}
            if not return_data and not return_attrs:
                return None, None
    except Exception as e:
        if fail_gracefully:
            if verbose:
                print(e)
            if query_write_check:
                if return_data or return_attrs:
                    return False, False
                else:
                    return False
            else:
                return False, False


def grep_found_in_files(search_str, fpaths):
    '''
    search_str: str
        If this string is found in an fpath, append fpath to list of files that the search string was found.
        Else, append fpath to list of files that the search string was not found.
    
    fpaths: iterable of str
        List of file paths which can be searched by opening them and reading the lines of the file.

    Return: fpaths_containing_search_str, fpaths_not_containing_search_str
        fpaths_containing_search_str: list of str
            List of files in fpaths that contain search_str
        fpaths_not_containing_search_str: list of str
            List of files in fpaths that do not contain search_str

    Purpose: Determine which files in fpaths contain search_str and which do not.
    '''
    _, found_fpaths = grep(search_str, fpaths, read_mode='r', fail_if_DNE=False, verbose=False, return_line_nums=False, return_fpaths=True)
    fpaths_containing_search_str = set(found_fpaths)
    fpaths_not_containing_search_str = set(fpaths) - fpaths_containing_search_str
    return list(fpaths_containing_search_str), list(fpaths_not_containing_search_str)


def get_dir_size(dir_path, recursive=True, lowmem=False, fail_if_DNE=False):
    fpaths = find(dir_path, '*', find_dirs=False, 
            recursive=recursive)
    
    if lowmem and fail_if_DNE:
        total_size = 0
        for fpath in fpaths:
            total_size += os.path.getsize(fpath)
    elif fail_if_DNE:
        total_size = 0
        for fpath in fpaths:
            try:
                total_size += os.path.getsize(fpath)
            except:
                continue
    else:
        total_size = np.sum([os.path.getsize(fpath) for fpath in fpaths])
    return total_size






