
"""
Created on Tue Feb  6 19:16:48 2018

@author: timcrose
"""

import csv, json, os, sys, pickle, h5py
from glob import glob
import shutil, fnmatch, random
from python_utils import time_utils, err_utils, list_utils
import platform
import numpy as np

python_version = float(platform.python_version()[:3])
if python_version >= 3.0:
    from python_utils.file_utils3 import *
elif python_version >= 2.0:
    from python_utils.file_utils2 import *
else:
    print('python version below 2.0, potential for some unsupported ' +
        'functions')


def write_pickle(fpath, data, fail_gracefully=False, verbose=False):
    '''
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

    Return: None

    Purpose: pickle a python object and store in
         a .pickle file with path fpath using pickle.dump().

    Notes: The .pickle file will be loadable by pickle.load()
    '''
    # Write the python object data to the file with path fpath
    try:
        with open(fpath, "wb") as pickle_out:
            pickle.dump(data, pickle_out)
    except Exception as e:
        err_utils.handle_error(e=e, err_message=err_message, fail_gracefully=\
            fail_gracefully, verbose=verbose)
        return None


def read_pickle(fpath, fail_gracefully=True, verbose=False):
    '''
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

    Return:
        if the file is not able to be unpickled: return None
        else: return the unpickled object

    Purpose: unpickle a python object that was stored in
         a .pickle file.

    Notes:
        The .pickle file must be loadable by pickle.load()
    '''

    # Make sure fpath has a .pickle extension
    if not fpath.endswith('.pickle'):
        err_message = 'file path must end with .pickle. fpath: ' + fpath
        err_utils.handle_error(err_message=err_message, fail_gracefully=\
            fail_gracefully, verbose=verbose)

        return None

    # Load the pickled python object into python_obj
    try:
        with open(fpath, 'rb') as f:
            python_obj = pickle.load(f)
        return python_obj
    except Exception as e:
        err_message = 'Could not load the file at fpath =  ' + fpath
        err_utils.handle_error(e=e, err_message=err_message, fail_gracefully=\
            fail_gracefully, verbose=verbose)

        return None


def get_lines_of_file(fname, mode='r'):
    with open(fname, mode) as f:
        lines = f.readlines()
    return lines


def grep_single_file(search_str, fpath, read_mode, verbose=False):
    try:
        lines = get_lines_of_file(fpath, mode=read_mode)
    except UnicodeDecodeError:
        if verbose:
            print('Warning, UnicodeDecodeError encountered by grep_single_file for fpath', fpath)
        lines = []
    found_result = [line for line in lines if search_str in line]
    found_result_line_nums = [i for i,line in enumerate(lines) if search_str in line]
    found_result_fpaths = [fpath] * len(found_result)
    return found_result, found_result_line_nums, found_result_fpaths
            

def grep_str(search_str, path, read_mode, fail_if_DNE=False, verbose=False):
    if type(path) is str:
        if os.path.isdir(path):
            return grep_dir_recursively(search_str, path, read_mode)
        elif os.path.isfile(path):
            return grep_single_file(search_str, path, read_mode, verbose)
        
    if not fail_if_DNE:
        if verbose:
            print('path DNE: ', path)
        return [], [], []
    else:
        raise FileNotFoundError('path DNE: ', path)


def grep(search_str, paths, read_mode='r', fail_if_DNE=False, verbose=False, return_line_nums=False, return_fpaths=False):
    found_lines = []
    found_line_nums = []
    found_fpaths = []
    if type(paths) is str:
        paths = [paths]         

    if hasattr(paths, '__iter__'):
        for path in paths:
            found_result, found_result_line_nums, found_result_fpaths = grep_str(search_str, path, read_mode, fail_if_DNE=fail_if_DNE, verbose=verbose)
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
        raise IOError('Cannot move, destination path needs to be of type str. dest_fpath:', dest_fpath)
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


def read_csv(path,mode='r', map_type=None, dtype=None):
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


def lock_file(fpath, message='locked', total_timeout=100000, time_frame=0.05, go_ahead_if_out_of_time=False):
    start_time = time.time()
    wait_for_file_to_vanish(fpath, total_timeout=total_timeout, time_frame=time_frame,  go_ahead_if_out_of_time=go_ahead_if_out_of_time)
    read_lockfile_message = 'Nonelkjlkj'
    while read_lockfile_message != lockfile_message:
        with open(fpath, 'w') as f:
            f.write(message)
        time.sleep(0.05)
        try:
            with open(fpath) as f:
                read_lockfile_message = f.read()
        except:
            pass
        if time.time() - start_time > total_timeout and not go_ahead_if_out_of_time:
            raise Exception('Took longer than total_timeout =', total_timeout, 'seconds to acquire lock file.')
    

def wait_for_file_to_vanish(fpath, total_timeout=100000, time_frame=0.05, go_ahead_if_out_of_time=False):
    start_time = time.time()
    if time_frame == 0:
        while os.path.exists(fpath):
            if time.time() - start_time > total_timeout and not go_ahead_if_out_of_time:
                raise Exception('file ' + fpath + ' still exists after a total of ' + str(total_timeout) + ' seconds') 
        return   
    #wait until a file is removed by some other process
    while os.path.exists(fpath):
        #sleep a random amount of time to help prevent clashing (if multiple ranks)
        time_utils.sleep(random.uniform(time_frame, 1.1 * time_frame))
        if time.time() - start_time > total_timeout and not go_ahead_if_out_of_time:
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


def get_new_task(lockfile_fpath, incomplete_tasks_fpath):
    lockfile_message = str(int(time.time() * 10000))
    read_lockfile_message = 'None'
    lock_file(lockfile_fpath, message='locked', total_timeout=100, time_frame=0.05, go_ahead_if_out_of_time=False)
    wait_for_file_to_be_written_to(incomplete_tasks_fpath, total_timeout=1000, time_frame=0.05)
    tasks_df = pd.read_csv(incomplete_tasks_fpath)
    if len(tasks_df.values[len(tasks_df) - 1]) > 0:
        task_id = tasks_df.values[len(tasks_df) - 1][0]
    else:
        rm(lockfile_fpath)
        return None
    tasks_df.drop(index=len(tasks_df) - 1, inplace=True)
    tasks_df.to_csv(incomplete_tasks_fpath, index=False)
    num_incomplete_tasks = len(tasks_df)
    del tasks_df
    rm(lockfile_fpath)
    return task_id


def add_completed_task(lockfile_fpath, complete_tasks_fpath, task_id, intermediate_func=None, intermediate_args=[]):
    # Use lockfile for complete tasks to let me know this task_id was complete.
    lockfile_message = str(int(time.time() * 10000))
    read_lockfile_message = 'None'
    lock_file(lockfile_fpath, message='locked', total_timeout=100, time_frame=0.05, go_ahead_if_out_of_time=False)
    if os.path.exists(complete_tasks_fpath):
        wait_for_file_to_be_written_to(complete_tasks_fpath, total_timeout=1000, time_frame=0.05)
        tasks_df = pd.read_csv(complete_tasks_fpath)
        tasks_df = tasks_df.append(pd.DataFrame({'task_id':[task_id]}))
    else:
        tasks_df = pd.DataFrame({'task_id':[task_id]})
    if intermediate_func is not None:
        intermediate_func(*intermediate_args)
    # Write to complete_tasks_fpath that this task is complete
    tasks_df.to_csv(complete_tasks_fpath, index=False)
    del tasks_df
    rm(lockfile_fpath)
    
    
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


def replace_line_in_file(search_str, replacement_line, fpath, num_of_occurrences=-1, search_from_top_to_bottom=True):
    '''
    search_str: str
        This string is searched for in the file as a substring of a line in that file.
        This line on which this string is found will be replaced by replacement_line.
    replacement_line: str
        The line to replace the line that search_str was found on.
    fpath: str
        file path to search and do the replacing on
    num_of_occurrences: int
        The first num_of_occurrences lines containing search_str will be replaced with replacement_line (in
        the direction specified by search_from_top_to_bottom). If num_of_occurrences == -1 then all lines
        containing search_str will be replaced with replacement_line.
    search_from_top_to_bottom: bool
        True: Iterate through the lines sequentially
        False: Iterate through the lines in reverse order

    Return: None

    Purpose: Find a line in a specified file that contains a search string, then replace
        that line with replacement_line.

    Notes: Put a newline character at the end of your replacement_line string if you wish one to be there.
    '''
    lines = get_lines_of_file(fpath)
    if not search_from_top_to_bottom:
        lines.reverse()
    num_occurrences = 0
    for i,line in enumerate(lines):
        if num_of_occurrences != -1 and num_occurrences >= num_of_occurrences:
            break
        if search_str in line:
            lines[i] = replacement_line
            num_occurrences += 1
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


def write_h5_file(h5_fpath, data, attrs_dct={}, dset_name=None, overwrite=True, fail_if_already_exists=False, verbose=False):
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


def read_h5_file(h5_fpath, row_start_idx=0, row_end_idx=None, col_start_idx=0, col_end_idx=None, dset_name=None, return_data=True, return_attrs=True):
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

    Return:
        data: np.array or None
            Data contained in the dataset referred to by dset_name in h5_fpath.

        attrs_dct: dict or None
            Keys are those in dset.attrs and values are dset.attrs[key]

    Purpose: Read dataset referred to by dset_name in h5_fpath and return the data inside and/or
        the attribute dictionary, if desired.
    '''
    if dset_name is None:
        dset_name = fname_from_fpath(h5_fpath)
    with h5py.File(h5_fpath, 'r') as hf:
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






