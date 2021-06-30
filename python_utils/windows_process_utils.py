# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:03:32 2021

@author: timcr
"""
import os
import time
import pandas as pd


def convert_mem_to_int(windows_tasks_df):
    '''
    Parameters
    ----------
    windows_tasks_df: pd.DataFrame
        See windows_tasks_df docs in get_windows_tasks_df() function docs.

    Returns
    -------
    None
    
    Purpose
    -------
    Calculate and insert a column into windows_tasks_df which contains the
    numeric integer value for memory usage of the processs. The Windows
    'tasklist' function only gives memory usage in terms of a number with 
    commas and a space and 'K' for the unit. e.g. '218,036 K' but I want to
    convert this to 218036.
    '''
    mem_usages = windows_tasks_df['mem_usage'].values
    mem_usage_numeric = []
    for mem_usage in mem_usages:
        # e.g. mem_usage is '218,036 K'
        number_portion = mem_usage.split()[0] # -> '218,036'
        removed_commas = number_portion.replace(',','') # -> '218036'
        if removed_commas == 'N/A':
            integer_mem = 0
        else:
            integer_mem = int(removed_commas) # -> 218036
        mem_usage_numeric.append(integer_mem)
    
    windows_tasks_df.loc[:, 'mem_usage_numeric'] = mem_usage_numeric
    
    
def convert_cpu_time_to_secs(windows_tasks_df):
    '''
    Parameters
    ----------
    windows_tasks_df: pd.DataFrame
        See windows_tasks_df docs in get_windows_tasks_df() function docs.

    Returns
    -------
    None
    
    Purpose
    -------
    Calculate and insert a column into windows_tasks_df which contains the
    numeric integer value for the number of seconds the processs have occupied
    the CPU so far. The Windows 'tasklist' function only gives CPU time in 
    terms of a string with format HH:MM:SS where HH could be greater than
    2 digits and MM and SS are integers between 00 and 59. e.g. '25:31:02' but 
    I want to convert this to 91862 total seconds.
    
    Method
    ------
    The code used in this function to get cpu_seconds is a shorthand of the 
    following:
        
    
    '''
    cpu_times = windows_tasks_df['cpu_time'].values
    cpu_seconds = []
    for cpu_time in cpu_times:
        if cpu_time == 'N/A':
            cpu_seconds_sum = 0
        else:
            # e.g. cpu_time is '25:31:02'
            split_cpu_time = cpu_time.split(':') # -> ['25', '31', '02']
            reversed_split_cpu_time = split_cpu_time[::-1] # -> ['02', '31', '25']
            cpu_seconds_lst = [int(entry) * (60 ** i) for i, entry in 
                enumerate(reversed_split_cpu_time)] # -> [2, 31 * 60, 25 * 3600]
            
            cpu_seconds_sum = sum(cpu_seconds_lst) # -> 91862
        cpu_seconds.append(cpu_seconds_sum)

    windows_tasks_df.loc[:, 'cpu_seconds'] = cpu_seconds
    
    
def wildcard_var_checks(vars_df, task_row):
    '''
    Parameters
    ----------
    vars_df: pd.DataFrame
        See vars_df docs under the get_vars_df() function.
        
    task_row: pd.Series
        A particular task_row out of windows_tasks_df representing a particular
        process. See windows_tasks_df docs under the get_windows_tasks_df() 
        function for more info.

    Returns
    -------
    passes: bool
        True: the values of its parameters pass all the filters (if any) with 
        regard to matching the input search strings (exact match if 
        'use_wildcard' is False or sub-string match if 'use_wildcard is True').

    Purpose
    -------
    For a particular process (task_row), make sure the values of its parameters
    pass all the filters (if any) with regard to matching the input search 
    strings. Wildcards allow matching a sub-string to suffice, while 
    disallowing wildcards means an exact match is required.
    
    ex. 1: if task_row is
    image_name process_id session_name session_num  mem_usage         status
    pythonw.exe       5352      Console           1    8,848 K        Unknown \
        
                user_name cpu_time          window_title  cpu_seconds
    DESKTOP-KHK29DQ\timcr  0:00:00                   N/A            0  \
        
    mem_usage_numeric
    8848
    
    and vars_df is
    name                value  use_wildcard lower_bound  upper_bound
    image_name          python    True        None          None
    process_id          None      True        None          None
    session_name        None      True        None          None
    session_num         None      True        None          None
    mem_usage_numeric   None      False       None          None
    status              None      True        None          None
    user_name           None      True        None          None
    cpu_seconds         None      False       None          None
    window_title        None      True        None          None
    
    then this task_row would pass because the only parameter with a filter is
    'image_name' and because 'use_wildcard' is True and "python" is a 
    sub-string of "pythonw.exe". If 'use_wildcard' was False then this task_row
    would fail because "python" is not an exact match of "pythonw.exe"
    
    Method
    ------
    Iterate through the parameters in vars_df and make sure the value recorded
    in windows_tasks_df for each parameter is within its bounds using the
    'upper_bound' and 'lower_bound' column values.
    
    Notes
    -----
    1. If the parameter doesn't have a filter, ('value' is None), the check
    is considered to pass for that parameter.
    
    2. The wildcards mentioned are simply sub-string checks; '*' cannot be 
    used.
    '''
    passes = True
    for i in vars_df.index:
        if vars_df.loc[i]['value'] is not None:
            if vars_df.loc[i]['use_wildcard']:
                passes = vars_df.loc[i]['value'] in task_row[vars_df.loc[i]['name']]
            else:
                passes = vars_df.loc[i]['value'] == task_row[vars_df.loc[i]['name']]
        if not passes:
            break
    return passes


def bound_checks(vars_df, task_row):
    '''
    Parameters
    ----------
    vars_df: pd.DataFrame
        See vars_df docs under the get_vars_df() function.
        
    task_row: pd.Series
        A particular task_row out of windows_tasks_df representing a particular
        process. See windows_tasks_df docs under the get_windows_tasks_df() 
        function for more info.

    Returns
    -------
    passes: bool
        True: All parameters for this process are within their bounds, or
            have None as their bounds.
        False: At least one parameter had a non-None bound for which the value
            for this process was outside of.

    Purpose
    -------
    For a particular process (task_row), make sure each parameter is within the 
    bounds requested by the user. If so, return True.
    
    Method
    ------
    Iterate through the parameters in vars_df and make sure the value recorded
    in windows_tasks_df for each parameter is within its bounds using the
    'upper_bound' and 'lower_bound' column values.
    
    Notes
    -----
    1. If the parameter doesn't have bounds (the bounds are None), then it
    is considered to pass the check.
    
    2. Some parameters don't make sense to have a bound. These are given the
    unchangeable hard-coded None for their bound input values.
    '''
    passes = True
    for i in vars_df.index:
        if vars_df.loc[i]['lower_bound'] is not None:
            if vars_df.loc[i]['upper_bound'] is not None:
                passes = (task_row[i] <= vars_df.loc[i]['upper_bound'] and 
                        task_row[i] >= vars_df.loc[i]['lower_bound'])
            
            else:
                passes = task_row[i] >= vars_df.loc[i]['lower_bound']
        else:
            if vars_df.loc[i]['upper_bound'] is not None:
                passes = task_row[i] <= vars_df.loc[i]['upper_bound']
            else:
                passes = True
        if not passes:
            break
    return passes


def get_windows_tasks_df():
    '''
    Parameters
    ----------
    None

    Returns
    -------
    windows_tasks_df: pd.DataFrame
        This dataframe contains the output of the Windows' function called
        'tasklist' formatted into a pandas DataFrame. Inserted into this
        dataframe are two additional columns: 'mem_usage_numeric' and
        'cpu_seconds'.
        
        Columns: 'image_name', 'process_id', 'session_name', 'session_num', 
        'mem_usage', 'status', 'user_name', 'cpu_time', 'window_title',
        'cpu_seconds', 'mem_usage_numeric'
            
        For docs on 'image_name', 'process_id', 'session_name', 'session_num', 
        'mem_usage', 'status', 'user_name', 'cpu_time', 'window_title', see 
        above docs for passing_df.
        
        'cpu_seconds': int
            The number of seconds that the application has occupied the CPU.
            
        'mem_usage_numeric': int
            The current memory usage of this application in kilobytes.
        
    Other Vars
    ----------
    windows_tasks_lines: list of list
        Data returned from the output of the Windows' function called
        'tasklist', formatted in a way that can be used as the data parameter
        for creating the windows_tasks_df dataframe using pd.DataFrame. See
        docs for windows_tasks_df for more details.
        
    Purpose
    -------
    Get the output of the Windows' function called 'tasklist' and format it 
    into a pandas DataFrame. Insert into this dataframe are two additional 
    columns: 'mem_usage_numeric' and 'cpu_seconds'. This dataframe will be 
    useful for monitoring the information associated with the tasks you want
    to run.

    Method
    ------
    Windows has a fancy command called 'takslist' which you can run through
    python using the os.popen command. Running 'tasklist /v /nh /fo csv' will
    get the required information. Now, I convert this csv string into a
    pd.DataFrame, format the mem_usage as an integer (kB) and cpu_seconds as 
    an integer (s).
    
    Notes
    -----
    1. See 
    https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/tasklist
    for Microsoft's documentation of their tasklist function.
    '''
    # Get the output from Windows' tasklist function into a list of lists
    # where each sub-list is the info for a single process (task).
    windows_tasks_lines = os.popen(
            'tasklist /v /nh /fo csv').read().strip().split('\n')
    
    windows_tasks_lines = [[entry.replace('"','') for entry in 
            line.split('","')] for line in windows_tasks_lines]
    
    windows_tasks_df = pd.DataFrame(windows_tasks_lines, columns=[
            'image_name', 'process_id', 'session_name', 'session_num', 
            'mem_usage', 'status', 'user_name', 'cpu_time', 'window_title'])
    
    # Create new columns for cpu time and mem usage which are numerical instead
    # of strings.
    convert_cpu_time_to_secs(windows_tasks_df)
    convert_mem_to_int(windows_tasks_df)
    return windows_tasks_df


def get_passing_df(vars_df):
    '''
    Parameters
    ----------
    vars_df: pd.DataFrame
        See the get_vars_df() function.

    Returns
    -------
    passing_df: pd.DataFrame
        This dataframe contains the rows of windows_tasks_df that pass all the
        filters that the user specifies through the input parameters to this
        function.
        
        Columns: 'image_name', 'process_id', 'session_name', 'session_num', 
            'mem_usage', 'status', 'user_name', 'cpu_time', 'window_title'
            
        For docs on 'image_name', 'process_id', 'session_name', 'session_num', 
             'status', 'user_name', 'window_title', see above docs for the 
             input parameters to this function.
             
        'mem_usage': str
            Integer string representing the amount of memory the process is 
            currently using. It's formatted like "145,240 K" where the number 
            uses the comma after every three digits and K represents the units
            which is always kilobytes.
            
        'cpu_time': str
            Amount of time the process has occupied the CPU. The format is
            HH:MM:SS where MM and SS are between 00 and 59 and HH is any
            unsigned int (which can grow to be larger than two digits).
        
    Purpose
    -------
    This function will return only information about the processes that pass
    all the filters you specify in the input vars_df to this function.
    
    Method
    ------
    Windows has a fancy command called 'takslist' which you can run through
    python using the os.popen command. I run this function through 
    get_windows_tasks_df() to get the info into a pd.DataFrame, format the 
    mem_usage as an integer (kB) and cpu_seconds as an integer (s). Then I 
    perform the filters that the user requested (organized into the input 
    vars_df dataframe) to return only the rows from the original pd.DataFrame 
    that the user wanted.
    
    Notes
    -----
    1. If vars_df was created without passing in any parameters, then this 
    function will return all processes that Windows' 'tasklist' function has 
    available to it (no filter is applied).
    '''
    windows_tasks_df = get_windows_tasks_df()
    passing_df = None
    # Each task_row (process) will be evaluated to make sure it passes all the 
    # filters requested by the user. Only those will be added to the returned
    # dataframe.
    for i in windows_tasks_df.index:
        task_row = windows_tasks_df.iloc[i]
        passes = bound_checks(vars_df, task_row)
        if not passes:
            continue
        passes = wildcard_var_checks(vars_df, task_row)
        if not passes:
            continue
        if passing_df is None:
            passing_df = pd.DataFrame([task_row])
        else:
            passing_df = passing_df.append([task_row], ignore_index=True)
    return passing_df


def get_vars_df(image_name=None, process_id=None, session_name=None, 
        session_num=None, mem_upper_bound=None, mem_lower_bound=None, 
        status=None, user_name=None, cpu_time_lower_bound=None, 
        cpu_time_upper_bound=None, window_title=None, 
        use_image_name_wildcard=True, use_process_id_wildcard=True,
        use_session_name_wildcard=True, use_session_num_wildcard=True,
        use_status_wildcard=True, use_user_name_wildcard=True, 
        use_window_title_wildcard=True):
    '''
    Parameters
    ----------
    image_name: str
        Name of the process.
        
        e.g. "cmd.exe"
        
    process_id: str
        Integer string representing the ID given to the process upon its
        creation. This ID does not change over time.
        
        e.g. "1232"
        
    session_name: str
        Name of the session.
        
        e.g. "Console"
        
    session_num: str
        Session number or identifier. It is an integer string.
        
        e.g. "1"
        
    mem_upper_bound: int
        Only include processes with mem_usage less than or equal to 
        mem_upper_bound. Units are kilobytes.
        
    mem_lower_bound: int
        Only include processes with mem_usage greater than or equal to 
        mem_lower_bound. Units are kilobytes.
        
    status: str
        The status of the process. Possible values include: 'Running',
        'Unknown', 'Not Responding'
        
    user_name: str
        Username which could be of the format 'user' or 'domain\\user'
        
        ex. 1: 'timcr'
        ex. 2: 'DESKTOP-KHK29DQ\\timcr'
    
    cpu_time_lower_bound: int
        Only include processes with cpu_seconds greater than or equal to 
        cpu_time_lower_bound. Units are seconds.
        
    cpu_time_upper_bound: int
        Only include processes with cpu_seconds less than or equal to 
        cpu_time_upper_bound. Units are seconds.
        
    window_title: str
        Title of the window. If the process doesn't have a GUI, then 'N/A'
        will be the window_title.
        
    use_image_name_wildcard: bool
        True: Include processes where image_name is a sub-string of the image name.
        False: Include processes where image_name exactly matches the image name.
        
        e.g. If True, and image_name is 'dog', then processes with names
        'dog.exe', 'dogs.exe', 'doggies.exe', etc would all be included.
        
    use_process_id_wildcard: bool
        True: Include processes where process_id is a sub-string of the PID.
        False: Include processes where process_id exactly matches the PID.
        
    use_session_name_wildcard: bool
        True: Include processes where session_name is a sub-string of the session name.
        False: Include processes where session_name exactly matches the session name.
        
    use_session_num_wildcard: bool
        True: Include processes where session_num is a sub-string of the session number.
        False: Include processes where session_num exactly matches the session number.
        
    use_status_wildcard: bool
        True: Include processes where status is a sub-string of the status.
        False: Include processes where status exactly matches the status.
        
    use_user_name_wildcard: bool
        True: Include processes where user_name is a sub-string of the username.
        False: Include processes where user_name exactly matches the username.
        
    use_window_title_wildcard: bool
        True: Include processes where window_title is a sub-string of the window title.
        False: Include processes where window_title exactly matches the window title.
    
    Returns
    -------
    vars_df
            
    vars_df: pd.DataFrame
        The input parameters are organized into this dataframe for clarity and
        reduction of boilerplate code. The index of this dataframe is the 
        same as the 'name' column values.
    
        Columns: 'name', 'value', 'use_wildcard', 'lower_bound', 'upper_bound'
    
        'name': str
            String name of the column header in windows_tasks_df that will be
            used in the filtering process. Values are 'image_name',
            'process_id', 'session_name', 'session_num', 'mem_usage_numeric', 
            'status', 'user_name', 'cpu_seconds', 'window_title'. For docs on
            these, see docs for windows_tasks_df.
            
        'value': int or str
            The value gotten from the user for the parameter with name 'name'
            (if any). In the case of 'mem_usage_numeric' and 'cpu_seconds',
            nothing was gotten from the user so None is the value here.
            
        'use_wildcard': bool
            The value of the corresponding input parameter for parameter with
            name 'name'.
            
            e.g. if 'name' is 'image_name', then 'use_wildcard' will have the
            value denoted by use_image_name_wildcard.
            
        'lower_bound': int
            The lower bound that will be used to filter out anything below this
            value for the parameter with 'name'. If there is no lower bound
            parameter for parameter with 'name', then None is used here which
            means don't check for whether the parameter is bounded.
            
            ex. 1: if 'name' is 'cpu_seconds' and cpu_time_lower_bound is
            100, then this says to filter out processes that have been running
            less than 100 seconds.
            
            ex. 2: if 'name' is 'image_name', there is no corresponding lower
            bound input parameter, thus None will be used and no check for
            lower bound will occur on a process' image name.
            
        'upper_bound': int
            Very similar to the documentation for 'lower_bound' above.
    
    Other Vars
    ----------
    data: list of list
        Each sub-list is an organized version of the user's input parameters.
        data is used to create the vars_df pd.DataFrame. See docs of vars_df
        for more info.
        
    Purpose
    -------
    Organize the input parameters which the user provides (to be used as filter
    criteria for windows tasks) into a pd.DataFrame which can be easily used
    in subsequent functions.
    '''
    # Store user input parameters in an organized way that can be put into
    # a dataframe.
    data = [
     ['image_name', image_name, use_image_name_wildcard, None, None],
     ['process_id', process_id, use_process_id_wildcard, None, None],
     ['session_name', session_name, use_session_name_wildcard, None, None],
     ['session_num', session_num, use_session_num_wildcard, None, None],
     ['mem_usage_numeric', None, False, mem_lower_bound, mem_upper_bound],
     ['status', status, use_status_wildcard, None, None],
     ['user_name', user_name, use_user_name_wildcard, None, None],
     ['cpu_seconds', None, False, cpu_time_lower_bound, cpu_time_upper_bound],
     ['window_title', window_title, use_window_title_wildcard, None, None]
     ]
    
    vars_df = pd.DataFrame(data, columns=['name', 'value', 'use_wildcard',
            'lower_bound', 'upper_bound'])
    
    # Set the index of vars_df to be the name of the parameters for easy and
    # clear access to the input parameters associated with the parameter names.
    vars_df.index = vars_df['name'].values
    return vars_df


def get_new_tasks_df(vars_df, old_passing_df):
    '''
    Parameters
    ----------
    vars_df: pd.DataFrame
        See docs in the get_vars_df() function.
        
    old_passing_df: pd.DataFrame
        See docs in the get_passing_df() function.
        
    Returns
    -------
    new_tasks_df: pd.DataFrame
        This dataframe contains the tasks that are currently gotten by 
        get_passing_df() that were not contained in the old_passing_df that was 
        passed into this function. See passing_df docs in the get_passing_df() 
        function for more details.

    Purpose
    -------
    Get the tasks that match the filters in vars_df that DNE in old_passing_df 
    (i.e. new tasks started after the original passing_df was gotten).
    
    Method
    ------
    Append the previous, old_passing_df to the current, new_passing_df 
    (according to the same vars_df) and then remove duplicate rows.
    '''
    new_passing_df = get_passing_df(vars_df)
    combined_passing_df = new_passing_df.append(old_passing_df, 
            ignore_index=True)
    
    new_tasks_df = combined_passing_df.drop_duplicates(subset='process_id', 
            keep=False)
    
    new_tasks_df.index = list(range(len(new_tasks_df.index)))
    return new_tasks_df


def monitor_windows_task(process_id=None, task_row=None, max_mem=None, 
        max_time=None,  no_not_responding_status=True, 
        kill_task_if_over_mem=True, kill_task_if_over_time=True, 
        kill_task_if_not_responding=True, sleep_delay=1):
    
    '''
    Parameters
    ----------
    process_id: str or None.
        If None, use the PID stored in task_row. See get_vars_df() function
        docs.
    
    task_row: pd.Series or None.
        If None, use process_id for the PID. See wildcard_var_checks() function 
        docs.
        
    max_mem: float or None
        If not None, check to make sure the memory usage of this task is less
        than max_mem. If it is above, then see kill_task_if_over_mem for 
        behavior.
    
    max_time: float or None
        If not None, check to make sure the time this task has occupied the
        CPU is less than max_time number of seconds. If it is above, then see 
        kill_task_if_over_time for behavior.
    
    no_not_responding_status: bool
        True: Check status and see kill_task_if_not_responding for behavior.
        
        False: Do not check status.
        
    kill_task_if_over_mem: bool
        True: If the memory usage goes over max_mem, then terminate the task.
        
        False: If the memory usage goes over max_mem, then return from this 
            function and notify the calling function.
            
    kill_task_if_over_time: bool
        True: If the time usage goes over max_time, then terminate the task.
        
        False: If the time usage goes over max_time, then return from this 
            function and notify the calling function.
            
    kill_task_if_not_responding: bool
        If no_not_responding_status is True and the status of the task is 
        "Not Responding" then the check will:
            
        True: Kill the task and return "Not Responding".
            
        False: Return "Not Responding".
            
    sleep_delay: float
        Number of seconds to sleep in between checks

    Returns
    -------
    monitor_result: str
        A description of what caused this function to return. Possible values
        are "Not Responding", "Over memory", "Over time"

    Purpose
    -------
    Monitor the information that Windows provides about a process including it's
    memory usage, status, and cpu run time. You can make sure it's not using 
    too much memory or running too long, or "Not Responding", and if it is, you
    can terminate the task and/or return a message saying what happened.
    '''
    if process_id is None:
        process_id = task_row['process_id']
    while True:
        windows_tasks_df = get_windows_tasks_df()
        new_task_row = windows_tasks_df[windows_tasks_df['process_id']
                == process_id]
        
        new_task_row.index = [0]
        print('new_task_row')
        print(new_task_row)
        
        # Check memory usage
        if max_mem is not None:
            if new_task_row.iloc[0]['mem_usage_numeric'] > max_mem:
                if kill_task_if_over_mem:
                    # kill task
                    os.system('taskkill /f /pid ' + process_id)
                return 'Over memory'
        # Check CPU time
        if max_time is not None:
            print('now new_task_row')
            print(new_task_row)
            if new_task_row.iloc[0]['cpu_seconds'] > max_time:
                if kill_task_if_over_mem:
                    # kill task
                    os.system('taskkill /f /pid ' + process_id)
                return 'Over time'
        # Check status
        if no_not_responding_status:
            if new_task_row.iloc[0]['status'] == 'Not Responding':
                if kill_task_if_not_responding:
                    # kill task
                    os.system('taskkill /f /pid ' + process_id)
                return 'Not Responding'
        # Delay between checks
        time.sleep(sleep_delay)
       
        
def main():
    pd.set_option('display.max_columns', None)
    vars_df = get_vars_df(image_name='python', use_image_name_wildcard=True)
    passing_df = get_passing_df(vars_df)
    print('passing_df')
    print(passing_df)
    # Start a new process while you wait for the sleep on the next line
    time.sleep(5)
    new_tasks_df = get_new_tasks_df(vars_df, passing_df)
    print('new_tasks_df')
    print(new_tasks_df)
    monitor_result = monitor_windows_task(task_row=new_tasks_df.iloc[0], 
            max_time=10, kill_task_if_over_time=True, sleep_delay=1)
    
    print('monitor_result:', monitor_result)
    
    
if __name__ == '__main__':
    main()
