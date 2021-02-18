
"""
Created on Tue Feb  6 19:26:58 2018

@author: timcrose
"""

import datetime
from time import sleep
from time import mktime
from time import time as gtime
from time import ctime
from time import localtime
from time import strftime
from time import strptime

def get_timestamp_from_date_str(date_str, date_fmt='%m/%d/%Y, %I:%M:%S %p'):
    '''
    date_str: str
        String of date and time with format corresponding to date_fmt

    date_fmt: str
        Format string of date_str according to datetime.datetime.datetime.datetime documentation.

    Return: number
        Number of seconds since the epoch as measured by datetime.datetime.
    
    Purpose: Get the number of seconds since the epoch as measured by datetime.datetime when
        
    '''
    dt = datetime.datetime.strptime(date_str, date_fmt)
    return dt.timestamp()

def get_date_str_from_timestamp(timestamp, date_fmt='%m/%d/%Y, %H:%M:%S'):
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime(date_fmt)
     

def get_greg_time_from_time_str(time_str='00:00:00', ds=None, DATE_FMT='%Y-%m-%d', TIME_FMT='%H:%M:%S'):

    if ds is None:
        #default to today
        ds = datetime.datetime.today().strftime(DATE_FMT)
        
    dtt = datetime.datetime.strptime(ds + ' ' + time_str, DATE_FMT + ' ' + TIME_FMT)
    greg_time = float(mktime(dtt.timetuple()))

    return greg_time

def get_secs_from_time_str(time_str='00:00:00', TIME_FMT='%H:%M:%S'):
    struct_time = strptime(time_str, TIME_FMT)
    secs = struct_time.tm_hour * 3600.0 + struct_time.tm_min * 60.0 + float(struct_time.tm_sec)
    return secs

def get_greg_from_mdYHMS(mon, day, yr, hr, minute, sec):
    fmt = '%m-%d-%Y %H:%M:%S'
    s = str(mon).zfill(2) + '-'  + str(day).zfill(2) + '-' + str(yr).zfill(4) + ' ' + str(hr).zfill(2) + ':' + str(minute).zfill(2) + ':' + str(int(sec)).zfill(2)
    dt = datetime.datetime.strptime(s, fmt)
    tt = dt.timetuple()
    greg_time = mktime(tt)
    return greg_time

def delay_start(time_of_day_to_start):
    start_time_greg = get_greg_time_from_time_str(time_str=time_of_day_to_start)

    if gtime() > start_time_greg:
        #trying to start tomorrow so need to get amount of delay until start time
        delay_time = get_greg_time_from_time_str(time_str='23:59:59') +1 - gtime() + start_time_greg - get_greg_time_from_time_str(time_str='00:00:00')
    else:
        delay_time = start_time_greg - gtime()

    sleep(delay_time)
    
def get_now_MM_DD_YYYY():
    return str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().year)


def wait_til_weekday(time_of_day_to_start='09:00:00'):
    weekno = datetime.datetime.today().weekday()

    #Monday is 0 and Sunday is 6
    while not ((weekno in range(5)) and gtime() < get_greg_time_from_time_str(time_str=time_of_day_to_start)):
        sleep(3666)
        weekno = datetime.datetime.today().weekday()


def wait_til_weekday_old(time_of_day_to_start='09:15:00'):
    weekno = datetime.datetime.today().weekday()
    weekno_original = datetime.datetime.today().weekday()

    if weekno_original == 4:
        start_time_greg = get_greg_time_from_time_str(time_str=time_of_day_to_start)
        if gtime() < start_time_greg:
            #trying to start today
            return

    # check if will be a weekday tmw
    #Monday is 0 and Sunday is 6
    #Wait until today and tmw is a weekday
    #today is a weekday if weekno is 0, 1, 2, 3, 4
    #tmw is a weekday if weekno is 0, 1, 2, 3, 6
    #So today is a weekday and tmw is a weekday if weekno is 0, 1, 2, 3
    #But if we started the program on Friday morning (before the requested start time), then we should also allow it to exit the loop
    while weekno not in [0, 1, 2, 3]:
        sleep(3666)
        weekno = datetime.datetime.today().weekday()


def get_date_time_str_from_greg(greg):
    return ctime(int(greg))
def get_time_str_from_greg(greg):
    return strftime("%H:%M:%S", localtime(greg))

def day_of_week_from_date_str(date_str, delim='_'):
    '''
    date_str: str
        must be of format month_day_year. e.g. 12_20_2018 (or other delimiter)
    delim: delimiter of passed input string. See date_str above
    
    return:
    day-of-the-week number 0 - 6 where 0 is Monday, 6 is Sunday
    '''
    month, day, year = [int(i) for i in date_str.split(delim)]
    born = datetime.date(year, month, day)
    #return born.strftime('%A') Day name like "Sunday"
    return born.weekday()

def get_date_str_from_today(delimiter='_'):
    '''
    month day year format
    '''
    month = str(datetime.datetime.today().month)
    dy = str(datetime.datetime.today().day)
    year = str(datetime.datetime.today().year)
    date_str = month + delimiter + dy + delimiter + year
    return date_str


def timedelta(dt, days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0, months=0, years=0):
    if months != int(months):
        raise ValueError('months must be a whole number. Got months = ', months)
    if years != int(years):
        raise ValueError('years must be a whole number. Got years = ', years)
    years = int(years)
    months = int(months)
    dt += datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds, milliseconds=milliseconds, minutes=minutes, hours=hours, weeks=weeks)
    months = months + dt.month
    num_years_to_add = ((months - 1) - ((months - 1) % 12)) / 12
    years += num_years_to_add
    year = int(dt.year + years)
    month = int(((months - 1) % 12) + 1)
    if type(dt) == datetime.datetime:
        dt = datetime.datetime(year, month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
    elif type(dt) == datetime.date:
        dt = datetime.date(year, month, dt.day)
    else:
        raise Exception('Incompatible type = ', type(dt), '. Only datetime.datetime and datetime.date are currently supported.')
    return dt




