
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
from python_utils.type_utils import Optional, Int, Scalar, Union


def get_timestamp_from_date_str(date_str: str, date_fmt: str='%m/%d/%Y, %I:%M:%S %p') -> float:
    '''
    date_str: str
        String of date and time with format corresponding to date_fmt

    date_fmt: str
        Format string of date_str according to datetime.datetime.datetime.datetime documentation.

    Return: float
        Number of seconds since the epoch as measured by datetime.datetime.
    
    Purpose: Get the number of seconds since the epoch as measured by datetime.datetime when
        
    '''
    dt = datetime.datetime.strptime(date_str, date_fmt)
    return dt.timestamp()


def get_date_str_from_timestamp(timestamp: float, date_fmt: str='%m/%d/%Y, %H:%M:%S') -> str:
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime(date_fmt)
     

def get_greg_time_from_time_str(time_str: str='00:00:00', ds: Optional[str]=None, DATE_FMT: str='%Y-%m-%d', TIME_FMT: str='%H:%M:%S') -> float:

    if ds is None:
        #default to today
        ds = datetime.datetime.today().strftime(DATE_FMT)
        
    dtt = datetime.datetime.strptime(ds + ' ' + time_str, DATE_FMT + ' ' + TIME_FMT)
    greg_time = float(mktime(dtt.timetuple()))

    return greg_time


def get_secs_from_time_str(time_str: str='00:00:00', TIME_FMT: str='%H:%M:%S') -> float:
    struct_time = strptime(time_str, TIME_FMT)
    secs = struct_time.tm_hour * 3600.0 + struct_time.tm_min * 60.0 + float(struct_time.tm_sec)
    return secs


def get_greg_from_mdYHMS(mon: Int, day: Int, yr: Int, hr: Int, minute: Int, sec: Scalar) -> float:
    fmt = '%m-%d-%Y %H:%M:%S'
    s = str(mon).zfill(2) + '-'  + str(day).zfill(2) + '-' + str(yr).zfill(4) + ' ' + str(hr).zfill(2) + ':' + str(minute).zfill(2) + ':' + str(int(sec)).zfill(2)
    dt = datetime.datetime.strptime(s, fmt)
    tt = dt.timetuple()
    greg_time = mktime(tt)
    return greg_time


def delay_start(time_of_day_to_start: str) -> None:
    start_time_greg = get_greg_time_from_time_str(time_str=time_of_day_to_start)

    if gtime() > start_time_greg:
        #trying to start tomorrow so need to get amount of delay until start time
        delay_time = get_greg_time_from_time_str(time_str='23:59:59') +1 - gtime() + start_time_greg - get_greg_time_from_time_str(time_str='00:00:00')
    else:
        delay_time = start_time_greg - gtime()

    sleep(delay_time)
    
    
def get_now_MM_DD_YYYY() -> str:
    return str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().year)


def wait_til_weekday(time_of_day_to_start: str='09:00:00') -> None:
    weekno = datetime.datetime.today().weekday()

    #Monday is 0 and Sunday is 6
    while not ((weekno in range(5)) and gtime() < get_greg_time_from_time_str(time_str=time_of_day_to_start)):
        sleep(3666)
        weekno = datetime.datetime.today().weekday()


def get_date_time_str_from_greg(greg: float) -> str:
    return ctime(int(greg))


def get_time_str_from_greg(greg: float) -> str:
    return strftime("%H:%M:%S", localtime(greg))


def day_of_week_from_date_str(date_str: str, delim: str='_') -> int:
    '''
    date_str: str
        must be of format month_day_year. e.g. 12_20_2018 (or other delimiter)
    delim: delimiter of passed input string. See date_str above
    
    return:
    day-of-the-week number 0 - 6 where 0 is Monday, 6 is Sunday
    '''
    month, day, year = [int(i) for i in date_str.split(delim)]
    born = datetime.date(year, month, day)
    return born.weekday()


def get_date_str_from_today(delimiter: str='_') -> str:
    '''
    month day year format
    '''
    month = str(datetime.datetime.today().month)
    dy = str(datetime.datetime.today().day)
    year = str(datetime.datetime.today().year)
    date_str = month + delimiter + dy + delimiter + year
    return date_str


def timedelta(dt: Union[datetime.date, datetime.datetime], days: Int=0, 
seconds: Int=0, microseconds: Int=0, milliseconds: Int=0, minutes: Int=0, 
hours: Int=0, weeks: Int=0, months: Int=0, years: Int=0) -> Union[datetime.date, datetime.datetime]:
    '''
    dt: datetime.date or datetime.datetime
        Date object that you want to modify.

    days: number
        The number of days to be added by datetime.timedelta after months and years have been modified.

    seconds: number
        The number of seconds to be added by datetime.timedelta after months and years have been modified.

    microseconds: number
        The number of microseconds to be added by datetime.timedelta after months and years have been modified.

    milliseconds: number
        The number of milliseconds to be added by datetime.timedelta after months and years have been modified.

    minutes: number
        The number of minutes to be added by datetime.timedelta after months and years have been modified.

    hours: number
        The number of hours to be added by datetime.timedelta after months and years have been modified.

    weeks: number
        The number of weeks to be added by datetime.timedelta after months and years have been modified.

    months: int
        The number of months to be added to the input dt.month (and taking into account the possibility of
        entering a different year).

    years: int
        The number of years added to dt.year is years plus any additional years gained or lost by the months modifier.
        
    Returns
    -------
    dt: datetime.date or datetime.datetime
        The returned date object will be the same type as the input dt. The returned date object now has all the
        modifications to the date and/or time that were requested.

    Purpose
    -------
    datetime.timedelta does not come with a built-in way to add/substract months or years. This is probably because
    of the ambiguity that arises. This function uses the methods described below to eliminate the ambiguity. This
    function returns the date object that has now been modified by all the date and/or time modifications that
    were requested (according to the method below).


    Methods
    -------
    1. months is the number of months we want to add to the current date assuming the input years is 0. Adding this
        number of months to the input date could cause a change in the output year.

        e.g. months=3 and dt.month = 11 would cause the month to be 2 (Feb) of the next year.

    2. In order to know how many years should be added due to the months input alone, get the number of months
        from Jan of the input dt.year. Then, divide this by 12 and subtract the remainder to get the number of years.

        e.g. # 1. months=3, dt.month=11, dt.year=2020 would mean we need to add 3 + 11 - 1 = 13 months to Jan 2020
        to reach Feb 2021. The "-1" is because Jan = 1. So months is now set to 13.

        13 / 12 = 1 + (1 / 12), so the reminader is (1 / 12). Thus the number of years to add due to the months input
        alone is (13 / 12) - (1 / 12) = 1. This could be re-expressed as (13 - 1) / 12 or (13 - (13 % 12)) / 12.

        e.g. # 2. months=15, dt.month=11, dt.year=2020 would mean we need to add 15 + 11 - 1 = 25 months to Jan 2020
        to reach Feb 2022.

        (25 - (25 % 12)) / 12 = (25 - 1) / 12 = 2 years will be added to dt.year due to the months input alone.

    3. Increment the input years by the number of years that months werer already calculated.

        e.g. # 1. if years=2 then do years += 1 to get years = 3

    4. Add years to dt.year to get the year value (before adding any weeks, days, hours, minutes, etc)

        e.g. # 1. 2020 + 3 = 2023

    5. The month value becomes the remainder (months % 12) + 1 where the 1 is becuase Jan=1.

        e.g. # 1. (13 % 12) + 1 = 2 (which is Feb)

    6. Now add the datetime.timedelta values to get the final date.

    Notes
    -----
    1. months and years modify dt first and then the rest (days, seconds, etc) are applied.
    2. Numbers can be negative.
    3. months start with 1 being January and end with 12 being December.
    4. If datetime.date is the input, then items like minutes and seconds will still be used (from 00:00:00), and then
        the resulting year, month, and day will be used to return the final datetime.date object.
    '''
    datetime_time = type(dt) == datetime.time
    if datetime_time:
        dt = datetime.datetime(2000, 1, 1, dt.hour, dt.minute, dt.second, dt.microsecond)
    if months != int(months):
        raise ValueError('months must be a whole number. Got months = ', months)
    if years != int(years):
        raise ValueError('years must be a whole number. Got years = ', years)
    years = int(years)
    months = int(months)
    # Increment months by dt.month and then subtract 1 because we are using January = 1
    # and months is the number of months to add. Thus, after this next line, months becomes
    # the number of months to add if you were to start from January of dt.year.
    months = months + dt.month - 1
    # num_years_to_add is a modifier to years such that when added to years, years becomes the
    # value to add to dt.year to get the dt.year used for subsequent application of datetime.timedelta to get the final result.
    # Given that months is required to be an integer, num_years_to_add will always be an integer value.
    num_years_to_add = (months - (months % 12)) / 12
    years += num_years_to_add
    year = int(dt.year + years)
    # month will be the dt.month used for subsequent application of datetime.timedelta to get the final result.
    # month at this point is simply the number of months to add starting from January and then + 1 because January = 1.
    month = int((months % 12) + 1)
    if type(dt) == datetime.datetime:
        dt = datetime.datetime(year, month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, tzinfo=dt.tzinfo)
        dt += datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds, milliseconds=milliseconds, minutes=minutes, hours=hours, weeks=weeks)
    elif type(dt) == datetime.date:
        dt = datetime.datetime(year, month, dt.day)
        dt += datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds, milliseconds=milliseconds, minutes=minutes, hours=hours, weeks=weeks)
        dt = datetime.date(dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)
    else:
        raise Exception('Incompatible type = ', type(dt), '. Only datetime.datetime and datetime.date are currently supported.')
    if datetime_time:
        dt = datetime.time(dt.hour, dt.minute, dt.second, dt.microsecond)
    return dt


def get_utc_time() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def is_within_time_range(start_time: Union[datetime.time, datetime.datetime], 
end_time: Union[datetime.time, datetime.datetime], time_to_query=None, 
compare_hours_only=True, tzinfo=datetime.timezone.utc) -> bool:
    if time_to_query is None:
        time_to_query = get_utc_time()
    if type(start_time) == datetime.time and type(end_time) == datetime.time and time_to_query == datetime.time:
        pass
    elif compare_hours_only:
        if type(start_time) == datetime.time:
            start_time = datetime.datetime(2000, 1, 1, hour=start_time.hour, minute=start_time.minute, second=start_time.second, microsecond=start_time.microsecond, tzinfo=tzinfo)
        else:
            start_time = start_time.replace(year=2000, month=1, day=1, tzinfo=tzinfo)
        if type(end_time) == datetime.time:
            end_time = datetime.datetime(2000, 1, 1, hour=end_time.hour, minute=end_time.minute, second=end_time.second, microsecond=end_time.microsecond, tzinfo=tzinfo)
        else:
            end_time = end_time.replace(year=2000, month=1, day=1, tzinfo=tzinfo)
        if type(time_to_query) == datetime.time:
            time_to_query = datetime.datetime(2000, 1, 1, hour=time_to_query.hour, minute=time_to_query.minute, second=time_to_query.second, microsecond=time_to_query.microsecond, tzinfo=tzinfo)
        else:
            time_to_query = time_to_query.replace(year=2000, month=1, day=1, tzinfo=tzinfo)
        if end_time < start_time:
            start_time = timedelta(start_time, days=-1)
    return time_to_query <= end_time and time_to_query >= start_time