
"""
Created on Tue Feb  6 19:26:58 2018

@author: timcrose
"""

from datetime import datetime
from time import sleep
from time import mktime
from time import time as gtime
from time import ctime
from time import localtime
from time import strftime

def get_greg_time_from_time_str(time_str='00:00:00', ds=None, DATE_FMT='%Y-%m-%d', TIME_FMT='%H:%M:%S'):

    if ds is None:
        #default to today
        ds = datetime.today().strftime(DATE_FMT)
        
    dtt = datetime.strptime(ds + ' ' + time_str, DATE_FMT + ' ' + TIME_FMT)
    greg_time = float(mktime(dtt.timetuple()))

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
    return str(datetime.now().month) + '_' + str(datetime.now().day) + '_' + str(datetime.now().year)

def wait_til_weekday():
    weekno = datetime.today().weekday()

    # check if will be a weekday tmw
    #while weekno > 3 and weekno < 6:
    #I believe Monday is 0 and Sunday is 6
    while weekno >= 5:
        sleep(3666)
        weekno = datetime.today().weekday()

def get_date_time_str_from_greg(greg):
    return ctime(int(greg))
def get_time_str_from_greg(greg):
    return strftime("%H:%M:%S", localtime(greg))
