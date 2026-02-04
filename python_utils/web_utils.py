"""
Created on Tue Feb  6 19:26:59 2018

@author: timcrose
"""

from subprocess import Popen, PIPE, STDOUT
import requests

def connected_to_internet(hostname):
    toping = Popen(['ping', hostname, '-c', '1'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    output = toping.communicate()[0]
    hostalive = toping.returncode
    if hostalive == 0:
        #internet is up
        return True
    else:
        return False

def wait_until_connected_to_internet():
    #zinc will only be reachable if the internet is on for both tin and lead
    hostname = 'zinc.materials.cmu.edu'
    while not connected_to_internet(hostname):
        pass


def online():
    url = 'http://www.google.com/'
    timeout = 5
    try:
        request = requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout) as exception:
        return False
    return False