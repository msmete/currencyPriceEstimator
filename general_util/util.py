# ---> Special function: convert <datetime.date> to <Timestamp>
from datetime import datetime


def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')