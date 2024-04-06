from datetime import datetime

def get_splitsuf(split_strategy,
                 splitsuf_yearly = 'T1821_V1920_T21',
                 splitsuf_random = 'T05_V025_T025',
                 splitsuf_NBH = '1821_T05_V025_T025',
                 splitsuf_hist = 'T19_V20_T21'):
    '''
    Get the suffix for the split strategy
    :param split_strategy: string, the split strategy 'randomsamp', 'yearly', 'NBH' or 'historical'
    :param splitsuf_random: string, the suffix for the random split strategy
    :param splitsuf_yearly: string, the suffix for the yearly split strategy
    :param splitsuf_NBH: string, the suffix for the NBH split strategy
    :param splitsuf_hist: string, the suffix for the historical split strategy
    :return: string, the suffix for the split strategy
    '''
    
    if split_strategy == 'yearly':
        splitsuf = splitsuf_yearly
    elif split_strategy == 'randomsamp':
        splitsuf = splitsuf_random
    elif split_strategy == 'NBH':
        splitsuf = splitsuf_NBH
    elif split_strategy == 'historical':
        splitsuf = splitsuf_hist
	
    return splitsuf


def get_datetime_string():
    '''
    Returns a string of the current date and time
    '''
    # datetime object containing current date and time
    now = datetime.now()

    dt_string = now.strftime("%y%m%d")
    time_string = now.strftime("%H%M%S")
    return dt_string, time_string

def get_user_confirmation(msg):
    '''
    Returns True if user responds with 'y' or 'Y'
    Returns False if user responds with 'n' or 'N'
    '''
    reply = str(input(msg) or 'n')
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        print(f'Response not understood')
        return get_user_confirmation(msg)