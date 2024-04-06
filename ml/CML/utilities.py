from datetime import datetime

def get_expname_datetime(options):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    expname = dt_string + '_' + options['name'] + '_' + options['model']+'_' + options['mode'] 
    print("exp. name ="+expname)
    return expname