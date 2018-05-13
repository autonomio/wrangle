import pandas as pd


def clean_header(data, modify_df=True):

    temp = pd.Series(data.columns)

    # remove special characters
    temp = temp.str.replace(r'[^A-Za-z0-9]+', ' ')

    # remove multiple spaces
    temp = temp.str.replace(r' +', ' ')

    # convert to lower case
    temp = temp.str.lower()

    # replace spaces
    temp = temp.str.replace(' ', '_')

    # remove preceding/trailing underscore
    temp = temp.str.replace('_$', '')
    temp = temp.str.replace('^_', '')

    if modify_df:
        data.columns = temp.values
    else:
        return temp

    return data
