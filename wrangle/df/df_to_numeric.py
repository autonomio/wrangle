from types import FunctionType
from inspect import isclass

from ..utils.is_number import is_number


def df_to_numeric(data, destructive=False):

    '''Convert numeric values to ints and floats.

    data : pandas dataframe
         A pandas dataframe to be transformed

    Takes in a dataframe and attempts to convert numeric columns
    into floats or ints respectively.'''

    if destructive is False:
        data = data.copy(deep=True)

    for col in data.columns:
        # avoid common non-numeric cases
        if data[col].dtype != 'O':
            continue
        if isclass(data[col][0]):
            continue
        if isinstance(data[col][0], FunctionType):
            continue
        if data[col].apply(is_number).sum() == 0:
            continue

        # perform the conversion to int or float
        if data[col].apply(is_number).sum() == len(data):
            try:
                data[col] = data[col].astype(int)
            except ValueError:
                try:
                    data[col] = data[col].astype(float)
                except ValueError:
                    data[col] = data[col]
        else:
            data[col] = data[col]

    return data
