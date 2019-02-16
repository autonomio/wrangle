def is_number(value):

    '''Checks if a string can be converted into
    a float (or int as a by product). Helper function
    for string_cols_to_numeric'''

    try:
        float(value)
        return True
    except ValueError:
        return False
