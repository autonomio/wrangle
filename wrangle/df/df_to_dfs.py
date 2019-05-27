def df_to_dfs(data, groupers, y):

    '''Converts a dataframe to dict of dataframes

    Takes in a dataframe where a column name prefix
    or similar can be used as a grouper for columns
    and returns a dictionary with corresponding keys
    for each subset dataframe.

    USE: df_to_dfs(df, handler=['First', 'Last', 'Median'], 'expire_flag')

    data : pandas DataFrame
        A pandas dataframe with an outcome feature and more than one
        columns with a label that have some common nominator in their
        naming.
    groupers : list
        One or more groupers to be used as an identifier for the
        split (the common nominator).
    y : str
        The outcome feature to be included in all the dataframes.

    '''

    import re

    if isinstance(groupers, list) is False:
        groupers = [groupers]

    out = {}

    for label in groupers:

        # find the columns with a handler
        cols = [i for i in data.columns if len(re.findall(label, i)) > 0]
        temp = data[cols]

        # remove the handler from the columns
        temp.columns = [i.replace(label, '') for i in temp.columns]

        # add the outcome feature
        temp[y] = data[y]

        # add dataframe to output dictionary
        out[label] = temp

    return out
