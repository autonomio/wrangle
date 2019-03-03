def df_to_dfs(data, groupers, y):

    '''Converts a dataframe to dict of dataframes

    Takes in a dataframe where a column name prefix
    or similar can be used as a grouper for columns
    and returns a dictionary with corresponding keys
    for each subset dataframe.

    USE: df_to_dfs(df, handler=['First', 'Last', 'Median', 'Max', 'Min'])

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

    out = {}

    for label in groupers:

        cols = [i for i in data.columns if len(re.findall(label, i)) > 0]
        temp = data[cols]
        temp.columns = [i.replace(label, '') for i in temp.columns]
        temp[y] = data[y]

        out[label] = temp

    return out
