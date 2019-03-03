def col_to_cols(data, label_col, handler_col):

    '''Convert a single multiclass column into columnsself.

    data : pandas dataframe
         A pandas dataframe to be transformed
    label_col : str
         A column with categorical labels (will become columns)
    handler_col : str
         A column with an identifier (e.g. country) which will become
         an index column.
    '''

    out = data.groupby([label_col, handler_col]).sum()
    out = out.unstack().transpose().reset_index()
    out = out.drop('level_0', axis=1)

    return out
