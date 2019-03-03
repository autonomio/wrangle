from .df_to_numeric import df_to_numeric


def df_corr_any(data, correlation='pearson'):

    '''Correlation for columns in a dataframe. Columns with
    non-numeric values are ignored. Columns with numeric values
    but wrong dtype are converted to numeric first.

    data : pandas dataframe
        The input data for the correlation
    correlation : str
        Method to be used for correlation: 'pearson', 'spearman', 'kendall'

    '''

    return df_to_numeric(data).corr(correlation)
