def df_resample_id(data, id_col, n=None, destructive=False):

    """Resamples a dataframe based on an id single column

    Create a sample where each id has a single observation
    in the whole dataset.

    data : DataFrame
        A pandas dataframe with at least one column
        with an identifier (id_col) and a data column.
    id_col : str
        The column with the id for limiting uniqueness
    n : int
        The size of the sample to be drawn
    destructive : bool

    """

    if destructive is False:
        data = data.copy(deep=False)

    data = data.drop_duplicates(id_col)
    data = data.sample(frac=1).head(n)

    return data
