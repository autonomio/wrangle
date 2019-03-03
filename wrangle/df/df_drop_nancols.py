def df_drop_nancols(data, threshold=0, destructive=False):

    '''Drop columns based on a threshold of nan values. The optimal
    approach, especially for small datasets, seems to be to first
    start with df_drop_nancols, then follow with df_drop_nanrows,
    and then do one more pass with df_drop_nancols.

    data : pandas dataframe
        A pandas dataframe to be processed
    threshold : float
        A decimal value between 1 (all can be nans) and 0 (no nans allowed).
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    '''

    # avoid destruction

    if destructive is False:
        data = data.copy(deep=True)

    threshold = threshold * len(data)

    return data.dropna(axis=1, thresh=threshold)
