def df_drop_nanrows(data, destructive=False):

    '''Drop rows based on presences of nan values. The optimal
    approach, especially for small datasets, seems to be to first
    start with df_drop_nancols, then follow with df_drop_nanrows,
    and then do one more pass with df_drop_nancols.

    data : pandas dataframe
        A pandas dataframe to be processed
    destructive : bool
        If set to True, will make changes directly to the dataframe which
        may be useful with very large dataframes instead of making a copy.

    '''

    if destructive is False:
        data = data.copy(deep=True)

    return data.dropna()
