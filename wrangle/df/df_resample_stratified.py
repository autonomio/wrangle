def df_resample_stratified(data, strat_col, n=None):

    if n is not None:
        n = data[strat_col].value_counts().values[-1]

    temp = data.groupby(strat_col, group_keys=False)
    return temp.apply(lambda x: x.sample(min(len(x), n)))
