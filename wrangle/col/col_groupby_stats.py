def col_groupby_stats(data, col, y):

    """Takes in category column and returns
    descriptive statistics based on an
    binary outcome feature.

    data :
    col :
    y :

    """

    import pandas as pd
    import wrangle as wr

    data = data.copy(deep=True)
    data = data[[col, y]]

    a = data.groupby(col).count()
    b = data.groupby(col).sum()
    d = data.groupby(col).mean()
    e = data.groupby(col).std()

    a.columns = ['n']
    b.columns = ['sum']
    d.columns = ['mean']
    e.columns = ['std']

    out = pd.merge(a, b, left_index=True, right_index=True)

    out = wr.df_merge(a, b)
    out = wr.df_merge(d, out)
    out = wr.df_merge(e, out)
    out = out.round(2)

    return out
