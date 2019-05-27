def col_corr_category(data,
                      col,
                      outcome,
                      rounding=2,
                      warning_threshold=40):

    '''Compute likeliness of a given
    outcome against a set of class
    attributes.

    USE: wr.col_corr_cut(first, 'BUN_groups', 'hospital_mortality')

    data : pandas DataFrame
    col : str
        A column with multilabel values
    outcome : str
        A column with binary class values
    rounding : int
        Number of decimals to be used for rounding.
    warning_threshold : int
        Number of samples required to avoid warning.
        Warning is expressed as ** following the class
        name.

    '''

    import pandas as pd

    a = data[[outcome, col]].groupby(col).sum()
    b = pd.DataFrame(data[[outcome, col]][col].value_counts())
    out = a.merge(b, left_index=True, right_index=True)
    out = round((out[outcome] / out[col]) * 100, rounding)

    out = pd.DataFrame(out)
    out.columns = [outcome]

    return out
