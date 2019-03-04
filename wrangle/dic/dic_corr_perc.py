def dic_corr_perc(data, y):

    """Performans a cut/bucket based correlation of values of all columns
    across multiple dataframes stored in a dictionary.

    USE: dict_corr_perc(dfs, y)

    data : dict
        A dictionary where each key corresponds to a Pandas DataFrame.
    y : str
        A shared outcome feature that is available in all the dataframes.

    """

    import pandas as pd
    import wrangle as wr

    final = pd.DataFrame()

    for label in list(data.keys()):
        for col in data[label].columns:
            if col != y:

                # avoid destruction
                buckets = wr.col_to_buckets(data[label], col)
                temp_df = pd.DataFrame({col: buckets, y: data[label][y]})

                # create a temp dataframe with correlation values
                temp_corrs = wr.col_corr_category(temp_df, col, y)
                temp_corrs = temp_corrs.reset_index()
                temp_corrs.columns = ['index', y]

                # create a temp dataframe with sample counts
                temp_count = temp_df[col].value_counts().reset_index()
                temp_count.columns = ['index', 'samples']
                temp_count['index'] = temp_count['index'].astype('O')

                # combine correlation values and sample counts
                out = pd.merge(temp_corrs, temp_count,
                               left_on='index', right_on='index')

                # add handle columns
                out['metric'] = col
                out['metric_context'] = label

                # append to the output dataframe
                final = final.append(out)

    return final
