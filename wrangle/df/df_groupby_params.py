def df_groupby_params(data,
                      metric,
                      dims=4,
                      gram_type='ngram',
                      group_by_func='sum'):

    '''Takes in a dataframe where each column contains
    parameter labels, and then creates combinations of the
    parameters based on ngram, skipgram, or flexgram approach.

    data | DataFrame | contains the columns with parameters
    dims | int | number of parameter dimensions in the output
    gram_type | str | 'ngram', 'skipgram', 'flexgram', 'none'
    group_by_func | str | one of the supported groupby functions

    '''

    import pandas as pd
    import signs

    from .df_restructure_values import df_restructure_values
    docs = df_restructure_values(data.drop(metric, 1), 'tuple').values

    out = []

    for i in range(len(docs)):

        model = signs.Grams(docs[i])

        if gram_type == 'flexgram':
            grams = model.flexgrams(dims + 1)

        elif gram_type == 'skipgram':
            grams = model.skipgrams(dims)

        elif gram_type == 'ngram':
            grams = model.ngrams(dims)

        elif gram_type == 'none':
            grams = docs[i]

        for ii in range(len(grams)):
            out.append([grams[ii], data[metric].values[i]])

    final = pd.DataFrame(out)
    final[0] = final[0].map(tuple)

    from .df_to_groupby import df_to_groupby
    return df_to_groupby(final, 0, group_by_func).sort_values(1, ascending=False)
