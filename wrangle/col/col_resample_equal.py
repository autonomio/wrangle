import pandas as pd


def col_resample_equal(data, col, sample_size):

    '''Resamples based on unique labels in a column.
    Results in a dataframe with equal number of rows
    per label in a given column.
    '''
    new_data = pd.DataFrame()

    for col_label in data[col].unique():
        sample = data[data[col] == col_label].sample(sample_size)
        new_data = new_data.append(sample)

    return new_data
