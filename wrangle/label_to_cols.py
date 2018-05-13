from pandas import get_dummies, merge


def label_to_cols(df, col):

    df = df.copy(deep=True)
    temp_cols = get_dummies(df[col])
    df = df.drop(col, axis=1)

    return merge(df, temp_cols, left_index=True, right_index=True)
