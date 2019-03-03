def read_large_csv(file_path, n, cols=None, chunks=None, dtype='float64'):

    '''Handles reading a very large CSV file in
    a highly memory efficient way. Returns a numpy
    array.

    file_path : str
        Location to the file either local or remote.
    n : int
        Number of samples to be drawn from the file.
    cols : None or list
        Accepts a list of columns to be drawn from the file.
    chunks : None or int
        Unless a value is provided, 1/100 of n will be set as
        a the chunksize.
    dtype : str
        A numpy datatype. Default is 'float64'.
    '''

    import gc
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    if chunks is None:
        chunks = int(n / 100)

    if cols is None:
        cols = pd.read_csv(file_path, nrows=2)
        cols_n = cols.shape[1]
        cols = cols.columns
    else:
        cols_n = len(cols)

    out = np.zeros([n, cols_n], dtype)

    start = 0
    end = chunks

    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunks, nrows=n)):

        out[start:end] = chunk[cols].values

        start += chunks
        end += chunks

        gc.collect()

    return out
