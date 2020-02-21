def df_parallelize_process(data, func, threads=16):
    
    '''Parallelize data processing on a dataframe.
    
    data | DataFrame | a pandas dataframe with the data
    func | function | the function to be applied onto the input data
    threads | int | number of threads to be used (max 2x your cores)
    '''
    
    from numpy import array_split
    from pandas import concat
    from multiprocessing import Pool
    
    df_split = array_split(data, threads)
    pool = Pool(threads)
    df = concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df
