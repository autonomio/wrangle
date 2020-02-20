def df_print_values(data, n=5):
    
    '''Shows the most common values for each column
    in a dataframe.
    
    data | DataFrame | a pandas dataframe with the data
    n | int | Number of most common values to show
    '''

    for col in data.columns:
        
        length = len(col)
        dashes = "-" * length
        
        print("%s \n%s" % (col, dashes))
        print(data[col].value_counts().head(n))
        print("\n")
