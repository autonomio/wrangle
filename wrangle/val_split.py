def val_split(x, y, split):

    split = int(len(x) * (1 - split)) 

    # separate into validation and train
    x_train = x[:split]
    y_train = y[:split]

    x_val = x[split:]
    y_val = y[split:]
    
    return x_train, y_train, x_val, y_val