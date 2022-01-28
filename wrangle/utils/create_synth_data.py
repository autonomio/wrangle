def create_synth_data(mode='binary', n=1000, features=20, classes=4):

    '''Create a 2-dimensional dataset in one of the four formats.

    mode : string
        'binary', 'multi_class', 'multi_label', or 'continuous'
    n : int
        Number of samples / observations in the data (i.e. rows)
    features : int
        Number of columns / features for x data
    classes : int (only multi_label and multi_class)
        Number of unique classes in the labels (prediction classes)
    '''

    import numpy as np
    import sklearn.datasets as data

    if mode == 'binary':
        x, y = np.random.rand(n, features), np.random.randint(0, 2, n)

    elif mode == 'multi_class':
        x, y = np.random.rand(n, features), np.random.randint(0, classes, n)

    elif mode == 'multi_label':
        x, y = data.make_multilabel_classification(n_samples=n,
                                                   n_features=features, 
                                                   n_classes=classes)

    elif mode == 'continuous':
        x, y = data.make_regression(n, features)

    return x, y
