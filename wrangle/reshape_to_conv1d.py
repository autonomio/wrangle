import numpy as np
from keras.utils import to_categorical

def reshape_to_conv1d(x):
    
    '''Reshapes regulard 2d array for Conv1D layer'''

    out = x.reshape(x.shape[0], x.shape[1], 1)
    
    return out