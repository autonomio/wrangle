from keras.models import Sequential
from keras.layers import Dense

from .create_synth_data import create_synth_data


def _base_for_model(mode, n=50, neurons=50):
    
    x, y = create_synth_data(mode, n=n)
    model = Sequential()
    model.add(Dense(neurons,
                    input_dim=x.shape[1],
                    activation='relu'))

    return x, y, model

    
def multi_class_model(n=50, neurons=50):

    x, y, model  = _base_for_model('multi_class', n=n, neurons=neurons)
    
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x, y)

    return model


def regression_model(n=50, neurons=50):

    x, y, model  = _base_for_model('regression', n=n, neurons=neurons)
        
    model.add(Dense(1))
    model.compile(optimizer='sgd', loss='mean_absolute_percentage_error')
    model.fit(x, y)

    return model


def multi_label_model(n=50, neurons=50):

    x, y, model  = _base_for_model('multi_label', n=n, neurons=neurons)
    
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(x, y)

    return model


def binary_model(n=50, neurons=50):

    x, y, model  = _base_for_model('binary', n=n, neurons=neurons)
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x, y)

    return model