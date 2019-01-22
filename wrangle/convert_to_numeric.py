import pandas as pd


def convert_to_numeric(data):
    
    '''Converts strings to binary dummies and 
    numbers to floats. '''
    
    data = data.copy(deep=True)

    for col in data.columns:
        try: 
            data[col] = data[col].astype(float)
        except ValueError:
            pass

    data = pd.get_dummies(data)
        
    return data