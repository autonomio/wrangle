import pandas as pd


def convert_to_numeric(data):
    
    '''Converts strings to binary dummies and 
    numbers to floats. '''
    
    data = data.copy(deep=True)
    
    data = pd.get_dummies(data)

    for col in data.columns:
        try: 
            data[col] = data[col].astype(float)
        except ValueError:
            pass
        
    return data