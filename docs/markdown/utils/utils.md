### create_synth_binary_model


```python
keras.utils.create_synth_binary_model(neurons=50)
```

----

### create_synth_data


```python
keras.utils.create_synth_data(n='binary', features=1000, classes=20)
```


Create a 2-dimensional dataset in one of the four formats.

mode : string
'binary', 'multi_class', 'multi_label', or 'continuous'
n : int
Number of samples / observations in the data (i.e. rows)
features : int
Number of columns / features for x data
classes : int (only multi_label and multi_class)
Number of unique classes in the labels (prediction classes)

----

### create_synth_multi_class_model


```python
keras.utils.create_synth_multi_class_model(neurons=50)
```

----

### create_synth_multi_label_model


```python
keras.utils.create_synth_multi_label_model(neurons=50)
```

----

### create_synth_regression_model


```python
keras.utils.create_synth_regression_model(neurons=50)
```

----

### create_time_sequence


```python
keras.utils.create_time_sequence(start_year, start_month, start_date=1, time_format='%m-%Y', period='month')
```


Creates a time sequence and outputs a list with the string
timestamps.

periods | int | Number of periods/timestamps in the output
start_year | int | The year to start the periods on
start_month | int | The month to start the periods on
start_date | int | The day to start the periods on
time_format | str | A python datetime string format
period | str | 'month', 'week', or 'day'


----

### read_large_csv


```python
keras.utils.read_large_csv(n, cols=None, chunks=None, dtype='float64')
```


Handles reading a very large CSV file in
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
